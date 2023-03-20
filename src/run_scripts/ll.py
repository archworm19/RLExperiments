"""Parallelism Models"""
import gymnasium as gym
import numpy as np
import pickle
import numpy.random as npr
from typing import List, Callable
from multiprocessing import connection, Queue, Pipe, Process
from functools import partial
from frameworks.agent import Agent
from replay_buffers.async_storage import TrajectoryStore
from run_scripts.builders import EnvConfigCont
from run_scripts.runner import simple_run


####
# Asynchronous run model
#   each environment runs in a different subprocess
#   only master process runs model
#
#   this approach makes sense if your model is huge
# TODO: there appears to be something wrong here!
#       model performance much worse in async
# what could be the problems?
# > wrong storage / trajectory data handling:
#       I've tested this pretty thoroughly and don't see any issues
# > run_step and master are communicating improperly
#       there might be some synchronicity issues here
#       ... performance was helped by syncing at beginning of epoch
#   TODO: potentially, simpler system
#       > run_step models epochs as well
#           each sub-process collects fixed number of points each epoch
#           explicitly handle initiation action vs. selection action


# TODO: subprocesses should request what they need!!!


def _async_single_run(pid: int,
                      env: gym.Env,
                      rng: npr.Generator,
                      conn: connection.Connection,
                      queue: Queue,
                      run_cutoff: int) -> int:
    env.reset(seed=int(rng.integers(10000)))

    # request initial action
    # TODO: comms? (pid, request_init, request_response, request_save, request_end, data)
    queue.put(pickle.dumps([pid, True, False, False, False, []]), 
              block=True)
    # wait for initial action --> send 0 state
    (init_bit, action) = pickle.loads(conn.recv())
    assert init_bit  # TODO: TESTING
    cur_state = env.step(action)[0]

    # world model: s_t + a_t --> r_t, s_{t+1}
    save_states = [cur_state]
    save_actions, save_rewards = [], []
    terminated = False
    for _ in range(run_cutoff):
        # request response
        # TODO: comms? (pid, request_init, request_response, request_save, request_end, data)
        queue.put(pickle.dumps([pid, False, True, False, False, cur_state]), 
                  block=True)

        # wait for action selection
        (init_bit, action) = pickle.loads(conn.recv())
        assert not init_bit  # TODO: TESTING

        # step
        step_output = env.step(action)
        cur_state = step_output[0]
        reward = step_output[1]

        # saves
        save_states.append(cur_state)
        save_actions.append(action)
        save_rewards.append(reward)

        if step_output[2]:
            terminated = True
            break
    # TODO: comms? (pid, run_end, epoch_end, data)
    save_states = np.array(save_states)
    save_actions = np.array(save_actions)
    save_rewards = np.array(save_rewards)

    # TODO: comms? (pid, request_init, request_response, request_save, request_end, data)
    queue.put(pickle.dumps([pid, False, False, True, False,
                            [save_states, save_actions, save_rewards, terminated]]), 
              block=True)
    # return number of tpts
    return len(save_rewards)


def _async_subproc(pid: int,
                   env_spec: EnvConfigCont,
                   rand_seed: int,
                   conn: connection.Connection,
                   queue: Queue,
                   num_epoch: int,
                   steps_per_epoch: int,
                   run_cutoff: int):
    # TODO: this needs to wrap _async_single_run
    # TODO: docstring
    # NOTE: subprocess is managing the memory now
    env = gym.make(env_spec.env_name, **env_spec.kwargs)
    rng = npr.default_rng(rand_seed)
    env.reset(seed=int(rng.integers(10000)))
    for _ in range(num_epoch):
        tot_steps = 0
        while tot_steps < steps_per_epoch:
            num_step = _async_single_run(pid, env, rng, conn, queue, run_cutoff)
            tot_steps += num_step
        # send "end epoch" signal
        # TODO: comms? (pid, request_init, request_response, request_save, request_end, data)
        queue.put(pickle.dumps([pid, False, False, False, True, []]), 
                  block=True)


def _launch_envs(num_procs: int,
                 env_spec: EnvConfigCont,
                 num_epoch: int,
                 steps_per_proc: int,
                 run_cutoff: int):
    # set up the processes:
    procs, conns = [], []
    q = Queue(maxsize=1000)  # TODO: meaning of maxsize? (in bytes? in elems?)
    for i in range(num_procs):
        parent_conn, child_conn = Pipe()
        func_target = partial(_async_subproc,
                              pid=i,
                              env_spec=env_spec,
                              rand_seed=i,
                              conn=child_conn,
                              queue=q,
                              num_epoch=num_epoch,
                              steps_per_epoch=steps_per_proc,
                              run_cutoff=run_cutoff)
        p = Process(target=func_target)
        p.start()
        procs.append(p)
        conns.append(parent_conn)
    return procs, conns, q


def _close_envs(conns: List[connection.Connection],
                procs: List[Process]):
    for c, p in zip(conns, procs):
        print("Closing")
        p.join()  # blocks until subprocess terminates


def async_master(num_procs: int,
                 env_spec: EnvConfigCont,
                 agent: Agent,
                 num_epoch: int,
                 steps_per_proc: int,
                 run_cutoff: int,
                 weight_directory: str,
                 state_name: str = "core_state",
                 load_pretrained: bool = False):
    if load_pretrained:
        agent.load_weights(weight_directory)
    # ordering = [state, action, reward, termination]
    rng = npr.default_rng(42)
    env_test = gym.make(env_spec.env_name, **env_spec.kwargs, render_mode="human")
    procs, conns, queue = _launch_envs(num_procs, env_spec, num_epoch, steps_per_proc, run_cutoff)
    for _ in range(num_epoch):
        end_count = 0
        sv, av, rv, tv = [], [], [], []
        while end_count < num_procs:
            # TODO: option to pull multiple elements from the queue?
            # TODO: comms? (pid, request_init, request_response, request_save, request_end, data)
            (pid, req_init, req_resp, req_save, req_end, dat) = pickle.loads(queue.get(block=True))
            if req_init:
                new_act = agent.init_action()[0]
                conns[pid].send(pickle.dumps((True, new_act)))
            if req_resp:
                state = dat
                new_act = agent.select_action({state_name: state[None]}, False, False)[0]
                conns[pid].send(pickle.dumps((False, new_act)))
            if req_save:
                [save_states, save_actions, save_rewards, terminated] = dat
                sv.append(save_states)
                av.append(save_actions)
                rv.append(save_rewards)
                tv.append(terminated)
            if req_end:
                end_count += 1

        # train:
        agent.train([{state_name: si} for si in sv],
                    rv,
                    av,
                    np.array(tv))

        # save model
        agent.save_weights(weight_directory)

        # test:
        env_test.reset(seed=int(rng.integers(10000)))
        _, _, test_rew, _ = simple_run(env_test, agent, run_cutoff, False, False)
        print("Test performance: " + str(np.sum(test_rew)))

    _close_envs(conns, procs)
    env_test.close()


####
# Synchronous run model
#   each subprocess gets most recent model copy --> 
#       runs model in an env
#   master process performs training


def _sync_run(conn: connection.Connection,
              queue: Queue,
              builder: Callable,  # --> env_run and agent
              weight_directory: str,
              T_run: int,
              run_cutoff: int,
              rand_seed: int,
              discrete_mode: bool = True):
    # process function:
    # > builds
    # > waits for signal --> load weights from directory --> adds run data to queue
    # how to tell parent that child is done? send empty dictionary
    #       real data sent as pickle(Dict[str, np.ndarray])
    rng = npr.default_rng(rand_seed)
    env_run, _, agent = builder()

    while True:
        run_sig = conn.recv()
        if not run_sig:
            break
        agent.load_weights(weight_directory)
        num_step = 0
        while(num_step < T_run):
            env_run.reset(seed=int(rng.integers(10000)))
            states, actions, rewards, term = simple_run(env_run, agent, run_cutoff, debug=False, discrete=discrete_mode)
            num_step += np.shape(rewards)[0]

            # send current trajectory through the queue
            d = {"states": np.array(states), "actions": np.array(actions),
                 "rewards": np.array(rewards), "term": term}
            queue.put(pickle.dumps(d), block=True)            
        # finished signal = empty dict thru queue:
        queue.put(pickle.dumps({}), block=True)


def _sync_launch_procs(num_procs: int,
                       builder: Callable,
                       weight_directory: str,
                       T_run: int,
                       run_cutoff: int,
                       discrete_mode: bool):
    procs, conns = [], []
    q = Queue(maxsize=100)  # TODO: meaning of maxsize? (in bytes? in elems?)
    for i in range(num_procs):
        parent_conn, child_conn = Pipe()
        func_target = partial(_sync_run,
                              conn=child_conn,
                              queue=q,
                              builder=builder,
                              weight_directory=weight_directory,
                              T_run=T_run,
                              run_cutoff=run_cutoff,
                              rand_seed=i,
                              discrete_mode=discrete_mode)
        p = Process(target=func_target)
        p.start()
        procs.append(p)
        conns.append(parent_conn)
    return procs, conns, q


def sync_master(builder: Callable,  # --> env_run, env_viz, agent
                num_procs: int,
                weight_directory: str,
                num_epoch: int,
                T_run: int,  # per process
                run_cutoff: int,
                rand_seed: int,
                discrete_mode: bool = True,
                viz_debug: bool = False,
                load_pretrained: bool = False):
    # NOTE: if load_pretrained --> start from current weights in weight_directory
    rng = npr.default_rng(rand_seed)
    # set up the processes:
    procs, conns, q = _sync_launch_procs(num_procs, builder, weight_directory,
                                         T_run, run_cutoff, discrete_mode)
    # build agent for training:
    _, env_viz, agent = builder()
    if load_pretrained:
        agent.load_weights(weight_directory)
    else:
        agent.save_weights(weight_directory)

    # run and train for num_epoch
    for _ in range(num_epoch):

        # run
        for conn in conns:
            conn.send(True)  # means "run it"

        # pull from queue till all processes have sent stop signal
        num_stops = 0
        sv, av, rv, tv = [], [], [], []
        r_fin = []
        while num_stops < num_procs:
            d = pickle.loads(q.get(block=True))
            # check for finished signal
            if len(d) < 1:
                num_stops += 1
                continue
            sv.append(d["states"])
            av.append(d["actions"])
            rv.append(d["rewards"])
            tv.append(d["term"])
            if d["term"] or len(d["rewards"]) == run_cutoff:
                r_fin.append(np.sum(d["rewards"]))
        print("Average Reward For Terminated Runs: " + str(np.mean(np.array(r_fin))))
        print("Max Reward For Terminated Runs: " + str(np.amax(np.array(r_fin))))

        # train and test:
        agent.train([{"core_state": si} for si in sv],
                    rv,
                    av,
                    np.array(tv))

        # save agent weights
        agent.save_weights(weight_directory)
        # test with viz
        if viz_debug:
            env_viz.reset(seed=int(rng.integers(10000)))
            _, _, reward, _ = simple_run(env_viz, agent, run_cutoff, debug=viz_debug, discrete=discrete_mode)
            print("Reward Test: " + str(np.sum(reward)))

    for p, conn in zip(procs, conns):
        conn.send(False)
        p.join()
