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

def run_step_continuous(pid: int,
                        env_spec: EnvConfigCont,
                        rand_seed: int,
                        conn: connection.Connection,
                        queue: Queue,
                        max_run_length: int):
    """subprocess runstep
        waits for action -->
            writes [pid, new_state, reward, termination bit, end bit] to queue
                =  [pid, s(t+1), r(t), term(t), ...]
        NOTE: termination bit and end bit are subtly different
            if env reaches max_run_length without termination
                --> termination bit = False, end bit = True

    Args:
        pid (int): process id
        env_spec (EnvConfigCont): environment specification
        rand_seed (int): random seed
        conn (connection.Connection): multiprocessing connection (input)
        queue (Queue): multiprocessing queue (output)
        max_run_length (int): max run length per epoch
    """
    env = gym.make(env_spec.env_name, **env_spec.kwargs)
    rng = npr.default_rng(rand_seed)
    env.reset(seed=int(rng.integers(10000)))
    rl = 0
    while True:
        action = pickle.loads(conn.recv())
        if len(action) == 0:  # stop signal
            break
        step_output = env.step(action)
        new_state = step_output[0]
        reward = step_output[1]
        term = step_output[2]
        rl += 1
        end_bit = term or (rl >= max_run_length)
        queue.put(pickle.dumps([pid, new_state, reward, term, end_bit]), block=True)
        if end_bit:
            env.reset(seed=int(rng.integers(10000)))
            rl = 0
    env.close()


def _launch_envs(num_procs: int,
                 env_spec: EnvConfigCont,
                 max_run_length: int):
    # set up the processes:
    procs, conns = [], []
    q = Queue(maxsize=100)  # TODO: meaning of maxsize? (in bytes? in elems?)
    for i in range(num_procs):
        parent_conn, child_conn = Pipe()
        func_target = partial(run_step_continuous,
                              pid=i,
                              env_spec=env_spec,
                              rand_seed=i,
                              conn=child_conn,
                              queue=q,
                              max_run_length=max_run_length)
        p = Process(target=func_target)
        p.start()
        procs.append(p)
        conns.append(parent_conn)
    return procs, conns, q


def _close_envs(conns: List[connection.Connection],
                procs: List[Process]):
    for c, p in zip(conns, procs):
        print("Closing")
        # send stop signal + wait for response
        c.send(pickle.dumps([]))
        p.join()  # blocks until subprocess terminates


# TODO: save model periodically...
def runtrain_onpolicy(num_procs: int,
                      env_spec: EnvConfigCont,
                      agent: Agent,
                      num_epoch: int,
                      epoch_run_length: int,
                      run_cutoff: int,
                      state_name: str = "core_state"):
    # ordering = [state, action, reward, termination]
    rng = npr.default_rng(42)
    env_test = gym.make(env_spec.env_name, **env_spec.kwargs, render_mode="human")
    dims = [env_spec.dims_obs, len(env_spec.action_bounds), 1, 1]
    procs, conns, queue = _launch_envs(num_procs, env_spec, run_cutoff)
    for _ in range(num_epoch):
        TS = TrajectoryStore(num_procs, epoch_run_length, dims)
        # seed with actions:
        for cn in conns:
            cn.send(pickle.dumps(agent.init_action()[0]))
        for _ in range(epoch_run_length):
            # TODO: option to pull multiple elements from the queue?
            [pid, new_state, reward, term, end_bit] = pickle.loads(queue.get(block=True))
            print(pid)
            print(new_state)
            print(reward)
            print(term)
            print(end_bit)
            input("cont?")
            if term:
                action = agent.init_action()[0]
            else:
                action = agent.select_action({state_name: new_state[None]}, False, False)[0]
            conns[pid].send(pickle.dumps(action))
            TS.add_datapt(pid, [new_state, action, reward, term*1], end_bit)
        [tr_states, tr_actions, tr_rewards, tr_terms] = TS.pull_trajectories()

        agent.train([{state_name: si} for si in tr_states],
                    [ri[1:,0] for ri in tr_rewards],  # TODO: is this definitely right?
                    [ai[1:] for ai in tr_actions],
                    [ti[-1, 0] > 0.5 for ti in tr_terms])

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

if __name__ == "__main__":
    from run_scripts.builders import  EnvsContinuous, build_continuous_ppo

    # async testing
    # _, _, agent = build_continuous_ppo(EnvsContinuous.pendulum, gamma=0.95,
    #                                                learning_rate=.0001,
    #                                                layer_sizes=[256, 128, 64],
    #                                                embed_dim=16,
    #                                                entropy_scale=0.0, eta=0.3)
    # test_T = 500
    # runtrain_onpolicy(4, EnvsContinuous.pendulum.value, agent, 100, int(10*test_T), test_T)

    # sync testing
    builder = partial(build_continuous_ppo, env=EnvsContinuous.pendulum, learning_rate=.0001,
                      layer_sizes=[256, 128, 64], embed_dim=16,
                      entropy_scale=0.0, eta=0.3)
    test_T = 500
    sync_master(builder, 2, "weights/", 100, int(test_T * 5), test_T, 42, False, True, load_pretrained=True)
