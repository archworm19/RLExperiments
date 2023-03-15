"""PPO"""
import pickle
import numpy as np
import numpy.random as npr
from typing import Callable
from multiprocessing import connection, Process, Pipe, Queue
from functools import partial
from run_scripts.runner import simple_run, run_step_continuous
from run_scripts.builders import EnvsDiscrete, EnvsContinuous, build_discrete_ppo, build_continuous_ppo


def run_and_train(env_run, env_viz, agent,
                  num_actions: int,
                  num_epoch: int,
                  T_run: int,
                  T_test: int,
                  run_cutoff: int,
                  viz_debug: bool = False,
                  discrete_mode: bool = True):
    # total train timepoints = T_run
    #   cutoff runs after [run_cutoff] steps
    # NOTE: applies one-hot to actions in discrete mode
    # T_test = max run length for visual env test
    rng = npr.default_rng(42)
    for _ in range(num_epoch):
        sv, av, rv, tv = [], [], [], []
        num_step = 0
        while(num_step < T_run):
            env_run.reset(seed=int(rng.integers(10000)))
            print("pre run")
            states, actions, rewards, term = simple_run(env_run, agent, run_cutoff, debug=False, discrete=discrete_mode)
            print("post run")
            sv.append(np.array(states))
            av.append(np.array(actions))
            rv.append(np.array(rewards))
            tv.append(term)
            num_step += np.shape(rv[-1])[0]
        r_fin = []
        for tvi, rvi in zip(tv, rv):
            if tvi or (len(rvi) == run_cutoff):
                r_fin.append(np.sum(rvi))
        print("Average Reward For Terminated Runs: " + str(np.mean(np.array(r_fin))))
        print("Max Reward For Terminated Runs: " + str(np.amax(np.array(r_fin))))
        agent.train([{"core_state": si} for si in sv],
                    rv,
                    av,
                    tv)
        # test with viz
        env_viz.reset(seed=int(rng.integers(10000)))
        _, _, reward, _ = simple_run(env_viz, agent, T_test, debug=viz_debug, discrete=discrete_mode)
        print(np.sum(reward))


# TODO: how to design this multiprocessing system to allow for relaunching of paused processes
# > I think _ll_run_queue can remain as is
# > should have a launcher for individual processes
# > keep track of last time since last pull
#       would need pids to come in with data
#       TODO: how to close process?
#           use join but that won't solve the thing being stalled...


def _ll_run_queue(conn: connection.Connection,
                  queue: Queue,
                  builder: Callable,  # --> env_run and agent
                  weight_directory: str,
                  T_run: int,
                  run_cutoff: int,
                  rand_seed: int,
                  discrete_mode: bool = True,
                  timeout: int = 30):  # in seconds
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


def ll_run_and_train_queue(builder: Callable,  # --> env_run, env_viz, agent
                           num_procs: int,
                           weight_directory: str,
                           num_epoch: int,
                           T_run: int,  # per process
                           T_test: int,
                           run_cutoff: int,
                           rand_seed: int,
                           discrete_mode: bool = True,
                           viz_debug: bool = False):
    rng = npr.default_rng(rand_seed)
    # set up the processes:
    procs, conns = [], []
    q = Queue(maxsize=100)  # TODO: meaning of maxsize? (in bytes? in elems?)
    for i in range(num_procs):
        parent_conn, child_conn = Pipe()
        func_target = partial(_ll_run_queue,
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
    # build agent for training:
    _, env_viz, agent = builder()
    # save weights
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
            if d["term"] or len(d["rewards"]) == T_test:
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
            _, _, reward, _ = simple_run(env_viz, agent, T_test, debug=viz_debug, discrete=discrete_mode)
            print("Reward Test: " + str(np.sum(reward)))

    for p, conn in zip(procs, conns):
        conn.send(False)
        p.join()


# TODO: simpler process model
#   envs run in processes and model runs in main process
#   makes sense if model is super beefy


# class SeqDataStore:
#     # TODO: this needs a ton of testing...

#     def __init__(self, num_procs: int, T: int, state_dims: int, action_dims: int):
#         # TODO: only works with single state for the moment
#         self.T = T
#         self.state_dims = state_dims
#         self.action_dims = action_dims
#         self._dat_state = np.zeros((num_procs, T, state_dims), dtype=np.float32)
#         self._dat_action = np.zeros((num_procs, T, action_dims), dtype=np.float32)
#         # trail tuple for each process id = (t0, cur_t, has trial terminated yet?)
#         self._ptrs = [[-1, -1, False] for _ in range(num_procs)]
#         self._old_ptrs = []  # (pid, t0, tend, terminated?)

#     def add_datapt(self, process_idx: int, state, action, termination_bit):
#         ptr = self._ptrs[process_idx]
#         # check if current ptr has terminated
#         if ptr[2]:
#             self._old_ptrs.append([process_idx] + ptr)
#             # add new info under new ptr
#             new_ptr = [ptr[1] + 1, ptr[1] + 1, termination_bit]
#             self._dat_state[process_idx, ptr[1] + 1] = state
#             self._dat_action[process_idx, ptr[1] + 1] = action
#             self._ptrs[process_idx] = new_ptr
#         else:
#             self._dat_state[process_idx, ptr[1] + 1] = state
#             self._dat_action[process_idx, ptr[1] + 1] = action
#             ptr[1] += 1
#             ptr[2] = termination_bit

#     def pull_data(self):
#         cptrs = [[i] + self._ptrs[i] for i in range(self._ptrs)]
#         ret_states, ret_actions, terms = [], [], []
#         for ptr in self._old_ptrs + cptrs:
#             pid = ptr[0]
#             ret_states.append(self._dat_state[pid, ptr[1]:ptr[2]+1])
#             ret_actions.append(self._dat_action[pid, ptr[1]:ptr[2]+1])
#             terms.append(ptr[3])
#         return ret_states, ret_actions, terms
            

# def _start_procs(env_builder: Callable, num_procs: int, max_run_length: int):
#     # starts up processes running run_step_continuous
#     q = Queue(maxsize=100)  # TODO: meaning of maxsize? (in bytes? in elems?)
#     procs, conns = [], []
#     for i in range(num_procs):
#         parent_conn, child_conn = Pipe()
#         func_target = partial(run_step_continuous,
#                               pid=i,
#                               env=env_builder(),
#                               rng=npr.default_rng(i),
#                               conn=child_conn,
#                               queue=q,
#                               max_run_length=max_run_length)
#         p = Process(target=func_target)
#         p.start()
#         procs.append(p)
#         conns.append(parent_conn)
#     return procs, conns, q


# TODO: master runner
# > pull from queue --> get pid
# > add to active run for given pid ~ handle termination
# > select action --> send through connection[pid]
# > train + flush all buffers at end of epoch
# > TODO: resetting system?
#       timeout on queue --> queue.Empty exception --> close em --> spin up new procs


if __name__ == "__main__":
    # env_run, env_viz, agent = build_discrete_ppo(EnvsDiscrete.cartpole)
    # num_actions = EnvsDiscrete.cartpole.value.dims_actions
    # env_run, env_viz, agent = build_discrete_ppo(EnvsDiscrete.acrobot)
    # num_actions = EnvsDiscrete.acrobot.value.dims_actions
    # env_run, env_viz, agent = build_discrete_ppo(EnvsDiscrete.lunar, entropy_scale=0.01, embed_dim=16, layer_sizes=[128, 64])
    # num_actions = EnvsDiscrete.lunar.value.dims_actions
    # discrete_mode = True

    # continuous
    # env_run, env_viz, agent = build_continuous_ppo(EnvsContinuous.pendulum, gamma=0.95,
    #                                                learning_rate=.0001,
    #                                                layer_sizes=[256, 128, 64],
    #                                                entropy_scale=0.0, eta=0.3)
    # num_actions = len(EnvsContinuous.pendulum.value.action_bounds)
    # env_run, env_viz, agent = build_continuous_ppo(EnvsContinuous.lunar_continuous,
    #                                                gamma=0.95,
    #                                                learning_rate=.0001,
    #                                                entropy_scale=0.0, eta=0.3,
    #                                                layer_sizes=[256, 128, 64])
    # num_actions = len(EnvsContinuous.lunar_continuous.value.action_bounds)
    env_run, env_viz, agent = build_continuous_ppo(EnvsContinuous.bi_walker,
                                                   gamma=0.95,
                                                   learning_rate=.00005,
                                                   train_batch_size=64,
                                                   train_epoch=10,
                                                   entropy_scale=0.0, eta=0.15,
                                                   embed_dim=64,
                                                   layer_sizes=[256, 128, 64],
                                                   scale_std_dev=.05)
    num_actions = len(EnvsContinuous.bi_walker.value.action_bounds)
    discrete_mode = False

    run_and_train(env_run, env_viz, agent, num_actions, 400, 10000, 1000, 2500, viz_debug=True, discrete_mode=discrete_mode)

    # parallel + queue
    # builder = partial(build_discrete_ppo, env=EnvsDiscrete.cartpole)
    # builder = partial(build_discrete_ppo, env=EnvsDiscrete.lunar, entropy_scale=1.0, lam=0.95)
    # ll_run_and_train_queue(builder, 4, 'weights/', 50, 4000, 500, 500, 42, discrete_mode=True, viz_debug=False)
    # builder = partial(build_continuous_ppo, env=EnvsContinuous.pendulum, learning_rate=.0001,
    #                         entropy_scale=0.0, eta=0.3)
    # builder = partial(build_continuous_ppo, EnvsContinuous.lunar_continuous, learning_rate=.0001,
    #                         entropy_scale=0.0, eta=0.15, layer_sizes=[256, 128, 64])
    # builder = partial(build_continuous_ppo, EnvsContinuous.bi_walker,
    #                         gamma=0.95, lam=0.95,
    #                         learning_rate=.0001,
    #                         entropy_scale=0.0, eta=0.2, layer_sizes=[512, 256, 128],
    #                         min_var=0.05, max_var=0.2, init_var=0.2,
    #                         train_batch_size=128)
    # ll_run_and_train_queue(builder, 4, 'weights/', 800, 4000, 1000, 1000, 42, discrete_mode=False, viz_debug=False)
