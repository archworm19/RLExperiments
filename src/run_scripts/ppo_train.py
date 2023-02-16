"""PPO"""
import os
import numpy as np
import numpy.random as npr
from typing import Callable
from multiprocessing import connection, Process, Pipe
from functools import partial
from run_scripts.runner import simple_run
from run_scripts.builders import EnvsDiscrete, EnvsContinuous, build_discrete_ppo, build_continuous_ppo


def _one_hot(x: np.ndarray, num_action: int):
    # --> T x num_action
    v = [1. * (x == i) for i in range(num_action)]
    return np.concatenate([vi[:, None] for vi in v], axis=1)


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
            states, actions, rewards, term = simple_run(env_run, agent, run_cutoff, debug=False, discrete=discrete_mode)
            sv.append(np.array(states))
            av.append(np.array(actions))
            rv.append(np.array(rewards))
            tv.append(term)
            num_step += np.shape(rv[-1])[0]
        agent.train([{"core_state": si} for si in sv],
                    rv,
                    av,
                    tv)
        # test with viz
        env_viz.reset(seed=int(rng.integers(10000)))
        _, _, reward, _ = simple_run(env_viz, agent, T_test, debug=viz_debug, discrete=discrete_mode)
        print(np.sum(reward))


# TODO: design for parallelism?
# > child vs. parent process?
#   parent train; children run
#   children copy parent model
# > model sharing
#   use keras save/load
#   interface?
#       save_model()
#       load_model(root_directory: str)
#   if there are multiple models --> model itself handles the internals (i.e. make subdirectories for diff models)
# > we need to make different keras models for ppo (forward/no_train model)
# > TODO: might be better/more efficient to just share weights... cuz graph will remain the same!!
# > data sharing
#   LAZY soln (no generator and potentially inefficient data storage)
#   save each field output of simple run as npz
#       Ex: reward = npz where different arrays are different trajectories
#   parent loads and passes into train
# > parallelism
#   start simple: just use pool
#       pass in model location directory to each process
#       get back the locations of the saved data
#   TODO: if we're sharing weights --> better to keep static pool of processes
#       
# > delete old saved data


def _ll_run(conn: connection.Connection,
            builder: Callable,  # --> env_run and agent
            process_id: int,
            weight_directory: str,
            data_directory: str,
            T_run: int,
            run_cutoff: int,
            rand_seed: int,
            discrete_mode: bool = True):
    # process function:
    # > builds
    # > waits for signal --> load weights from directory --> saves run data
    rng = npr.default_rng(rand_seed)
    env_run, _, agent = builder()

    while True:
        run_sig = conn.recv()

        if not run_sig:
            break

        agent.load_weights(weight_directory)
        sv, av, rv, tv = [], [], [], []
        num_step = 0
        while(num_step < T_run):
            env_run.reset(seed=int(rng.integers(10000)))
            states, actions, rewards, term = simple_run(env_run, agent, run_cutoff, debug=False, discrete=discrete_mode)
            sv.append(np.array(states))
            av.append(np.array(actions))
            rv.append(np.array(rewards))
            tv.append(term)
            num_step += np.shape(rv[-1])[0]
        # overwrite old data:  NOTE: this is dangerous
        np.savez(os.path.join(data_directory, "states{0}.npz".format(process_id)), *sv)
        np.savez(os.path.join(data_directory, "actions{0}.npz").format(process_id), *av)
        np.savez(os.path.join(data_directory, "rewards{0}.npz").format(process_id), *rv)
        np.save(os.path.join(data_directory, "termination{0}.npy").format(process_id), np.array(tv))
        conn.send(True)  # tell parent that run is finished


def _load_npz_helper(rdir: str, fn: str, pid: int):
    d = np.load(os.path.join(rdir, "{0}{1}.npz".format(fn, pid)))
    l = [d["arr_"+str(i)] for i in range(len(d))]
    return l


# TODO: experiment with transmitting data over a queue

def ll_run_and_train(builder: Callable,  # --> env_run, env_viz, agent
                     num_procs: int,
                     weight_directory: str,
                     data_directory: str,
                     num_epoch: int,
                     T_run: int,
                     T_test: int,
                     run_cutoff: int,
                     rand_seed: int,
                     discrete_mode: bool = True,
                     viz_debug: bool = False):
    rng = npr.default_rng(rand_seed)
    # set up the processes:
    procs, conns = [], []
    for i in range(num_procs):
        parent_conn, child_conn = Pipe()
        func_target = partial(_ll_run,
                              conn=child_conn,
                              builder=builder,
                              process_id=i,
                              weight_directory=weight_directory,
                              data_directory=data_directory,
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
        # block till all processes complete:
        for conn in conns:
            conn.recv()

        sv, av, rv, tv = [], [], [], []
        # load data from each child:
        for i in range(num_procs):
            sv.extend(_load_npz_helper(data_directory, "states", i))
            av.extend(_load_npz_helper(data_directory, "actions", i))
            rv.extend(_load_npz_helper(data_directory, "rewards", i))
            tv.extend(np.load(os.path.join(data_directory, "termination{0}.npy").format(i)).tolist())
        # train and test:
        agent.train([{"core_state": si} for si in sv],
                    rv,
                    av,
                    tv)

        # save agent weights
        agent.save_weights(weight_directory)
        # test with viz
        env_viz.reset(seed=int(rng.integers(10000)))
        _, _, reward, _ = simple_run(env_viz, agent, T_test, debug=viz_debug, discrete=discrete_mode)
        print(np.sum(reward))

    for p, conn in zip(procs, conns):
        conn.send(False)  # TODO: is this necessary?
        p.join()


if __name__ == "__main__":
    # env_run, env_viz, agent = build_discrete_ppo(EnvsDiscrete.cartpole)
    # num_actions = EnvsDiscrete.cartpole.value.dims_actions
    # env_run, env_viz, agent = build_discrete_ppo(EnvsDiscrete.acrobot)
    # num_actions = EnvsDiscrete.acrobot.value.dims_actions
    # env_run, env_viz, agent = build_discrete_ppo(EnvsDiscrete.lunar, entropy_scale=0.5, embed_dim=16, layer_sizes=[128, 64])
    # num_actions = EnvsDiscrete.lunar.value.dims_actions
    # discrete_mode = True

    # continuous
    # env_run, env_viz, agent = build_continuous_ppo(EnvsContinuous.pendulum, init_var=1., learning_rate=.0001,
    #                                                vf_scale=.1, entropy_scale=0.1, eta=0.3)
    # num_actions = len(EnvsContinuous.pendulum.value.action_bounds)
    # env_run, env_viz, agent = build_continuous_ppo(EnvsContinuous.lunar_continuous, init_var=1., learning_rate=.0005,
    #                                                vf_scale=.1, entropy_scale=0.1, eta=0.3, layer_sizes=[128, 64])
    # num_actions = len(EnvsContinuous.lunar_continuous.value.action_bounds)
    # discrete_mode = False

    # run_and_train(env_run, env_viz, agent, num_actions, 20, 20000, 500, 500, viz_debug=False, discrete_mode=discrete_mode)


    # parallel
    builder = partial(build_discrete_ppo, env=EnvsDiscrete.cartpole)
    ll_run_and_train(builder, 2, 'weights/', 'data/', 10, 5000, 500, 500, 42, discrete_mode=True)
