"""Parallelism Models"""
import gymnasium as gym
import pickle
import numpy.random as npr
from typing import Callable, List
from multiprocessing import connection, Queue, Pipe, Process
from functools import partial
from replay_buffers.async_storage import TrajectoryStore
from run_scripts.builders import EnvConfigCont


def run_step_continuous(pid: int,
                        env_spec: EnvConfigCont,
                        rand_seed: int,
                        conn: connection.Connection,
                        queue: Queue,
                        max_run_length: int):
    """subprocess runstep
        waits for action --> writes [pid, new_state, reward, termination bit] to queue

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
    rl = 0
    while True:
        action = pickle.loads(conn.recv())
        if len(action) == 0:  # stop signal
            break
        step_output = env.step(action)
        new_state = step_output[0]
        reward = step_output[1]
        term = step_output[2]
        queue.put(pickle.dumps([pid, new_state, reward, term]), block=True)
        rl += 1
        if term or rl >= max_run_length:
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
        print("AYO")
        # send stop signal + wait for response
        c.send(pickle.dumps([]))
        p.join()  # blocks until subprocess terminates

# TODO: sending the environment builder function to the subprocess
#   is probably not the greatest of ideas
#   --> send the env specification instead!!

# TODO: master process
# > launch the subprocesses
# > build the model
# > for epochs
#       build new trajectory store
#       pull from queue --> send action back to connection
#       at end of epoch (total number of steps) --> train
# TODO: save model periodically...
# def ll_run(builder: Callable,  # --> env_run, env_viz, agent
#            num_procs: int,
#            num_epoch: int,
#            T_run: int,  # total
#            run_cutoff: int,
#            rand_seed: int):
#     # launch the subprocesses / environments

#     pass

if __name__ == "__main__":
    # launch testing

    from run_scripts.builders import EnvsContinuous

    procs, conns, q = _launch_envs(2, EnvsContinuous.pendulum.value, 10)
    _close_envs(conns, procs)
