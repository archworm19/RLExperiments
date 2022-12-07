"""Gymnasium Cartpole-V1
    One of the easier environments

    action space = discrete(2)
    observation shape = (4,)

"""
import gymnasium as gym
import numpy as np
import numpy.random as npr
from frameworks.agent import RunData

from run_scripts.runner import runner
from run_scripts.utils import build_dense_qagent, purge_run_data


def _generate_seeds(num_seeds: int):
    v = npr.randint(0, 1000, num_seeds)
    return [int(vi) for vi in v]


def _append_rundat(struct: RunData, add_struct: RunData):
    return RunData(np.concatenate([struct.states, add_struct.states], axis=0),
                        np.concatenate([struct.states_t1, add_struct.states_t1], axis=0),
                        np.concatenate([struct.actions, add_struct.actions], axis=0),
                        np.concatenate([struct.rewards, add_struct.rewards], axis=0),
                        np.concatenate([struct.termination, add_struct.termination], axis=0))


if __name__ == "__main__":
    agent = build_dense_qagent(num_actions=2, num_observations=4,
                               layer_sizes=[32, 16],
                               drop_rate=0.,
                               gamma=0.6, tau=.1,
                               num_batch_sample=2)
    agent.rand_act_prob = 1.
    rap_decay_rate = .993
    run_length = 200
    # number of seeds per 
    num_runs = 1000
    run_per_train = 1
    run_per_copy = 1

    # render modes:
    #   None(default): no render
    #   "human": continuously render in current display
    #   "rgb_array", "ansi", and a few others
    env_run = gym.make("CartPole-v1")
    env_disp = gym.make("CartPole-v1", render_mode="human")

    struct = None

    # different seed for each run:
    seeds = _generate_seeds(num_runs)
    for i, seed in enumerate(seeds):
        if i % 20 == 0:  # use display
            active_env = env_disp
        else:
            active_env = env_run

        active_env.reset(seed=seed)
        # gather run data
        add_struct = runner(active_env, agent, run_length,
                            debug=False)
        print(np.sum(add_struct.rewards))

        # append
        if struct is None:
            struct = add_struct
        else:
            struct = _append_rundat(struct, add_struct)

        # train
        if i % run_per_train == 0:
            agent.train(struct, debug=False)

        # copy
        if i % run_per_copy == 0:
            agent._copy_model(debug=False)

        # purge struct:
        struct = purge_run_data(struct, 100000)

        # decay:
        agent.rand_act_prob *= rap_decay_rate

    env_run.close()
    env_disp.close()
