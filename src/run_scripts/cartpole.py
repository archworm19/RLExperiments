"""Gymnasium Cartpole-V1
    One of the easier environments

    action space = discrete(2)
    observation shape = (4,)

"""

import gymnasium as gym
import numpy as np
import numpy.random as npr
from frameworks.agent import RunData


if __name__ == "__main__":
    from run_scripts.runner import runner, run_epoch
    from run_scripts.utils import build_dense_qagent
    agent = build_dense_qagent(num_actions=2, num_observations=4,
                               layer_sizes=[32, 16],
                               drop_rate=0.)
    agent.rand_act_prob = 0.25
    rap_decay_rate = 1.
    run_length = 200
    seeds = [i for i in range(5)]

    # render modes:
    #   None(default): no render
    #   "human": continuously render in current display
    #   "rgb_array", "ansi", and a few others
    env_run = gym.make("CartPole-v1")
    env_disp = gym.make("CartPole-v1", render_mode="human")

    struct = None
    for i in range(20):
        # display
        for z in seeds:
            env_disp.reset(seed=z)
            s0 = runner(env_disp, agent, run_length,
                        debug=False)
            print(np.sum(s0.rewards))
            if struct is None:
                struct = s0

        for j in range(100):
            # train
            struct = run_epoch(env_run, agent, struct,
                            run_length, seeds,
                            debug=False)

            # purge struct:
            # TODO: move and stabilize
            num_purge = np.shape(struct.rewards)[0] - 10000
            if num_purge > 0:
                struct = RunData(struct.states[-num_purge:],
                                struct.states_t1[-num_purge:],
                                struct.actions[-num_purge:],
                                struct.rewards[-num_purge:],
                                struct.termination[-num_purge:])

        # decay:
        agent.rand_act_prob *= rap_decay_rate

    env_run.close()
    env_disp.close()