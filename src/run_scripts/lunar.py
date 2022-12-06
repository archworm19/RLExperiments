"""Gymnasium[Box2D] lunar lander
"""
import gymnasium as gym
import numpy as np
import numpy.random as npr
from frameworks.agent import RunData


if __name__ == "__main__":
    from run_scripts.runner import runner, run_epoch
    from run_scripts.utils import build_dense_qagent, purge_run_data
    agent = build_dense_qagent(num_actions=4, num_observations=8,
                               layer_sizes=[32, 16],
                               drop_rate=0.)
    agent.rand_act_prob = 0.25
    rap_decay_rate = 1.
    run_length = 400
    seeds = [i for i in range(5)]

    # render modes:
    #   None(default): no render
    #   "human": continuously render in current display
    #   "rgb_array", "ansi", and a few others
    env_run = gym.make("LunarLander-v2")
    env_disp = gym.make("LunarLander-v2", render_mode="human")

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
            struct = purge_run_data(struct, 100000)

        # decay:
        agent.rand_act_prob *= rap_decay_rate

    env_run.close()
    env_disp.close()
