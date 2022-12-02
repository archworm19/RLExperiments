"""Gymnasium[Box2D] lunar lander
"""
import gymnasium as gym
import numpy as np
import numpy.random as npr
from frameworks.agent import RunData

if __name__ == "__main__":
    from run_scripts.runner import runner, run_epoch
    from run_scripts.utils import build_dense_qagent
    agent = build_dense_qagent(num_actions=4, num_observations=8)
    agent.rand_act_prob = 0.3  # will decay this down over time
    rap_decay_rate = 0.9

    # render modes:
    #   None(default): no render
    #   "human": continuously render in current display
    #   "rgb_array", "ansi", and a few others
    # lunar lander
    # env = gym.make("LunarLander-v2", render_mode="human")
    env_run = gym.make("LunarLander-v2")
    env_disp = gym.make("LunarLander-v2", render_mode="human")
    struct = None
    for _ in range(30):
        print("Run Epoch")
        # display
        for z in range(3):
            env_disp.reset(seed=z)
            s0 = runner(env_disp, agent, 400)
            print(np.mean(s0.rewards))
            if struct is None:
                struct = s0

        # train
        struct = run_epoch(env_run, agent, struct,
                           400, 10, .2, npr.default_rng(42))
        # purge struct:
        num_purge = np.shape(struct.rewards)[0] - 25000
        if num_purge > 0:
            struct = RunData(struct.states[num_purge:],
                             struct.states_t1[num_purge:],
                             struct.actions[num_purge:],
                             struct.rewards[num_purge:])

        # decay:
        agent.rand_act_prob *= rap_decay_rate

    env_run.close()
    env_disp.close()
