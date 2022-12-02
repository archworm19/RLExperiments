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
                               layer_sizes=[8],
                               drop_rate=0.)
    agent.rand_act_prob = 0.1  # will decay this down over time
    rap_decay_rate = 1.
    # NOTE: if this is greater than max run length (250)
    #       termination reward won't work properly
    #       TODO: make termination reward more general
    run_length = 200

    # render modes:
    #   None(default): no render
    #   "human": continuously render in current display
    #   "rgb_array", "ansi", and a few others
    env_run = gym.make("CartPole-v1")
    env_disp = gym.make("CartPole-v1", render_mode="human")
    struct = None
    for _ in range(30):
        print("Run Epoch")
        # display
        for z in range(3):
            env_disp.reset(seed=z)
            s0 = runner(env_disp, agent, 400,
                        debug=True)
            print(np.sum(s0.rewards))
            if struct is None:
                struct = s0

        # train
        struct = run_epoch(env_run, agent, struct,
                           run_length, 50, 1., npr.default_rng(42),
                           termination_reward=-1.)  # NOTE: this is key
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