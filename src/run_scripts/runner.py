"""Running Gym Simulations"""
import gymnasium as gym
import numpy as np
from frameworks.agent import Agent


def runner(env: gym.Env,
           agent: Agent,
           max_step: int = 200,
           init_action: int = 0,
           step_per_train: int = 1,
           step_per_copy: int = 1,
           train_mode: bool = True,
           debug: bool = False):
    # state-action model
    # s_t --> model --> a_t --> env --> s_{t+1}, r_{t+1}
    #
    # action_model must keep track of (memory)
    #   1. previous observations, 2. previous actions
    # action model must take in env.step output
    action = agent.init_action()
    cur_state = env.step(init_action)[0]
    save_rewards = []
    for i in range(max_step):
        action = agent.select_action([cur_state], debug=debug)
        step_output = env.step(action)
        new_state = step_output[0]
        reward = step_output[1]
        termination = step_output[2]

        agent.save_data([cur_state], [new_state],
                        action, reward,
                        termination)

        # TODO: bring in steps_per_train/copy

        if train_mode:
            if i % step_per_train == 0:
                agent.train()
            # TODO: make this part of interface?
            if i % step_per_copy == 0:
                agent._copy_model()

        save_rewards.append(reward)
        cur_state = new_state

        # check for termination
        if step_output[2]:
            break
    return save_rewards
