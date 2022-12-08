"""Running Gym Simulations"""
import gymnasium as gym
import numpy as np
from frameworks.agent import Agent, RunData


def _one_hot(x: np.ndarray, num_action: int):
    # array of indices --> num_sample x num_action one-hot array
    x_oh = np.zeros((np.shape(x)[0], num_action))
    x_oh[np.arange(np.shape(x)[0]), x] = 1.
    return x_oh


# TODO: biggest concern: I'm still not sure alignment is correct
def runner(env: gym.Env,
           agent: Agent,
           max_step: int = 200,
           init_action: int = 0,
           debug: bool = False):
    # state-action model
    # s_t --> model --> a_t --> env --> s_{t+1}, r_{t+1}
    #
    # action_model must keep track of (memory)
    #   1. previous observations, 2. previous actions
    # action model must take in env.step output
    action = agent.init_action()
    state0 = env.step(init_action)[0]
    cur_state = state0
    obs, actions, rewards, termination = [state0], [], [], []
    for _ in range(max_step):
        action = agent.select_action([cur_state], debug=debug)
        step_output = env.step(action)
        cur_state = step_output[0]
        obs.append(cur_state)
        actions.append(action)
        rewards.append(step_output[1])
        termination.append(step_output[2])

        if debug:
            print("rewards")
            print(rewards[-1])
            input("cont?")

        # check for termination
        if step_output[2]:
            break

    # return alignment: s_t, a_t, r_{t+1}
    # where: s_t -> a_t -> r_{t+1}
    return RunData(np.array(obs[:-1]),
                   np.array(obs[1:]),
                   # TODO: hack...
                   _one_hot(np.array(actions), agent.num_actions),
                   np.array(rewards),
                   np.array(termination) * 1)
