"""Running Gym Simulations"""
import gymnasium as gym
import numpy as np
from frameworks.agent import Agent, TrainEpoch, TrainOnLine


# type intersections


class AgentOnLine(Agent, TrainOnLine):
    pass


def simple_run(env: gym.Env,
               agent: Agent,
               num_step: int,
               debug: bool = False,
               discrete: bool = True):
    # TODO: docstring ~ return types
    # TODO: should probably take in state names
    # NOTE: if discrete --> assumes 1-hot input
    action = agent.init_action()[0]
    if discrete:
        cur_state = env.step(np.where(action > 0.5)[0][0])[0]
    else:
        cur_state = env.step(action)[0]
 
    # world model: s_t + a_t --> r_t, s_{t+1}
    save_states = [cur_state]
    save_actions, save_rewards = [], []
    terminated = False

    for _ in range(num_step):
        action = agent.select_action({"core_state": np.array(cur_state)[None]}, debug=debug, test_mode=False)[0]
        if discrete:
            step_output = env.step(np.where(action > 0.5)[0][0])
        else:
            step_output = env.step(action)
        cur_state = step_output[0]
        reward = step_output[1]

        # saves
        save_states.append(cur_state)
        save_actions.append(action)
        save_rewards.append(reward)

        if debug:
            print(reward)
            input("cont?")
        if step_output[2]:
            terminated = True
            break
    return save_states, save_actions, save_rewards, terminated


def runner(env: gym.Env,
           agent: AgentOnLine,
           max_step: int = 200,
           step_per_train: int = 1,
           step_per_copy: int = 1,
           debug: bool = False,
           discrete: bool = True):
    # state-action model
    # s_t --> model --> a_t --> env --> s_{t+1}, r_{t}
    #
    # action_model must keep track of (memory)
    #   1. previous observations, 2. previous actions
    # action model must take in env.step output
    action = agent.init_action()[0]
    if discrete:
        cur_state = env.step(np.where(action > 0.5)[0][0])[0]
    else:
        cur_state = env.step(action)[0]
    save_rewards = []
    for i in range(max_step):
        action = agent.select_action({"core_state": np.array(cur_state)[None]}, debug=debug, test_mode=False)[0]
        if discrete:
            step_output = env.step(np.where(action > 0.5)[0][0])
        else:
            step_output = env.step(action)
        new_state = step_output[0]
        reward = step_output[1]
        termination = step_output[2]

        agent.save_data({"core_state": np.array(cur_state)[None]},
                        {"core_state": np.array(new_state)[None]},
                        np.array(action)[None],
                        np.array(reward)[None],
                        np.array(termination)[None])

        if i % step_per_train == 0:
            agent.train()
        # TODO: make this part of interface
        if i % step_per_copy == 0:
            agent._copy_model()

        save_rewards.append(reward)
        cur_state = new_state

        if debug:
            input("cont?")

        # check for termination
        if step_output[2]:
            break
    return save_rewards
