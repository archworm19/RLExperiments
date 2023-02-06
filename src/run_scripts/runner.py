"""Running Gym Simulations"""
import gymnasium as gym
import numpy.random as npr
from frameworks.agent import Agent, AgentEpoch


def simple_run(env: gym.Env,
               agent: AgentEpoch,  # TODO: move to new framework
               num_step: int):
    action = agent.init_action()
    cur_state = env.step(action)[0]
    reward_sum = 0.
    for _ in range(num_step):
        action = agent.select_action({"core_state": cur_state})
        step_output = env.step(action)
        cur_state = step_output[0]
        reward = step_output[1]
        reward_sum += reward
        if step_output[2]:
            break
    return reward_sum


def runner_epoch(env: gym.Env,
                 agent: AgentEpoch,  # TODO: update when agent framework updates
                 num_step: int,
                 rng: npr.Generator):
    # run for num_step --> restart if necessary
    # state-action model
    # s_t --> model --> a_t --> env --> s_{t+1}, r_{t}
    # returns: (lists = different trajectories)
    #       start new trajectory upon termination
    # 1. states (T + 1 x ...), 2. actions (T), 3. rewards (T),
    # 4. terminations (1 entry for each trajectory)
    #   NOTE: T shows the relative array lengths
    action = agent.init_action()  # not saved
    cur_state = env.step(action)[0]
    save_states = [[cur_state]]
    save_actions = [[]]
    save_rewards = [[]]
    save_terms = [False]
    for _ in range(num_step):
        # TODO: add other states (pixels!)
        # TODO: take in state names?
        action = agent.select_action({"core_state": cur_state})
        step_output = env.step(action)
        cur_state = step_output[0]
        reward = step_output[1]

        # save data
        save_states[-1].append(cur_state)
        save_actions[-1].append(action)
        save_rewards[-1].append(reward)

        # if terminated --> reset env
        if step_output[2]:
            env.reset(seed=int(rng.integers(0, 100000)))
            action = agent.init_action()  # not saved
            cur_state = env.step(action)[0]
            save_states.append([cur_state])
            save_actions.append([])
            save_rewards.append([])
            save_terms[-1] = True
            save_terms.append(False)
    return save_states, save_actions, save_rewards, save_terms


def runner(env: gym.Env,
           agent: Agent,
           max_step: int = 200,
           step_per_train: int = 1,
           step_per_copy: int = 1,
           train_mode: bool = True,
           timeout: bool = False,
           debug: bool = False):
    # state-action model
    # s_t --> model --> a_t --> env --> s_{t+1}, r_{t}
    #
    # action_model must keep track of (memory)
    #   1. previous observations, 2. previous actions
    # action model must take in env.step output
    # NOTE: this function should be agnostic to continuous vs. discrete
    #   control as long as agent and environment are compatible
    # NOTE: model records 1. states, 2. normalized time, 3. TODO: pixels
    # NOTE: if timeout is set --> end of run marked as termination
    test_mode = not train_mode
    action = agent.init_action()
    cur_state = env.step(action)[0]
    save_rewards = []
    for i in range(max_step):
        action = agent.select_action([cur_state, [(i + 0.) / max_step]],
                                     test_mode=test_mode, debug=debug)
        step_output = env.step(action)
        new_state = step_output[0]
        reward = step_output[1]
        termination = step_output[2]

        # timeout
        if (i >= (max_step - 1)) and timeout:
            termination = True

        # only save training data
        if train_mode:
            agent.save_data([cur_state, [(i + 0.) / max_step]],
                            [new_state, [(i + 1.) / max_step]],
                            action, reward,
                            termination)

        if train_mode:
            if i % step_per_train == 0:
                agent.train()
            # TODO: make this part of interface?
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
