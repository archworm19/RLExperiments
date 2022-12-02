"""Running Gym Simulations"""

import gymnasium as gym
import numpy as np
import numpy.random as npr

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
    obs, actions, rewards, other = [state0], [], [], []
    for _ in range(max_step):
        action = agent.select_action([cur_state], debug=debug)
        step_output = env.step(action)
        cur_state = step_output[0]
        obs.append(cur_state)
        actions.append(action)
        rewards.append(step_output[1])
        other.append(step_output[2:])

        if debug:
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
                   np.array(rewards))


# epochs


def run_epoch(env: gym.Env,
              agent: Agent,
              struct: RunData,
              max_step: int,
              run_iters: int,
              p: float, rng: npr.Generator,
              termination_reward: float = None,
              debug: float = False):
    # samping/processing wrapper for run output
    def sample(v: RunData):
        sel = rng.random(np.shape(v.states)[0]) <= p
        # TODO: this is lazy
        if termination_reward is not None:
            v.rewards[-1] = termination_reward
        return RunData(v.states[sel], v.states_t1[sel],
                       v.actions[sel], v.rewards[sel])

    for z in range(run_iters - 1):
        env.reset(seed=z)  # TODO: expose this as arg
        add_struct = sample(runner(env, agent, max_step))
        # merge:
        struct = RunData(np.concatenate([struct.states, add_struct.states], axis=0),
                         np.concatenate([struct.states_t1, add_struct.states_t1], axis=0),
                         np.concatenate([struct.actions, add_struct.actions], axis=0),
                         np.concatenate([struct.rewards, add_struct.rewards], axis=0))
    agent.train(struct, 24, debug=debug)
    return struct
