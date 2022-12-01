"""Gymnasium[Box2D] lunar lander


    TODO: once we've done some experiments --> we can probably
        figure out how to make some runner functions that can be used
        across systems

"""
import gymnasium as gym
import numpy as np
import numpy.random as npr
from typing import Callable, Tuple, Dict
from random import randint

from frameworks.agent import Agent, RunData
from agents.q_agents import QAgent


def _one_hot(x: np.ndarray, num_action: int):
    # array of indices --> num_sample x num_action one-hot array
    x_oh = np.zeros((np.shape(x)[0], num_action))
    x_oh[np.arange(np.shape(x)[0]), x] = 1.
    return x_oh


def runner(env: gym.Env,
           agent: Agent,
           max_step: int = 200,
           init_action: int = 0):
    # state-action model
    # s_t --> model --> a_t --> env --> s_{t+1}, r_{t+1}
    #
    # action_model must keep track of (memory)
    #   1. previous observations, 2. previous actions
    # action model must take in env.step output
    action = init_action
    obs, actions, rewards, other = [], [], [], []
    for _ in range(max_step):
        # yields: s_{t+1}, r_{t+1}
        step_output = env.step(action)
        # yields: a_{t+1}
        # TODO: generalize kind of information
        #   that can be handed to agent
        action = agent.select_action([step_output[0]])
        obs.append(step_output[0])
        actions.append(action)
        rewards.append(step_output[1])
        other.append(step_output[2:])
    # return alignment: s_t, a_t, r_{t+1}
    # where: s_t -> a_t -> r_{t+1}
    return RunData(np.array(obs[:-1]),
                   np.array(obs[1:]),
                   # TODO: hack...
                   _one_hot(np.array(actions[:-1]), 4),
                   np.array(rewards[1:]))


# epochs


def run_epoch(env: gym.Env,
              agent: Agent,
              struct: RunData,
              max_step: int,
              run_iters: int,
              p: float, rng: npr.Generator):
    # TODO: currently resetting struct every epoch

    # samping/processing wrapper for run output
    def sample(v: RunData):
        sel = rng.random(np.shape(v.states)[0]) <= p
        return RunData(v.states[sel], v.states_t1[sel],
                       v.actions[sel], v.rewards[sel])

    env.reset(seed=42)
    for _ in range(run_iters - 1):
        env.reset(seed=42)
        add_struct = sample(runner(env, agent, max_step))
        # merge:
        struct = RunData(np.concatenate([struct.states, add_struct.states], axis=0),
                         np.concatenate([struct.states_t1, add_struct.states_t1], axis=0),
                         np.concatenate([struct.actions, add_struct.actions], axis=0),
                         np.concatenate([struct.rewards, add_struct.rewards], axis=0))
    agent.train(struct, 12)
    return struct


if __name__ == "__main__":
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Layer
    from arch_layers.simple_networks import DenseNetwork
    from typing import List
    class DenseScalar(Layer):
        def __init__(self):
            super(DenseScalar, self).__init__()
            self.d_act = Dense(8)
            self.d_state = Dense(8)
            self.net = DenseNetwork([50, 25, 20], 1, 0.)

        def call(self, action_t: tf.Tensor, state_t: List[tf.Tensor]):
            x_a = self.d_act(action_t)
            x_s = self.d_state(state_t[0])
            yp = self.net(tf.concat([x_a, x_s], axis=1))
            return yp[:, 0]  # to scalar
    agent = QAgent(DenseScalar(), 4, 8)

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
        for _ in range(3):
            env_disp.reset(seed=42)
            s0 = runner(env_disp, agent, 200)
            print(np.mean(s0.rewards))
            if struct is None:
                struct = s0

        # train
        struct = run_epoch(env_run, agent, struct,
                           200, 10, .2, npr.default_rng(42))
        # purge struct:
        num_purge = np.shape(struct.rewards)[0] - 25000
        if num_purge > 0:
            struct = RunData(struct.states[num_purge:],
                             struct.states_t1[num_purge:],
                             struct.actions[num_purge:],
                             struct.rewards[num_purge:])

    env_run.close()
    env_disp.close()
