"""Test PPO agents"""
import numpy as np
import numpy.random as npr
import tensorflow as tf
from typing import List, Dict
from unittest import TestCase
from tensorflow.keras.layers import Dense
from frameworks.layer_signatures import ScalarStateModel, DistroStateModel
from agents.ppo_agents import PPODiscrete


# utils


def one_hot(x: np.ndarray, num_dims: int):
    # --> T x num_dims
    v = [x == i for i in range(num_dims)]
    return np.concatenate([vi[:,None] for vi in v], axis=1) * 1


# fake reward funcs


def fake_reward_statecorr(state: Dict[str, np.ndarray], action: np.ndarray,
                          reward_action_ind: int = 0):
    # > if sum(states) > 0 and take action [reward_action_ind] --> 1 reward
    # > if sum(states) < 0 and take action [reward_action_ind] --> -1 reward
    # > else --> 0 reward
    # assumes: each state has shape T x d_s_i
    #          action = one-hot rep = T x d_a
    # returns rewards: shape = T
    v = [np.sum(state[k], axis=1)[:, None] for k in state]
    v_pos = np.sum(np.concatenate(v, axis=1), axis=1) > 0
    pos_reward = 1. * v_pos * (action[:, reward_action_ind] > 0.5)
    neg_reward = 1. * (np.logical_not(v_pos)) * (action[:, reward_action_ind] > 0.5)
    return pos_reward - neg_reward


# models


class DistroMod(DistroStateModel):
    def __init__(self, num_actions: int, num_states: int):
        super(DistroMod, self).__init__()
        self.layers = [Dense(num_actions) for _ in range(num_states)]

    def call(self, state_t: List[tf.Tensor]) -> tf.Tensor:
        v = [l(si) for l, si in zip(self.layers, state_t)]
        return tf.nn.softmax(tf.add_n(v), axis=1)


class Critic(ScalarStateModel):
    def __init__(self, num_states: int):
        super(Critic, self).__init__()
        self.layers = [Dense(1) for _ in range(num_states)]

    def call(self, state_t: List[tf.Tensor]) -> tf.Tensor:
        v = [l(si) for l, si in zip(self.layers, state_t)]
        return tf.math.reduce_sum(tf.add_n(v), axis=1)


class TestDiscreteAgent(TestCase):

    def setUp(self) -> None:
        self.rng = npr.default_rng(42)
        num_actions = 3
        state_dims = {"s1": (3,), "s2": (5,)}
        self.agent = PPODiscrete(lambda: DistroMod(num_actions, len(state_dims)),
                                 lambda: Critic(len(state_dims)),
                                 num_actions, state_dims,
                                 entropy_scale=0.0,
                                 train_epoch=8)
        self.num_actions = num_actions

    def train_statecorr(self, T: int = 10000):
        # TODO: this isn't working for shit
        # train on state correlation reward
        s1 = self.rng.random((T, 3)) - 0.5
        s2 = self.rng.random((T, 5)) - 0.5
        action = one_hot(self.rng.integers(0, self.num_actions, (T,)), 3)
        # --> 1st 
        reward = fake_reward_statecorr({"s1": s1, "s2": s2}, action, 1)
        print(np.sum(s1, axis=1) + np.sum(s2, axis=1))
        print(action)
        print(reward)
        # bunch of fake runs with same data
        for _ in range(10):
            self.agent.train([{"s1": s1, "s2": s2}],
                            [reward[:-1]],
                            [action[:-1]],
                            [False])

        # simulate the agent:
        pos_actions, neg_actions = [], []
        for i in range(T):
            ind = self.agent.select_action({"s1": s1[i], "s2": s2[i]})
            if reward[i] >= 0.5:
                pos_actions.append(ind)
            elif reward[i] <= -0.5:
                neg_actions.append(ind)
        print(pos_actions)



if __name__ == "__main__":
    T = TestDiscreteAgent()
    T.setUp()
    # T._sim_agent_statecorr()
    T.train_statecorr(1000)
