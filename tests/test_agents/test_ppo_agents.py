"""Test PPO agents"""
import numpy as np
import numpy.random as npr
import tensorflow as tf
from typing import List, Dict
from unittest import TestCase
from tensorflow.keras.layers import Dense
from frameworks.layer_signatures import ScalarStateModel, DistroStateModel
from agents.ppo_agents import PPODiscrete, PPOContinuous


# utils


def one_hot(x: np.ndarray, num_dims: int):
    # --> T x num_dims
    v = [x == i for i in range(num_dims)]
    return np.concatenate([vi[:,None] for vi in v], axis=1) * 1


# fake reward funcs


def fake_reward_statecorr(state: Dict[str, np.ndarray], action: np.ndarray,
                          reward_action_ind: int = 0):
    # > if sum(states) > 0 and action [reward_action_ind] > 0.5 --> 1 reward
    # > if sum(states) < 0 and action [reward_action_ind] > 0.5 --> -1 reward
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


class DistroGauss(DistroStateModel):
    def __init__(self, num_actions: int, num_states: int):
        super(DistroGauss, self).__init__()
        self.num_actions = num_actions  # action dimensions
        self.layers = [Dense(4, activation="relu") for _ in range(num_states)]
        self.outlayer = Dense(int(2 * num_actions))

    def call(self, state_t: List[tf.Tensor]) -> tf.Tensor:
        v = [l(si) for l, si in zip(self.layers, state_t)]
        v = tf.concat(v, axis=1)
        vout = self.outlayer(v)
        mu = vout[:, :self.num_actions]
        # upper bound for stability
        prec = tf.math.sigmoid(vout[:, self.num_actions:] - 3.) * 10.
        return tf.concat([mu, prec], axis=1)


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
                                 eta=0.3,
                                 train_epoch=8,
                                 learning_rate=0.01,
                                 gamma=0.8)  # set this low to make problem easier
        self.num_actions = num_actions

    def train_statecorr(self):
        T = 1000
        # train on state correlation reward

        def gen_data(T):
            s1 = self.rng.random((T, 3)) - 0.5
            s2 = self.rng.random((T, 5)) - 0.5
            action = one_hot(self.rng.integers(0, self.num_actions, (T,)), 3)
            # --> 1st 
            reward = fake_reward_statecorr({"s1": s1, "s2": s2}, action, 1)
            return s1, s2, action, reward

        for _ in range(10):  # N successive runs:
            s1, s2, action, reward = gen_data(T)
            # bunch of fake runs with same data
            self.agent.train([{"s1": s1, "s2": s2}],
                             [reward[:-1]],
                             [action[:-1]],
                             [False])

        # simulate the agent:
        s1, s2, action, reward = gen_data(T)
        pos_actions, neg_actions = [], []
        for i in range(T):
            ind_vec = self.agent.select_action({"s1": s1[i:i+1], "s2": s2[i:i+1]}, False, False)
            ind = np.where(ind_vec[0] > 0.5)[0][0]
            self.assertTrue(np.shape(ind_vec) == (1, self.num_actions))
            self.assertAlmostEqual(np.sum(ind_vec), 1., 3)
            if reward[i] >= 0.5:
                pos_actions.append(ind)
            elif reward[i] <= -0.5:
                neg_actions.append(ind)
        pos_1perc = np.sum(np.array(pos_actions) == 1) / len(pos_actions)
        neg_1perc = np.sum(np.array(neg_actions) == 1) / len(neg_actions)
        self.assertTrue(pos_1perc >= 0.5)
        self.assertTrue(neg_1perc <= 0.1)


class TestContinuousAgent(TestCase):

    def setUp(self) -> None:
        self.rng = npr.default_rng(42)
        action_bounds = [(-10, 10) for _ in range(3)]
        state_dims = {"s1": (3,), "s2": (5,)}
        self.agent = PPOContinuous(lambda: DistroGauss(len(action_bounds), len(state_dims)),
                                   lambda: Critic(len(state_dims)),
                                   action_bounds, state_dims,
                                   entropy_scale=0.0,
                                   eta=0.3,
                                   train_epoch=8,
                                   learning_rate=0.01,
                                   gamma=0.8)  # set this low to make problem easier

    def train_statecorr(self):
        T = 2000
        # train on state correlation reward

        def gen_data(T):
            s1 = self.rng.random((T, 3)) - 0.5
            s2 = self.rng.random((T, 5)) - 0.5
            action = (self.rng.random((T, 3)) - 0.5) * 5.
            # --> 1st 
            reward = fake_reward_statecorr({"s1": s1, "s2": s2}, action, 1)
            return s1, s2, action, reward

        for _ in range(20):  # N successive runs:
            s1, s2, action, reward = gen_data(T)
            # bunch of fake runs with same data
            self.agent.train([{"s1": s1, "s2": s2}],
                            [reward[:-1]],
                            [action[:-1]],
                            [False])

        # simulate the agent:
        s1, s2, action, reward = gen_data(T)
        pos_actions, neg_actions = [], []
        for i in range(T):
            v = self.agent.select_action({"s1": s1[i:i+1], "s2": s2[i:i+1]}, False, False)[0]
            if reward[i] >= 0.5:
                pos_actions.append(v)
            elif reward[i] <= -0.5:
                neg_actions.append(v)
        pos_actions = np.array(pos_actions)
        neg_actions = np.array(neg_actions)
        pos_mu = np.mean(pos_actions[:, 1])
        neg_mu = np.mean(neg_actions[:, 1])
        print(pos_mu)
        print(neg_mu)
        self.assertTrue((pos_mu - neg_mu) > 1.)


if __name__ == "__main__":
    T = TestDiscreteAgent()
    T.setUp()
    T.train_statecorr()
    T2 = TestContinuousAgent()
    T2.setUp()
    T2.train_statecorr()
