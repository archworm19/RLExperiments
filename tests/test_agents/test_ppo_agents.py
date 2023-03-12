"""Test PPO agents"""
import numpy as np
import numpy.random as npr
import tensorflow as tf
from typing import List, Dict
from unittest import TestCase
from tensorflow.keras.layers import Dense, Layer
from frameworks.layer_signatures import ScalarStateModel, VectorStateModel, DistroStateModel
from agents.ppo_agents import PPODiscrete, PPOContinuous, calculate_vpred, calculate_vpred_end


class Rotator:
    # 2d observation space; 1d continuous action space
    #   > action rotates about the origin
    #   > reward relative to pointing in target direction
    def __init__(self, target_angle: float = np.pi, init_angle: float = 0.):
        self.target_angle = target_angle
        self._vt = np.array([np.cos(self.target_angle), np.sin(self.target_angle)])
        self.init_angle = init_angle
        self.reset()

    def reset(self):
        self._angle = self.init_angle
        return np.array([np.cos(self._angle), np.sin(self._angle)])

    def step(self, action: float):
        # action -> delta angle
        self._angle += action
        # new obs calc:
        obs = np.array([np.cos(self._angle), np.sin(self._angle)])
        # reward
        reward = np.sum(obs * self._vt)
        return obs, reward


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


# Model Components


class DistroMod(Layer):
    def __init__(self, num_actions: int, num_states: int):
        super(DistroMod, self).__init__()
        self.layers = [Dense(num_actions) for _ in range(num_states)]

    def call(self, state_t: List[tf.Tensor]) -> tf.Tensor:
        v = [l(si) for l, si in zip(self.layers, state_t)]
        return tf.nn.softmax(tf.add_n(v), axis=1)


class DistroGauss(Layer):
    def __init__(self, num_actions: int, num_states: int):
        super(DistroGauss, self).__init__()
        self.num_actions = num_actions  # action dimensions
        self.layers = [Dense(32, activation="relu") for _ in range(num_states)]
        self.outlayer = Dense(int(num_actions))
        self.log_std_dev = tf.Variable(0.)

    def call(self, state_t: List[tf.Tensor]) -> tf.Tensor:
        v = [l(si) for l, si in zip(self.layers, state_t)]
        v = tf.concat(v, axis=1)
        mu = self.outlayer(v)
        lsd = tf.ones(tf.shape(mu)) * self.log_std_dev
        return tf.concat([mu, lsd], axis=1)


class Critic(Layer):
    def __init__(self, num_states: int):
        super(Critic, self).__init__()
        self.layers = [Dense(1) for _ in range(num_states)]

    def call(self, state_t: List[tf.Tensor]) -> tf.Tensor:
        v = [l(si) for l, si in zip(self.layers, state_t)]
        return tf.math.reduce_sum(tf.add_n(v), axis=1)


class MeanCritic(Layer):
    def __init__(self):
        super(MeanCritic, self).__init__()

    def call(self, state_t: List[tf.Tensor]) -> tf.Tensor:
        vmu = [tf.math.reduce_mean(st, axis=1) for st in state_t]
        return tf.add_n(vmu) / len(state_t)


# Testers


class TestUtilities(TestCase):

    def test_vpred(self):
        MC = VectorStateModel(MeanCritic())
        a = np.array([[1., 1.],
                      [2., 2.],
                      [3., 3.]])
        b = a + 100.
        states = [{"B": b, "A": a}]
        vpred = calculate_vpred(MC, states, ["A", "B"])
        self.assertTrue(len(vpred) == 1)  # 1 trajectory
        self.assertTrue(np.all(vpred[0] == np.array([51., 52., 53.])))

    def test_vpred_end(self):
        MC = VectorStateModel(MeanCritic())
        a = np.array([[1., 1.],
                      [2., 2.],
                      [3., 3.]])
        b = a + 100.
        states = [{"B": b, "A": a}]
        vpred = calculate_vpred_end(MC, states, ["A", "B"])
        self.assertEqual(vpred, [53.0])


class TestDiscreteAgent(TestCase):

    def setUp(self) -> None:
        self.rng = npr.default_rng(42)
        num_actions = 3
        state_dims = {"s1": (3,), "s2": (5,)}
        self.agent = PPODiscrete(lambda: DistroStateModel(DistroMod(num_actions, len(state_dims))),
                                 lambda: ScalarStateModel(Critic(len(state_dims))),
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
        # make agent for rotator environment (2d obs, 1d action)
        action_bounds = [(-10, 10) for _ in range(1)]
        state_dims = {"s1": (2,)}
        self.agent = PPOContinuous(lambda: VectorStateModel(DistroGauss(len(action_bounds), len(state_dims))),
                                   lambda: ScalarStateModel(Critic(len(state_dims))),
                                   action_bounds, state_dims,
                                   entropy_scale=0.0,
                                   eta=0.3,
                                   train_epoch=8,
                                   learning_rate=0.00001,
                                   gamma=0.8)  # set this low to make problem easier

    def train_rotator(self):
        T = 2000
        # train on state correlation reward
        env = Rotator()

        def gen_data(T, debug=False):
            obs = env.reset()
            save_obs, save_action, save_reward = [obs], [], []
            for _ in range(T):
                action = self.agent.select_action({"s1": obs[None]}, False, debug)
                obs, reward = env.step(action[0][0])
                save_obs.append(obs)
                save_action.append(action[0])
                save_reward.append(reward)
            return np.array(save_obs), np.array(save_action), np.array(save_reward)

        epoch_rews = []
        for _ in range(10):  # N successive runs:
            s1, action, reward = gen_data(T)
            # bunch of fake runs with same data
            self.agent.train([{"s1": s1}],
                            [reward],
                            [action],
                            [False])
            epoch_rews.append(np.sum(reward))
            print(epoch_rews[-1])

        # better on average?
        self.assertTrue(epoch_rews[-1] > 0)
        # big improvement
        self.assertTrue((epoch_rews[-1] - epoch_rews[0]) > 100)

        # simulate the agent again --> how does it do?
        # obs, acts, rews = gen_data(50, True)
        # print(obs)
        # print(acts)
        # print(rews)


if __name__ == "__main__":
    T = TestUtilities()
    T.test_vpred()
    T.test_vpred_end()
    T = TestDiscreteAgent()
    T.setUp()
    T.train_statecorr()
    T2 = TestContinuousAgent()
    T2.setUp()
    T2.train_rotator()
