"""Test PPO agents"""
import numpy as np
import numpy.random as npr
import tensorflow as tf
from typing import List, Dict
from unittest import TestCase
from tensorflow.keras.layers import Dense, Layer
from frameworks.layer_signatures import ScalarStateModel, VectorStateModel, DistroStateModel, VectorModel
from agents.ppo_agents import PPODiscrete, PPOContinuous, PPOContinuousExplo, filter_short_trajs


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


# TODO: obsolete ~ remove
def fake_reward_cycle(action: np.ndarray,
                      reward_action_ind: int = 0,
                      theta0: float = 0.):
    # control style

    # Dynamics:
    # > dtheta/dt = action[reward_action_ind]
    # > state = (cos(theta), sin(theta))

    # Reward:
    # > if state dot [1, 0] > 0. and action[reward_action_ind] > 0.5 --> 1 reward
    # > if state dot [1, 0] < 0. and action[reward_action_ind] > 0.5 --> -1 reward
    # > else --> 0 reward
    # returns 1. state array, 2. reward
    dtheta = action[:, reward_action_ind]
    theta = np.cumsum(dtheta) + theta0
    state = np.vstack((np.cos(theta), np.sin(theta))).T
    pos_reward = 1. * (state[:, 0] > 0.) * (action[:, reward_action_ind] > 0.5)
    neg_reward = 1. * (state[:, 0] < 0.) * (action[:, reward_action_ind] > 0.5)
    return state, pos_reward - neg_reward


def sim_env_cycle_control(theta: float,
                          action: np.ndarray,
                          reward_action_ind: int):
    # Dynamics:
    # > dtheta/dt = action[reward_action_ind]
    # > state = (cos(theta), sin(theta))

    # Reward:
    # > reward = dot(state, [1, 1])

    # Returns: 1. new theta, 2. state, 3. reward
    new_theta = theta + action[reward_action_ind]
    state = np.array([np.cos(new_theta), np.sin(new_theta)])
    reward = np.sum(state * np.array([1., 1.]))
    return new_theta, state, reward


# models


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


class Critic(Layer):
    def __init__(self, num_states: int):
        super(Critic, self).__init__()
        self.layers = [Dense(1) for _ in range(num_states)]

    def call(self, state_t: List[tf.Tensor]) -> tf.Tensor:
        v = [l(si) for l, si in zip(self.layers, state_t)]
        return tf.math.reduce_sum(tf.add_n(v), axis=1)


class ForwardModel(Layer):

    def __init__(self, encode_dims: int, num_state: int, scale: float = 1.):
        super(ForwardModel, self).__init__()
        self.scale = scale
        self.encode_dims = encode_dims
        self.state_layers = [Dense(encode_dims, activation="relu") for _ in range(num_state)]
        self.action_layer = Dense(encode_dims, activation="relu")
        self.clayer = Dense(encode_dims)

    def call(self, action_t: tf.Tensor, state_t: List[tf.Tensor]):
        vs = [sl(st) for sl, st in zip(self.state_layers, state_t)]
        va = self.action_layer(action_t)
        return self.clayer(tf.concat(vs + [va], axis=1)) * self.scale


class Encoder(Layer):

    def __init__(self, encode_dims: int, num_state: int, scale: float = 1.):
        super(Encoder, self).__init__()
        self.scale = scale
        self.encode_dims = encode_dims
        self.state_layers = [Dense(encode_dims, activation="relu") for _ in range(num_state)]
        self.clayer = Dense(encode_dims)

    def call(self, state_t: List[tf.Tensor]):
        vs = [sl(st) for sl, st in zip(self.state_layers, state_t)]
        return self.clayer(tf.concat(vs, axis=1)) * self.scale


# tests


class TestUtils(TestCase):

    def test_filter(self):
        batch_size = 8
        batch_size2 = 12
        actions = [np.zeros((batch_size, 2)), np.zeros((batch_size2, 2))]
        states = [{"A": np.zeros((batch_size, 3)), "B": np.zeros((batch_size, 4))},
                  {"A": np.zeros((batch_size2, 3)), "B": np.zeros((batch_size2, 4))}]
        reward = [np.zeros((batch_size,)), np.zeros((batch_size2,))]
        terminated = [True, False]
        # case 1: threshold < batch_size --> every trajectory passes
        [actions2, states2, rewards2, terms2] = filter_short_trajs(actions, [states, reward, terminated], min_length=5)
        self.assertTrue(np.all(actions[0] == actions2[0]))
        self.assertTrue(np.all(actions[1] == actions2[1]))
        self.assertTrue(np.all(states[0]["A"] == states2[0]["A"]))
        self.assertTrue(np.all(states[1]["B"] == states2[1]["B"]))
        self.assertTrue(np.all(reward[0] == rewards2[0]))
        self.assertTrue(np.all(reward[1] == rewards2[1]))
        self.assertTrue(terms2 == terminated)
        # case 2: first trajectory set gets filtered out
        [actions2, states2, rewards2, terms2] = filter_short_trajs(actions, [states, reward, terminated], min_length=10)
        self.assertTrue(np.all(actions[1] == actions2[0]))
        self.assertTrue(np.all(states[1]["A"] == states2[0]["A"]))
        self.assertTrue(np.all(reward[1] == rewards2[0]))
        self.assertTrue(terms2 == [False])


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
        action_bounds = [(-10, 10) for _ in range(3)]
        state_dims = {"s1": (3,), "s2": (5,)}
        self.agent = PPOContinuous(lambda: VectorStateModel(DistroGauss(len(action_bounds), len(state_dims))),
                                   lambda: ScalarStateModel(Critic(len(state_dims))),
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


class TestContinuousExplo(TestCase):

    def setUp(self) -> None:
        self.rng = npr.default_rng(42)
        action_bounds = [(-10, 10) for _ in range(1)]
        state_dims = {"s1": (2,), "s2": (4,)}
        self.encode_dims = 3
        self.agent = PPOContinuousExplo(lambda: VectorStateModel(DistroGauss(len(action_bounds), len(state_dims))),
                                   lambda: ScalarStateModel(Critic(len(state_dims))),
                                   lambda: VectorStateModel(Encoder(self.encode_dims, 2)),
                                   lambda: VectorModel(ForwardModel(self.encode_dims, 2)),
                                   action_bounds, state_dims,
                                   entropy_scale=0.0,
                                   eta=0.3,
                                   train_epoch=8,
                                   learning_rate=0.01,
                                   gamma=0.8)  # set this low to make problem easier

    def test_state_offsetting(self):
        states = [{"A": np.zeros((10, 2)), "B": np.ones((10, 4))},
                  {"A": np.zeros((12, 2)), "B": np.ones((12, 4))}]
        states_t, states_t1 = self.agent._make_offset_states(states)
        self.assertTrue(np.shape(states_t["A"]) == (20, 2))
        self.assertTrue(np.shape(states_t1["A"]) == (20, 2))
        self.assertTrue(np.shape(states_t["B"]) == (20, 4))
        self.assertTrue(np.shape(states_t1["B"]) == (20, 4))

    def test_train_cycle(self):
        def gen_data(T, agent):
            theta = self.rng.random()
            init_action = self.agent.init_action()[0]
            theta, s1, reward = sim_env_cycle_control(theta, init_action, 0)
            s2 = self.rng.random((4,))
            s1_save, s2_save, action_save, reward_save = [s1], [s2], [], []
            for _ in range(T):
                s2 = self.rng.random((4,))
                action = agent.select_action({"s1": s1[None], "s2": s2[None]}, False, False)[0]
                theta, s1, reward = sim_env_cycle_control(theta, action, 0)
                s1_save.append(s1)
                s2_save.append(s2)
                action_save.append(action)
                reward_save.append(reward)
            return np.array(s1_save), np.array(s2_save), np.array(action_save), np.array(reward_save)

        T = 2000
        reward_hist = []
        for _ in range(8):  # N successive runs:
            s1, s2, action, reward = gen_data(T, self.agent)
            reward_hist.append(np.sum(reward))
            # bunch of fake runs with same data
            self.agent.train([{"s1": s1, "s2": s2}],
                            [reward],
                            [action],
                            [False])
        print(reward_hist)
        self.assertTrue(reward_hist[-1] > (reward_hist[0] + 100))
        #self.assertTrue((reward_hist[-1] - reward_hist[0]) > 100.)

        # TODO: expectation = agent should learn to keep action near [1,1]
        # TODO: expectation = agent should be very good at predicting future!
        s1, s2, action, reward = gen_data(1024, self.agent)
        print(np.mean(s1, axis=0))
        self.assertTrue(np.sum(np.mean(s1, axis=0)) > 0.1)
        # --> (T + 1) x encode_dims
        enc = self.agent.phi([s1, s2])[0].numpy()
        # --> (T) x encode_dims (does not predict 0th entry)
        pred_for = self.agent.forward_model(action, [s1[:-1], s2[:-1]])[0].numpy()
        # correlation between prediction and actual:
        def calc_corr(x1, x2):
            num = np.sum(x1 * x2, axis=1)
            d1 = np.sqrt(np.sum(x1 * x1, axis=1))
            d2 = np.sqrt(np.sum(x2 * x2, axis=1))
            return num / (d1 * d2)
        corr_null = np.mean(calc_corr(pred_for, enc[:-1]))
        corr = np.mean(calc_corr(pred_for, enc[1:]))
        print(corr_null)
        print(corr)
        self.assertTrue(corr > corr_null)
        self.assertTrue(corr > 0.4)


if __name__ == "__main__":
    T = TestUtils()
    T.test_filter()
    T = TestDiscreteAgent()
    T.setUp()
    T.train_statecorr()
    T2 = TestContinuousAgent()
    T2.setUp()
    T2.train_statecorr()
    T = TestContinuousExplo()
    T.setUp()
    T.test_state_offsetting()
    T.test_train_cycle()
