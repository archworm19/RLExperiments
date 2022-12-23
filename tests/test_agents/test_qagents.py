"""Testing qagents on fake tasks"""
from re import S
import numpy as np
import numpy.random as npr
import tensorflow as tf
from typing import List, Tuple
from unittest import TestCase
from tensorflow.keras.layers import Dense
from frameworks.layer_signatures import ScalarModel, ScalarStateModel
from agents.q_agents import QAgent, RunIface, QAgent_cont, RunIfaceCont
from arch_layers.simple_networks import DenseNetwork


class DenseScalar(ScalarModel):
    # simple scalar model
    def __init__(self):
        super(DenseScalar, self).__init__()
        self.d_act = Dense(4)
        self.d_state = Dense(4)
        self.net = DenseNetwork([10], 1, 0.)

    def call(self, action_t: tf.Tensor, state_t: List[tf.Tensor]):
        x_a = self.d_act(action_t)
        x_s = self.d_state(state_t[0])
        yp = self.net(tf.concat([x_a, x_s], axis=1))
        return yp[:, 0]  # to scalar


class DenseScalarPi(ScalarStateModel):
    # pi: pi(a | s)
    def __init__(self,
                 bounds: List[Tuple],
                 embed_dim: int,
                 layer_sizes: List[int],
                 drop_rate: float):
        # bounds = list of (lower bound, upper bound) pairs
        super(DenseScalarPi, self).__init__()
        self.d_state = Dense(embed_dim)
        self.d_out = Dense(len(bounds))
        self.net = DenseNetwork(layer_sizes, 1, drop_rate)
        self.offset = tf.constant([vi[0] for vi in bounds], tf.float32)
        self.ranges = tf.constant([vi[1] - vi[0] for vi in bounds], tf.float32)

    def call(self, state_t: List[tf.Tensor]):
        x_s = self.d_state(state_t[0])
        yp = self.net(x_s)
        raw_act = self.d_out(yp)
        # apply bounds via sigmoid:
        return (self.ranges * tf.math.sigmoid(raw_act)) + self.offset


def _fake_data_reward_button(num_samples: int = 5000,
                             r = 1.):
    # discrete case
    # "reward button":
    # if agent executes the reward action --> gets reward
    rng = npr.default_rng(42)
    states = rng.random((num_samples, 2)) - 0.5
    action0 = (rng.random((num_samples,)) > 0.5) * 1
    actions = action0
    rewards = action0 * r
    terms = np.zeros((num_samples,))
    return (states[:-1], states[1:], actions[:-1],
                  rewards[:-1], terms[:-1])


def _fake_data_reward_button_cont(num_samples: int = 5000,
                                  action_dims: int = 2,
                                  state_dims: int = 3):
    # continuous case "reward button":
    # reward = sum(action dims)
    rng = npr.default_rng(42)
    states = rng.random((num_samples, state_dims)) - 0.5
    actions = rng.random((num_samples, action_dims)) - 0.5
    rewards = np.sum(actions, axis=1)
    terms = np.zeros((num_samples,))
    return (states[:-1], states[1:], actions[:-1],
                  rewards[:-1], terms[:-1])


def _fake_data_positive_response_cont(num_samples: int = 5000,
                                      action_dims: int = 2,
                                      state_dims: int = 3):
    # positive response == response should be correlated with sum(state)
    # reward = sum(actions) * sum(state)
    rng = npr.default_rng(42)
    states = rng.random((num_samples, state_dims)) - 0.5
    actions = rng.random((num_samples, action_dims)) - 0.5
    rewards = np.sum(actions, axis=1) * np.sum(states, axis=1)
    terms = np.zeros((num_samples,))
    return (states[:-1], states[1:], actions[:-1],
                  rewards[:-1], terms[:-1])


class TestDQN(TestCase):

    def setUp(self) -> None:
        eval_model = DenseScalar()
        rng = npr.default_rng(42)
        run_iface = RunIface(eval_model, 2, 0.25, rng)
        self.QA = QAgent(run_iface,
                         eval_model, DenseScalar(),
                         rng,
                         2, 2,
                         gamma=0.9,
                         tau=.5)
        # load in data:
        self.r = 2
        dat = _fake_data_reward_button(100, r=self.r)
        for i in range(len(dat[0])):
            self.QA.save_data([dat[0][i]], [dat[1][i]], dat[2][i],
                              dat[3][i], dat[4][i])

    def test_dset_build(self):
        dset = self.QA._draw_sample().batch(32)
        for v in dset:
            self.assertEqual(tf.shape(v["reward"]).numpy(), (32,))
            self.assertTrue(tf.reduce_all(tf.shape(v["state"]) ==
                                          tf.constant([32, 2], dtype=tf.int32)))
            vout = self.QA.kmodel(v)
            self.assertTrue(len(tf.shape(vout["loss"]).numpy()) == 0)
            break

    def test_reward_button(self):
        # train each model a few times
        for _ in range(200):
            self.QA.train()
            self.QA._copy_model()
        # expectation?
        # Q learning: Q(t) = r_{t+1} + gamma * max_{a} [ Q(t+1) ]
        #       with 'reward button' --> can get reward in any state
        #       reward sequence, assuming constant reward r:
        #           r + gamma * r + gamma^2 * r
        #           = r * (1 + gamma * gamma^2 + ...)
        #           = r * 1 / (1 - gamma) (power series)
        #   so, Q should converge to this value
        #
        # back to og eqn: (where r_{t+1} = r; else subtract r from this)
        #       sub in max_{a} [ Q(t+1) ] = r + gamma * r + ...
        #       Q(t) = r + gamma * (r + gamma + gamma^2)
        #       Q(t) = r + gamma * r + gamma^2 * r
        # in fact, this is closely related to the Bellman eqns on
        #   which Q learning is built
        exp_q = self.r * 1. / (1. - self.QA.gamma)

        for v in self.QA._draw_sample().batch(32):
            rews = v["reward"].numpy()
            q = self.QA.memory_model(v["action"], [v["state"]]).numpy()
            q_rew = np.mean(q[rews >= 0.5])
            q_no = np.mean(q[rews <= 0.5])
            print(q)
            self.assertAlmostEqual(exp_q, q_rew, places=1)
            self.assertAlmostEqual(exp_q - self.r, q_no, places=1)
            break


class TestDQNcont(TestCase):

    def setUp(self) -> None:
        rng = npr.default_rng(42)
        bounds = [(-1, 1), (-1, 1)]
        run_iface = RunIfaceCont(bounds, [0.15] * len(bounds),
                                 [0.2] * len(bounds), rng)
        def q_builder():
            return DenseScalar()
        def pi_builder():
            return DenseScalarPi(bounds, 4, [4], 0.)
        self.state_dims = 3
        self.QA = QAgent_cont(run_iface, q_builder, pi_builder,
                              rng, len(bounds), self.state_dims,
                              tau=0.1,
                              critic_lr=.01, actor_lr=.005)
        self.buffer_size = 5000
        self.QA.mem_buffer.buffer_size = self.buffer_size

    def _load_data(self, dat):
        for i in range(len(dat[0])):
            self.QA.save_data([dat[0][i]], [dat[1][i]], dat[2][i],
                                dat[3][i], dat[4][i])

    def test_dset_build(self):
        # load in data:
        self._load_data(_fake_data_reward_button_cont(num_samples=self.buffer_size,
                                                      action_dims=2,
                                                      state_dims=self.state_dims))

        dset = self.QA._draw_sample().batch(32)
        for v in dset:
            self.assertEqual(tf.shape(v["reward"]).numpy(), (32,))
            self.assertTrue(tf.reduce_all(tf.shape(v["state"]) ==
                                          tf.constant([32, self.state_dims], dtype=tf.int32)))
            vout = self.QA.kmodel(v)
            self.assertTrue(len(tf.shape(vout["loss"]).numpy()) == 0)
            break

    def test_reward_button(self):
        # load in data:
        self._load_data(_fake_data_reward_button_cont(num_samples=self.buffer_size,
                                                      action_dims=2,
                                                      state_dims=self.state_dims))

        # train each model a few times
        for _ in range(80):
            self.QA.train()
            self.QA._copy_model()

        # expected action: at upper bound == 1, 1
        # exp q: just look at diff between:
        #   1. Q(pi, state_t), 2. r + gamma * Q(pi, state_t1)
        for v in self.QA._draw_sample().batch(32):
            # action calc
            act = self.QA.piprime_model([v["state"]])
            self.assertTrue(tf.math.reduce_all(act >= 0.95))

            # q calc ~ this is basically just testing whether
            # q loss in model is calculated correctly
            q = self.QA.qprime_model(v["action"], [v["state"]])
            act_t1 = self.QA.piprime_model([v["state_t1"]])
            q_t1 = self.QA.qprime_model(act_t1, [v["state_t1"]])
            target = tf.cast(v["reward"], q_t1.dtype) + self.QA.gamma * q_t1
            self.assertTrue(tf.math.reduce_mean(tf.math.pow(target - q, 2.)) < .2)
            break

    def test_positive_response(self):
        # load in data:
        self._load_data(_fake_data_positive_response_cont(num_samples=self.buffer_size,
                                                          action_dims=2,
                                                          state_dims=self.state_dims))

        # train each model a few times
        for _ in range(160):
            self.QA.train()
            self.QA._copy_model()

        for v in self.QA._draw_sample().batch(128):
            # action calc
            s = tf.math.reduce_sum(v["state"], axis=1)
            act = tf.math.reduce_sum(self.QA.piprime_model([v["state"]]), axis=1)
            # state - action correlation ~ should be high
            corr = tf.math.reduce_mean(s * tf.cast(act, s.dtype))
            self.assertTrue(corr.numpy() > 0.3)

            # correlation with s(t+1)? shouldn't be
            s = tf.math.reduce_sum(v["state"], axis=1)
            act = tf.math.reduce_sum(self.QA.piprime_model([v["state_t1"]]), axis=1)
            corr = tf.math.reduce_mean(s * tf.cast(act, s.dtype))
            self.assertTrue(corr.numpy() < 0.3)
            break



if __name__ == "__main__":
    T = TestDQN()
    T.setUp()
    T.test_dset_build()
    T.test_reward_button()
    T = TestDQNcont()
    T.setUp()
    T.test_dset_build()
    T.test_positive_response()
    T.test_reward_button()
