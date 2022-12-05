"""Testing qagents on fake tasks"""
import numpy as np
import numpy.random as npr
import tensorflow as tf
from typing import List
from unittest import TestCase
from tensorflow.keras.layers import Dense, Layer
from frameworks.agent import RunData
from agents.q_agents import QAgent
from arch_layers.simple_networks import DenseNetwork


class DenseScalar(Layer):
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


def _fake_data_reward_button(num_samples: int = 5000,
                             r = 1.):
    # "reward button":
    # if agent executes the reward action --> gets reward
    rng = npr.default_rng(42)
    states = rng.random((num_samples, 2)) - 0.5
    action0 = (rng.random((num_samples,)) > 0.5) * 1
    # --> one-hot
    actions = np.vstack((action0, 1. - action0)).T
    rewards = action0 * r
    terms = np.zeros((num_samples,))
    dat = RunData(states[:-1], states[1:], actions[:-1],
                  rewards[:-1], terms[:-1])
    return dat


class TestDQN(TestCase):

    def setUp(self) -> None:
        self.QA = QAgent(DenseScalar(), DenseScalar(),
                         2, 2,
                         gamma=0.9,
                         num_batch_sample=1,
                         batch_size=128,
                         tau=0.01)

    def test_dset_build(self):
        dat = _fake_data_reward_button(100)
        dset = self.QA._build_dset(dat).batch(32)
        for v in dset:
            self.assertEqual(tf.shape(v["reward"]).numpy(), (32,))
            self.assertTrue(tf.reduce_all(tf.shape(v["state"]) ==
                                          tf.constant([32, 2], dtype=tf.int32)))
            vout = self.QA.kmodel(v)
            self.assertTrue(len(tf.shape(vout["loss"]).numpy()) == 0)
            break

    def test_reward_button(self):
        r = 2.
        dat = _fake_data_reward_button(5000, r=r)
        # train each model a few times
        for z in range(1000):
            self.QA.train(dat, debug=True)
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
        exp_q = r * 1. / (1. - self.QA.gamma)

        for v in self.QA._build_dset(dat).batch(32):
            rews = v["reward"].numpy()
            q = self.QA.eval_model(v["action"], [v["state"]]).numpy()
            q_rew = np.mean(q[rews >= 0.5])
            q_no = np.mean(q[rews <= 0.5])
            print(q)
            self.assertAlmostEqual(exp_q, q_rew, places=1)
            self.assertAlmostEqual(exp_q - r, q_no, places=1)
            break


if __name__ == "__main__":
    T = TestDQN()
    T.setUp()
    T.test_dset_build()
    T.test_reward_button()
