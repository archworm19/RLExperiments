"""Q learning tests"""
import tensorflow as tf
from typing import List
from unittest import TestCase
from tensorflow.keras.layers import Layer
from frameworks.q_learning import (calc_q_error_sm, _greedy_select, calc_q_error_huber,
                                   calc_q_error_cont)


class AModel(Layer):
    # returns action_t[:, target_idx]

    def __init__(self, target_idx: int):
        super(AModel, self).__init__()
        self.target_idx = target_idx

    def call(self, action_t: tf.Tensor, state_t: List[tf.Tensor]):
        return action_t[:, self.target_idx]


class TestHuber(TestCase):

    def test_huber(self):
        q1 = tf.constant([1., 2., 3., 4., 2.5])
        q2 = tf.constant([3., 2. , 1., 0., 2.0])
        reward = tf.constant([0., 0., 0., 0., 0.])
        err, _ = calc_q_error_huber(q1, q2, reward, 1.)
        target = tf.constant([1.5, 0., 1.5, 3.5, 0.124999])
        self.assertTrue(tf.math.reduce_all(tf.round(100 * err) ==
                                           tf.round(100 * target)))


class TestQL(TestCase):

    def setUp(self) -> None:
        self.model0 = AModel(0)
        self.model1 = AModel(1)

    def test_greedy_select(self):
        state_t1 = tf.constant([[1., 2.],
                                [3., 4.],
                                [1., 2.]])
        max_q, qs =_greedy_select(self.model0, 3, state_t1)
        self.assertTrue(tf.math.reduce_all(max_q == tf.constant([0, 0], tf.int64)))
        self.assertTrue(tf.math.reduce_all(qs == tf.constant([[1., 0., 0.],
                                                              [1., 0., 0.]])))
        max_q, qs =_greedy_select(self.model1, 3, state_t1)
        self.assertTrue(tf.math.reduce_all(max_q == tf.constant([1, 1], tf.int64)))
        self.assertTrue(tf.math.reduce_all(qs == tf.constant([[0., 1., 0.],
                                                              [0., 1., 0.]])))

    def test_double_q(self):
        gamma = 0.8
        batch_size = 4
        reward_t1 = tf.zeros(batch_size)
        state_t = [tf.zeros(batch_size)]
        state_t1 = [tf.zeros(batch_size)]
        action_t = tf.constant([[0, 0, 1] * batch_size], dtype=tf.float32)
        terminatiion = tf.zeros(batch_size)
        num_action = 3

        # same model (within reward = 0)
        # --> max_a[Q(t+1)] = 1 for each sample
        # --> target = 0 + gamma * 1
        # if feed in action = 0 --> error = gamma^2
        qerr, yt = calc_q_error_sm(self.model0, self.model0, self.model0,
                                   action_t,
                                   reward_t1,
                                   state_t,
                                   state_t1,
                                   terminatiion,
                                   num_action,
                                   gamma,
                                   huber=False)
        # 100 = precision
        self.assertTrue(tf.math.reduce_all(tf.round(100 * qerr) ==
                                           tf.round(100 * gamma**2.)))
        self.assertTrue(tf.math.reduce_all(tf.round(100 * yt) ==
                                           tf.round(100 * gamma)))

        # different model (with 0 reward)
        # --> max_a[Q(t+1)] = 0 for each sample
        # --> target = 0 + gamma * 0
        qerr, yt = calc_q_error_sm(self.model0, self.model0, self.model1,
                                   action_t,
                                   reward_t1,
                                   state_t,
                                   state_t1,
                                   terminatiion,
                                   num_action,
                                   gamma)
        self.assertTrue(tf.math.reduce_all(tf.round(100 * qerr) ==
                                           tf.round(100 * 0.)))
        self.assertTrue(tf.math.reduce_all(tf.round(100 * yt) ==
                                           tf.round(100 * 0.)))


class QModel(Layer):

    def __init__(self):
        super(QModel, self).__init__()

    def call(self, action: tf.Tensor, state: List[tf.Tensor]):
        # assumes: state and action space have same dimensionality
        # both are batch_size x ...
        return action + state[0]


class PiModel(Layer):
    # returns action_t[:, target_idx]

    def __init__(self):
        super(PiModel, self).__init__()

    def call(self, state_t: List[tf.Tensor]):
        return state_t[0]


class TestQLcont(TestCase):

    # TODO: test design
    # > positive control: use same Q, pi but add reward/gamma offset to prime versions
    # > error minimization: ???

    def setUp(self) -> None:
        self.Q = QModel()
        self.pi = PiModel()

    def test_positive_control(self):
        # def: Q(pi, s) = r + gamma * Q(pi, s)
        # --> error = d(Q - [r + gamma * Q])
        #           = d(Q * (1 - gamma) - r)
        batch_size = 8
        s = tf.random.uniform([batch_size])
        r = tf.random.uniform([batch_size])
        term = tf.zeros([batch_size])
        gamma = 0.85
        Qval = self.Q(self.pi([s]), [s])
        err, Y_t = calc_q_error_cont(self.Q, self.pi, self.Q, self.pi,
                                     r, [s], [s], term, gamma, huber=False)
        target = tf.pow(Qval * (1. - gamma) - r, 2.)
        self.assertTrue(tf.math.reduce_all(tf.round(err * 100) ==
                                           tf.round(target * 100)))

    def test_negative_control(self):
        # positive control condition fails when state t1 is changed
        #   <-- models is operating in a different state
        batch_size = 8
        s = tf.random.uniform([batch_size])
        s_t1 = s + 1.
        r = tf.random.uniform([batch_size])
        term = tf.zeros([batch_size])
        gamma = 0.85
        Qval = self.Q(self.pi([s]), [s])
        err, Y_t = calc_q_error_cont(self.Q, self.pi, self.Q, self.pi,
                                     r, [s], [s_t1], term, gamma, huber=False)
        target = tf.pow(Qval * (1. - gamma) - r, 2.)
        self.assertFalse(tf.math.reduce_all(tf.round(err * 100) ==
                                            tf.round(target * 100)))


if __name__ == "__main__":
    T = TestHuber()
    T.test_huber()
    T = TestQL()
    T.setUp()
    T.test_greedy_select()
    T.test_double_q()
    T = TestQLcont()
    T.setUp()
    T.test_positive_control()
    T.test_negative_control()
