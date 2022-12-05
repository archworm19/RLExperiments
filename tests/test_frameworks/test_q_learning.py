"""Q learning tests"""
import tensorflow as tf
from typing import List
from unittest import TestCase
from tensorflow.keras.layers import Layer
from frameworks.q_learning import calc_q_error_sm, _greedy_select


class AModel(Layer):
    # returns action_t[:, target_idx]

    def __init__(self, target_idx: int):
        super(AModel, self).__init__()
        self.target_idx = target_idx

    def call(self, action_t: tf.Tensor, state_t: List[tf.Tensor]):
        return action_t[:, self.target_idx]


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
                                   gamma)
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


if __name__ == "__main__":
    T = TestQL()
    T.setUp()
    T.test_greedy_select()
    T.test_double_q()
