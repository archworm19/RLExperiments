"""Forward Exploration Testing"""
import tensorflow as tf
from typing import List
from unittest import TestCase
from tensorflow.keras.layers import Dense, Layer
from frameworks.layer_signatures import VectorStateModel, MapperModel, VectorModel
from frameworks.exploration_forward import forward_surprisal, inverse_dynamics_error, randomnet_error


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


class Mapper(Layer):

    def __init__(self, encode_dims: int):
        super(Mapper, self).__init__()
        self.layer = Dense(encode_dims)

    def call(self, x: tf.Tensor):
        return self.layer(x)


class TestForwardSurprisal(TestCase):

    def test_forward_surprisal(self):
        batch_size = 16
        encode_dims = 8
        state = [tf.ones((batch_size, 5)), tf.ones((batch_size, 7))]
        state_t1 = [tf.ones((batch_size, 5)), tf.ones((batch_size, 7))]
        action = tf.ones((batch_size, 3))
        # case 1: error > 0
        F = VectorModel(ForwardModel(encode_dims, 2))
        E = VectorStateModel(Encoder(encode_dims, 2))
        err, check_bit = forward_surprisal(F, E, state, state_t1, action)
        self.assertTrue(tf.math.reduce_all(tf.shape(err) == tf.constant([batch_size])))
        self.assertTrue(tf.math.reduce_all(err >= 0.))
        # test independence
        self.assertTrue(tf.math.reduce_sum(tf.math.abs(err[:-1] - err[1:])) < .001)
        self.assertTrue(check_bit)
        # case 2: scale by 0 --> error should be 0
        F = VectorModel(ForwardModel(encode_dims, 2, 0.))
        E = VectorStateModel(Encoder(encode_dims, 2, 0.))
        err, check_bit = forward_surprisal(F, E, state, state_t1, action)
        self.assertTrue(tf.math.reduce_sum(err) < .001)
        self.assertTrue(check_bit)


class TestInvDyn(TestCase):

    def test_inverse_dynamics(self):
        batch_size = 16
        encode_dims = 8
        action_dims = 3
        state = [tf.ones((batch_size, 5)), tf.ones((batch_size, 7))]
        state_t1 = [tf.ones((batch_size, 5)), tf.ones((batch_size, 7))]
        action = tf.ones((batch_size, action_dims))
        E = VectorStateModel(Encoder(encode_dims, 2))
        EA = MapperModel(Mapper(action_dims))
        err, check_bit = inverse_dynamics_error(E, EA, state, state_t1, action)
        self.assertTrue(tf.math.reduce_all(tf.shape(err) == tf.constant([batch_size])))
        self.assertTrue(tf.math.reduce_all(err >= 0.))
        # test independence
        self.assertTrue(tf.math.reduce_sum(tf.math.abs(err[:-1] - err[1:])) < .001)
        self.assertTrue(check_bit)


class TestRandNet(TestCase):

    def test_random_net_error(self):
        batch_size = 16
        encode_dims = 4
        state = [tf.ones((batch_size, 5)), tf.ones((batch_size, 7))]
        Er = VectorStateModel(Encoder(encode_dims, 2))
        El = VectorStateModel(Encoder(encode_dims, 2))
        err, check_bit = randomnet_error(Er, El, state)
        self.assertTrue(tf.math.reduce_all(tf.shape(err) == tf.constant([batch_size])))
        self.assertTrue(tf.math.reduce_all(err >= 0.))
        self.assertTrue(check_bit)


if __name__ == "__main__":
    T = TestForwardSurprisal()
    T.test_forward_surprisal()
    T = TestInvDyn()
    T.test_inverse_dynamics()
    T = TestRandNet()
    T.test_random_net_error()
