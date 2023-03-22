import tensorflow as tf
from unittest import TestCase
from arch_layers.transformers import TransformerBase


class TestTBase(TestCase):

    def test_shape(self):
        output_dim = 6
        TB = TransformerBase(2, 4, 4, output_dim, [8, 8])
        batch_size = 4
        T = 10
        x = tf.ones([batch_size, T, output_dim])
        vout = TB(x)
        # --> batch_size x T1 x TB.output_dim
        self.assertTrue(tf.math.reduce_all(tf.shape(vout) ==
                                           tf.constant([batch_size, T, output_dim], dtype=tf.int32)))

    def test_causal(self):
        # TODO: test causal mask
        #   how? modify following tokens --> shouldn't effect target
        output_dim = 6
        TB = TransformerBase(2, 4, 4, output_dim, [8, 8])
        batch_size = 4
        T = 10
        x0 = tf.ones([batch_size, T, output_dim])
        v0 = TB(x0, use_causal_mask=True)

        for i in range(T):
            xi = tf.concat([tf.ones([batch_size, i, output_dim]),
                            tf.ones([batch_size, T - i, output_dim]) * 2.],
                            axis=1)
            vi = TB(xi, use_causal_mask=True)
            self.assertTrue(tf.math.reduce_all(v0[:, :i] == vi[:, :i]))
            self.assertTrue(tf.math.reduce_all(v0[:, i:] != vi[:, i:]))


if __name__ == "__main__":
    T = TestTBase()
    T.test_shape()
    T.test_causal()
