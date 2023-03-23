import tensorflow as tf
from unittest import TestCase
from arch_layers.transformers import TransformerBlock, positional_embedding, Transformer


class TestPosEmbedding(TestCase):

    def test_pos_embed(self):
        # standard positional embedding ~ from Vaswani
        T = 12
        d = 6
        n = T  # this seems to be the right choice
        pe = positional_embedding(T, d, n)
        self.assertTrue(tf.math.reduce_all(tf.shape(pe) ==
                                           tf.constant([T, 2 * d], dtype=tf.int32)))


class TestTBlock(TestCase):

    def test_shape(self):
        output_dim = 6
        TB = TransformerBlock(2, 4, 4, output_dim, [8, 8])
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
        TB = TransformerBlock(2, 4, 4, output_dim, [8, 8])
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


class TestTransformer(TestCase):


    def test_tformer(self):
        num_blocks = 4
        output_dim = 5
        num_heads = 2
        key_dim = 8
        value_dim = 8
        layer_output_dim = 4
        dense_layer_sizes = [12, 12]
        Tfr = Transformer(num_blocks,
                          output_dim,
                          num_heads,
                          key_dim, value_dim, layer_output_dim,
                          dense_layer_sizes)
        batch_size = 16
        T = 24
        d = 12
        x = tf.ones([batch_size, T, d])
        y = Tfr(x)
        self.assertTrue(tf.math.reduce_all(tf.shape(y) ==
                                           tf.constant([batch_size, T, output_dim])))


if __name__ == "__main__":
    T = TestPosEmbedding()
    T.test_pos_embed()
    T = TestTBlock()
    T.test_shape()
    T.test_causal()
    T = TestTransformer()
    T.test_tformer()
