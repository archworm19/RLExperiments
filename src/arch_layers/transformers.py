"""Transformers for time series
"""
import tensorflow as tf
from typing import List
from tensorflow.keras.layers import Layer, MultiHeadAttention, LayerNormalization, Add, Dense
from arch_layers.simple_networks import DenseNetwork


# utilities


def positional_embedding(T: int, d: int, n: int = 10000,
                         dtype=tf.float32):
    # fourier-based embeddings from OG transformer paper
    #   P(k, 2i) = sin(k / n^(2i / d))
    #   P(k, 2i + 1) = cos(k / n^(2i / d))
    #   k = position of an object in the sequence
    #   d = dimension of output embedding space
    #   n = hyperparam scalar
    # Returns: T x (2 * d)
    x = tf.range(T, dtype=dtype)
    basis = tf.range(d, dtype=dtype)
    # --> shape = d
    denom = tf.pow(n, tf.math.divide(2. * basis, d))
    # --> shape = T x d for both
    s = tf.math.sin(tf.math.divide(tf.expand_dims(x, 1),
                                   tf.expand_dims(denom, 0)))
    c = tf.math.cos(tf.math.divide(tf.expand_dims(x, 1),
                                   tf.expand_dims(denom, 0)))
    return tf.concat([s, c], axis=1)


# TODO: attention layers (base, linformer)


class TransformerBlock(Layer):
    # self-attention only
    # Pre-LN transformer
    #   implemented according to Xiong et al, 2020
    #   > norm before mha
    #   > norm before FF

    def __init__(self,
                 num_heads: int,
                 key_dim: int, value_dim: int, output_dim: int,
                 dense_layer_sizes: List[int],
                 attention_dropout: float = 0.,
                 dense_dropout: float = 0.):
        super(TransformerBlock, self).__init__()
        # NOTE: causal mask = call argument
        self.value_dim = value_dim
        self.output_dim = output_dim
        self.ladd = Add()
        self.lnorm = LayerNormalization(axis=-1)
        self.mha = MultiHeadAttention(num_heads, key_dim, value_dim,
                                      dropout=attention_dropout)
        self.dense_net = DenseNetwork(dense_layer_sizes, output_dim, dense_dropout)

    def call(self, x: tf.Tensor, train_mode: bool = False,
             use_causal_mask: bool = False) -> tf.Tensor:
        # TODO: we add in the non-normed space before norming
        #       this is NOT how the original paper works...
        """
        self attention

        Args:
            x (tf.Tensor): input. batch_size x T x self.output_dim
                NOT normed

        tf.Tensor: output of dense network
            batch_size x T x self.output_dim
        """
        # layernorm at beginning ~ change from og paper
        # normalize within each timepoint
        xnorm = self.lnorm(x)

        # --> batch_size x T x self.output_dim
        v = self.mha(xnorm, xnorm, xnorm,
                     use_causal_mask=use_causal_mask,
                     training=train_mode)

        # add then norm
        v_add = self.ladd([x, v])
        a_out = self.lnorm(v_add)

        # run each timepoint through dense_net independently
        # --> (batch_size * T) x d
        d = tf.shape(a_out)[-1]
        T = tf.shape(a_out)[1]
        raw_out = self.dense_net(tf.reshape(a_out, [-1, d]))
        # from (batch_size * T) x output_dim --> batch_size x T x output_dim
        return self.ladd([tf.reshape(raw_out, [-1, T, self.output_dim]), v_add])


# whole transformer(s)


class Transformer(Layer):
    # self-attention only
    # uses original positional encoding system from Vaswani et al

    def __init__(self,
                 num_blocks: int,
                 output_dim: int,
                 num_heads: int,
                 key_dim: int, value_dim: int, layer_output_dim: int,
                 dense_layer_sizes: List[int],
                 attention_dropout: float = 0.,
                 dense_dropout: float = 0.,
                 T_exp: int = 100):
        # T_exp = expected T length
        super(Transformer, self).__init__()
        self.layer_output_dim = layer_output_dim
        self.output_dim = output_dim
        self._blcks = [TransformerBlock(num_heads, key_dim, value_dim, layer_output_dim,
                                        dense_layer_sizes, attention_dropout, dense_dropout)
                       for _ in range(num_blocks)]
        # need to transform input into transformer.output_dim
        self._imap = Dense(layer_output_dim, activation="linear")
        # final dense mapping!
        self._omap = Dense(output_dim, activation="linear")
        # positional embedding params:
        # get positional embedding
        self._ped = int(T_exp / 2)
        self._pen = T_exp

    def call(self, x: tf.Tensor, train_mode: bool = False,
             use_causal_mask: bool = False) -> tf.Tensor:
        """
        self attention
        NOTE: currently only feeding positional embedding into first layer
            ez change to feed it into all layers

        Args:
            x (tf.Tensor): input. batch_size x T x d
                NOT normed

        tf.Tensor: output of dense network
            batch_size x T x self.output_dim
        """
        batch_size = tf.shape(x)[0]
        T = tf.shape(x)[1]
        d = tf.shape(x)[2]

        # add positional embedding:
        # --> T x (2 * d_ped)
        pe = positional_embedding(T, self._ped, self._pen)
        pe_re = tf.tile(tf.expand_dims(pe, 0), [batch_size, 1, 1])
        # --> batch_size x T x (d_ped + d)
        x = tf.concat([x, pe_re], axis=2)

        # use dense layer --> map to batch_size x T x layer_output_dim
        x_re = tf.reshape(x, [-1, d + (2 * self._ped)])
        x = tf.reshape(self._imap(x_re), [-1, T, self.layer_output_dim])

        # run through transformer blocks
        for blk in self._blcks:
            x = blk(x, train_mode=train_mode, use_causal_mask=use_causal_mask)

        # final dense mapping --> batch_size x T x self.output_dim
        x_re = tf.reshape(x, [-1, self.layer_output_dim])
        return tf.reshape(self._omap(x_re), [-1, T, self.output_dim])
