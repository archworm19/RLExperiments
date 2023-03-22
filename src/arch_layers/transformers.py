"""Transformers for time series
"""
import tensorflow as tf
from typing import List
from tensorflow.keras.layers import Layer, MultiHeadAttention, LayerNormalization, Add
from arch_layers.simple_networks import DenseNetwork

# TODO: attention layers (base, linformer)


class TransformerBase(Layer):
    # self-attention only

    def __init__(self,
                 num_heads: int,
                 key_dim: int, value_dim: int, output_dim: int,
                 dense_layer_sizes: List[int],
                 attention_dropout: float = 0.,
                 dense_dropout: float = 0.):
        super(TransformerBase, self).__init__()
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


# TODO: whole transformer

# TODO: how to do residuals???
#   F = transformer layer
#   input to layer i+1 = F_i(x) + x
#       where F_i(x) and x are not normed
#   effect? each layer recieves a sum of outputs
#       of all previous layers

if __name__ == "__main__":
    output_dim = 6
    TB = TransformerBase(2, 4, 4, output_dim, [8, 8])
    batch_size = 4
    T = 10
    x = tf.ones([batch_size, T, output_dim])
    vout = TB(x)
    # --> batch_size x T1 x TB.output_dim
    print(vout)
