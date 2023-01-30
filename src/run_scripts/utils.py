"""Sub-models used by model builders"""
import tensorflow as tf
from tensorflow.keras.layers import Dense
from frameworks.layer_signatures import ScalarModel, ScalarStateModel, DistroModel
from arch_layers.simple_networks import DenseNetwork
from typing import List, Tuple


class DenseScalar(ScalarModel):
    # > run all states thru embeddings
    # > concat
    # > run thru dense network 
    def __init__(self,
                 embed_dims: List[int],
                 layer_sizes: List[int],
                 drop_rate: float):
        super(DenseScalar, self).__init__()
        self.d_act = Dense(embed_dims[0])
        self.d_states = [Dense(ed) for ed in embed_dims]
        self.net = DenseNetwork(layer_sizes, 1, drop_rate)

    def call(self, action_t: tf.Tensor, state_t: List[tf.Tensor]):
        x_a = self.d_act(action_t)
        x_s = [dse(s) for dse, s in zip(self.d_states, state_t)]
        yp = self.net(tf.concat([x_a] + x_s, axis=1))
        return yp[:, 0]  # to scalar


class DenseScalarPi(ScalarStateModel):
    # NOTE: unbounded outputs
    # pi: pi(a | s)
    def __init__(self,
                 output_dims: int,
                 embed_dims: List[int],
                 layer_sizes: List[int],
                 drop_rate: float,
                 bounds: List[Tuple[int]] = None):
        # bounds = list of (lower bound, upper bound) pairs
        super(DenseScalarPi, self).__init__()
        self.d_states = [Dense(ed) for ed in embed_dims]
        self.d_out = Dense(output_dims)
        self.net = DenseNetwork(layer_sizes, 1, drop_rate)
        if bounds is not None:
            assert len(bounds) == output_dims, "dim mismatch"
            self.offsets = tf.constant([bi[0] for bi in bounds], dtype=tf.float32)
            self.ranges = tf.constant([bi[1] - bi[0] for bi in bounds], dtype=tf.float32)
            self.activation = tf.math.sigmoid
        else:
            self.offsets = 0.
            self.ranges = 1.
            self.activation = lambda x: x

    def call(self, state_t: List[tf.Tensor]):
        x_s = [dse(s) for dse, s in zip(self.d_states, state_t)]
        yp = self.net(tf.concat(x_s, axis=1))
        return self.activation(self.d_out(yp)) * self.ranges + self.offsets


class DenseDistro(DistroModel):
    def __init__(self,
                 embed_dims: List[int],
                 layer_sizes: List[int],
                 drop_rate: float,
                 num_atoms: int = 51):
        # NOTE: use sigmoid_scale to avoid overflows
        super(DenseDistro, self).__init__()
        self.d_act = Dense(embed_dims[0])
        self.d_states = [Dense(ed) for ed in embed_dims]
        self.net = DenseNetwork(layer_sizes, num_atoms, drop_rate)

    def call(self, action_t: tf.Tensor, state_t: List[tf.Tensor]):
        # NOTE: only uses 0th tensor in state_t
        x_a = self.d_act(action_t)
        x_s = [dse(s) for dse, s in zip(self.d_states, state_t)]
        yp = self.net(tf.concat([x_a] + x_s, axis=1))
        return yp
