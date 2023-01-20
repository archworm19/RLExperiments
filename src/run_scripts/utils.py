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


class DenseDistro(DistroModel):
    def __init__(self,
                 embed_dim: int,
                 layer_sizes: List[int],
                 drop_rate: float,
                 num_atoms: int = 51):
        # NOTE: use sigmoid_scale to avoid overflows
        super(DenseDistro, self).__init__()
        self.d_act = Dense(embed_dim)
        self.d_state = Dense(embed_dim)
        self.net = DenseNetwork(layer_sizes, num_atoms, drop_rate)

    def call(self, action_t: tf.Tensor, state_t: List[tf.Tensor]):
        # NOTE: only uses 0th tensor in state_t
        x_a = self.d_act(action_t)
        x_s = self.d_state(state_t[0])
        yp = self.net(tf.concat([x_a, x_s], axis=1))
        return yp
