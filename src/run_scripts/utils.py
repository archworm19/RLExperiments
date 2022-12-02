import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer
from arch_layers.simple_networks import DenseNetwork
from typing import List

from agents.q_agents import QAgent


class DenseScalar(Layer):
    def __init__(self,
                 embed_dim: int,
                 layer_sizes: List[int],
                 drop_rate: float):
        super(DenseScalar, self).__init__()
        self.d_act = Dense(embed_dim)
        self.d_state = Dense(embed_dim)
        self.net = DenseNetwork(layer_sizes, 1, drop_rate)

    def call(self, action_t: tf.Tensor, state_t: List[tf.Tensor]):
        x_a = self.d_act(action_t)
        x_s = self.d_state(state_t[0])
        yp = self.net(tf.concat([x_a, x_s], axis=1))
        return yp[:, 0]  # to scalar


def build_dense_qagent(num_actions: int = 4,
                       num_observations: int = 8,
                       embed_dim: int = 4,
                       layer_sizes: List[int] = [32, 16],
                       drop_rate: float = 0.1):
    return QAgent(DenseScalar(embed_dim, layer_sizes, drop_rate),
                  num_actions, num_observations)
