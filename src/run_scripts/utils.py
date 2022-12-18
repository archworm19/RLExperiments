import numpy.random as npr
import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer
from arch_layers.simple_networks import DenseNetwork
from typing import List, Tuple

from agents.q_agents import QAgent, RunIface, QAgent_cont, RunIfaceCont


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
        # NOTE: only uses 0th tensor in state_t
        x_a = self.d_act(action_t)
        x_s = self.d_state(state_t[0])
        yp = self.net(tf.concat([x_a, x_s], axis=1))
        return yp[:, 0]  # to scalar


class DenseScalarPi(Layer):
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


def build_dense_qagent(num_actions: int = 4,
                       num_observations: int = 8,
                       embed_dim: int = 4,
                       layer_sizes: List[int] = [32, 16],
                       drop_rate: float = 0.1,
                       gamma: float = 0.6,
                       num_batch_sample: int = 1,
                       tau: float = 0.15,
                       train_epoch: int = 1,
                       batch_size: int = 128):
    rng = npr.default_rng(42)
    free_model = DenseScalar(embed_dim, layer_sizes, drop_rate)
    memory_model = DenseScalar(embed_dim, layer_sizes, drop_rate)
    run_iface = RunIface(memory_model, num_actions, 1., rng)
    return QAgent(run_iface,
                  free_model, memory_model,
                  rng,
                  num_actions, num_observations, gamma=gamma,
                  tau=tau,
                  num_batch_sample=num_batch_sample,
                  train_epoch=train_epoch,
                  batch_size=batch_size)


def build_dense_qagent_cont(action_bounds: List[float] = [(-1, 1), (-1, 1)],
                       num_observations: int = 8,
                       embed_dim: int = 4,
                       layer_sizes: List[int] = [32, 16],
                       drop_rate: float = 0.1,
                       gamma: float = 0.6,
                       num_batch_sample: int = 1,
                       tau: float = 0.15,
                       train_epoch: int = 1,
                       batch_size: int = 128):
    # continious control Q agent
    rng = npr.default_rng(42)
    def build_q():
        return DenseScalar(embed_dim, layer_sizes, drop_rate)
    def build_pi():
        return DenseScalarPi(action_bounds, embed_dim, layer_sizes, drop_rate)
    run_iface = RunIfaceCont(action_bounds, 0.5, rng)
    Qa = QAgent_cont(run_iface, build_q, build_pi, rng,
                        len(action_bounds),
                        num_observations,
                        gamma=gamma,
                        tau=tau,
                        batch_size=batch_size,
                        num_batch_sample=num_batch_sample,
                        train_epoch=train_epoch
                        )
    return Qa


if __name__ == "__main__":
    # quick assembly test
    Q = build_dense_qagent_cont()
