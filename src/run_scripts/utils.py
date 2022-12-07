import numpy as np
import numpy.random as npr
import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer
from arch_layers.simple_networks import DenseNetwork
from typing import List

from frameworks.agent import RunData
from agents.q_agents import QAgent, RunIface


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
                       drop_rate: float = 0.1,
                       gamma: float = 0.6,
                       num_batch_sample: int = 1,
                       tau: float = 0.15):
    rng = npr.default_rng(42)
    free_model = DenseScalar(embed_dim, layer_sizes, drop_rate)
    memory_model = DenseScalar(embed_dim, layer_sizes, drop_rate)
    run_iface = RunIface(memory_model, num_actions, 1., rng)
    return QAgent(run_iface,
                  free_model, memory_model,
                  rng,
                  num_actions, num_observations, gamma=gamma,
                  tau=tau,
                  num_batch_sample=num_batch_sample)


def purge_run_data(struct: RunData, max_len: int):
    # keep the last [max_len] elements of run_dat
    if np.shape(struct.states)[0] > max_len:
        return RunData(struct.states[-max_len:],
                       struct.states_t1[-max_len:],
                       struct.actions[-max_len:],
                       struct.rewards[-max_len:],
                       struct.termination[-max_len:])
    return struct
