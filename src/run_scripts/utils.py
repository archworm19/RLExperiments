import numpy.random as npr
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from frameworks.layer_signatures import ScalarModel, ScalarStateModel, DistroModel
from arch_layers.simple_networks import DenseNetwork
from typing import List, Tuple

from agents.q_agents import QAgent, RunIface, QAgent_cont, RunIfaceCont, QAgent_distro


class DenseScalar(ScalarModel):
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
    def build_q():
        return DenseScalar(embed_dim, layer_sizes, drop_rate)
    run_iface = RunIface(num_actions, 1., rng)
    return QAgent(run_iface,
                  build_q,
                  rng,
                  num_actions, num_observations, gamma=gamma,
                  tau=tau,
                  num_batch_sample=num_batch_sample,
                  train_epoch=train_epoch,
                  batch_size=batch_size)


def build_dense_qagent_cont(action_bounds: List[float] = [(-1, 1), (-1, 1)],
                       num_observations: int = 8,
                       embed_dim: int = 4,
                       layer_sizes: List[int] = [128, 64],
                       drop_rate: float = 0.1,
                       gamma: float = 0.6,
                       num_batch_sample: int = 1,
                       tau: float = 0.15,
                       train_epoch: int = 1,
                       batch_size: int = 128,
                       sigma: float = 0.2,
                       theta: float = 0.15):
    # continious control Q agent
    rng = npr.default_rng(42)
    def build_q():
        return DenseScalar(embed_dim, layer_sizes, drop_rate)
    def build_pi():
        return DenseScalarPi(action_bounds, embed_dim, layer_sizes, drop_rate)
    run_iface = RunIfaceCont(action_bounds,
                             [theta] * len(action_bounds),
                             [sigma] * len(action_bounds),
                             rng)
    Qa = QAgent_cont(run_iface, build_q, build_pi, rng,
                        len(action_bounds),
                        num_observations,
                        gamma=gamma,
                        tau=tau,
                        batch_size=batch_size,
                        num_batch_sample=num_batch_sample,
                        train_epoch=train_epoch,
                        )
    return Qa


def build_dense_qagent_distro(num_actions: int = 4,
                              num_observations: int = 8,
                              num_atoms: int = 51,
                              Vmin: float = -20.,
                              Vmax: float = 20.,
                              embed_dim: int = 4,
                              layer_sizes: List[int] = [32, 16],
                              drop_rate: float = 0.1,
                              gamma: float = 0.6,
                              num_batch_sample: int = 1,
                              tau: float = 0.15,
                              train_epoch: int = 1,
                              batch_size: int = 128):
    # discrete control + distribution approach
    # middle atom = index of middle atom
    #       Ex: if 0 --> right skewed distro
    rng = npr.default_rng(42)
    def build_q():
        return DenseDistro(embed_dim, layer_sizes, drop_rate, num_atoms)
    run_iface = RunIface(num_actions, 1., rng)
    ind0 = np.argmin(np.fabs(np.linspace(Vmin, Vmax, num_atoms)))
    v0 = [0] * num_atoms
    v0[ind0] = 1.
    vector0 = tf.constant(v0,
                          tf.float32)
    return QAgent_distro(run_iface,
                         build_q,
                         rng,
                         num_actions, num_observations,
                         vector0,
                         Vmin=Vmin,
                         Vmax=Vmax,
                         gamma=gamma,
                         tau=tau,
                         num_batch_sample=num_batch_sample,
                         train_epoch=train_epoch,
                         batch_size=batch_size,
                         learning_rate=.002,
                         rand_act_decay=0.95)


if __name__ == "__main__":
    # quick assembly test
    # Q = build_dense_qagent_cont()
    Q = build_dense_qagent_distro()
