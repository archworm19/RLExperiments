"""Builder functions
    > specify the environment and the agent jointly
    > enforces dependencies --> compatibility guaranteed at build
    > all builder funcs return 1. env object, 2. env visualization object, 3. agent object
    > there are different enums for each "type" of environment (ex: discrete vs continuous)
    
    """
import gymnasium as gym
import numpy as np
import numpy.random as npr
import tensorflow as tf
from typing import Dict, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from agents.q_agents import QAgent, RunIface, QAgent_cont, RunIfaceCont, QAgent_distro
from run_scripts.utils import DenseScalar, DenseScalarPi, DenseDistro


@dataclass
class EnvConfig:
    env_name: str
    kwargs: Dict  # arguments to env init
    dims_actions: int  # dim of action space
    dims_obs: int  # dim of observation space
    run_length: int


@dataclass
class EnvConfigCont:
    # continuous control
    env_name: str
    kwargs: Dict  # arguments to env init
    action_bounds: List[Tuple]  # len = action space dims (tuple = (lb, ub))
    dims_obs: int  # dim of observation space
    run_length: int


class EnvsDiscrete(Enum):
    cartpole = EnvConfig("CartPole-v1", {}, 2, 4, 1000)
    lunar = EnvConfig("LunarLander-v2", {}, 4, 8, 1000)
    acrobot = EnvConfig("Acrobot-v1", {}, 3, 6, 200)


class EnvsContinuous(Enum):
    pendulum = EnvConfig('Pendulum-v1', {}, [(-2., 2.)],
                         3, 500)
    lunar_continuous = EnvConfig("LunarLander-v2", {"continuous": True},
                                  [(-1., 1.), (-1., 1.)],
                                  8, 1000)


def _build_env(env: Union[EnvConfig, EnvConfigCont]):
    env_run = gym.make(env.env_name, **env.kwargs)
    env_disp = gym.make(env.env_name, render_mode="human",
                        **env.kwargs)
    return env_run, env_disp


def build_discrete_q(env: EnvsDiscrete,
                     embed_dim: int = 4,
                     layer_sizes: List[int] = [64, 32],
                     drop_rate: float = 0.05,
                     gamma: float = 0.99,
                     num_batch_sample: int = 1,
                     tau: float = 0.05,
                     train_epoch: int = 1,
                     batch_size: int = 64):
    # build environment
    env_run, env_disp = _build_env(env.value)
    # build the discrete q learning model
    rng = npr.default_rng(42)
    def build_q():
        return DenseScalar(embed_dim, layer_sizes, drop_rate)
    run_iface = RunIface(env.value.dims_actions, 1., rng)
    agent = QAgent(run_iface,
                   build_q,
                   rng,
                   env.value.dims_actions, env.value.dims_obs,
                   gamma=gamma,
                   tau=tau,
                   num_batch_sample=num_batch_sample,
                   train_epoch=train_epoch,
                   batch_size=batch_size)
    return env_run, env_disp, agent


def build_discrete_q_atoms(env: EnvsDiscrete,
                           num_atoms: int = 51,
                           Vmin: float = -20.,
                           Vmax: float = 20.,
                           embed_dim: int = 4,
                           layer_sizes: List[int] = [64, 32],
                           drop_rate: float = 0.05,
                           gamma: float = 0.99,
                           num_batch_sample: int = 1,
                           tau: float = 0.05,
                           train_epoch: int = 1,
                           batch_size: int = 64):
    # NOTE: num_atoms, Vmin, Vmax specify the support of the distribution
    # build environment
    env_run, env_disp = _build_env(env.value)
    # build agent
    num_actions = env.value.dims_actions
    num_obs = env.value.dims_obs
    rng = npr.default_rng(42)
    def build_q():
        return DenseDistro(embed_dim, layer_sizes, drop_rate, num_atoms)
    run_iface = RunIface(num_actions, 1., rng)
    ind0 = np.argmin(np.fabs(np.linspace(Vmin, Vmax, num_atoms)))
    v0 = [0] * num_atoms
    v0[ind0] = 1.
    vector0 = tf.constant(v0,
                          tf.float32)
    Qa = QAgent_distro(run_iface,
                         build_q,
                         rng,
                         num_actions, num_obs,
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
    return env_run, env_disp, Qa


def build_continuous_q(env: EnvsContinuous,
                       embed_dim: int = 4,
                       layer_sizes: List[int] = [64, 32],
                       drop_rate: float = 0.05,
                       gamma: float = 0.99,
                       num_batch_sample: int = 1,
                       tau: float = 0.05,
                       train_epoch: int = 1,
                       batch_size: int = 64,
                       sigma: float = 0.2,
                       theta: float = 0.15):
    # NOTE: sigma, theta used for correlated noise
    env_run, env_disp = _build_env(env.value)
    # continuous control Q agent
    rng = npr.default_rng(42)
    action_bounds = env.value.action_bounds
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
                        env.value.dims_obs,
                        gamma=gamma,
                        tau=tau,
                        batch_size=batch_size,
                        num_batch_sample=num_batch_sample,
                        train_epoch=train_epoch,
                        )
    return env_run, env_disp, Qa
