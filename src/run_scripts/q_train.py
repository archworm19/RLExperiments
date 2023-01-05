"""
    Deep Q Learning + Open AI Gym

"""
import gymnasium as gym
import numpy as np
import numpy.random as npr
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
from frameworks.agent import Agent

from run_scripts.runner import runner
from run_scripts.utils import (build_dense_qagent, build_dense_qagent_cont,
                               build_dense_qagent_distro)


@dataclass
class EnvConfig:
    env_name: str
    kwargs: Dict
    num_actions: int   # dim of action space
    num_obs: int
    run_length: int
    continuous: bool


@dataclass
class DefaultParams:
    gamma: float  # discount factor
    tau: float  # weight for model copying
    num_batch_sample: int  # number of batches pulled for training
    batch_size: int
    train_epoch: int
    step_per_train: int
    step_per_copy: int
    action_bounds: List[Tuple] = None  # only defined in continuous case


class Envs(Enum):
    cartpole = (EnvConfig("CartPole-v1", {}, 2, 4, 1000, False),
                DefaultParams(0.99, .05,
                              1, 64,
                              1,
                              1, 1))
    lunar = (EnvConfig("LunarLander-v2", {}, 4, 8, 1000, False),
             DefaultParams(0.99, .05,
                           1, 64,
                           1,
                           1, 1))
    acrobot = (EnvConfig("Acrobot-v1", {}, 3, 6, 200, False),
               DefaultParams(0.99, .05,
                              1, 64,
                              1,
                              1, 1))
    pendulum = (EnvConfig('Pendulum-v1', {}, 1, 3, 500, True),
                DefaultParams(0.99, .01,
                              1, 1, 1, 1, 1,
                              [(-2., 2.)]))
    lunar_continuous = (EnvConfig("LunarLander-v2", {"continuous": True},
                                  4, 8, 1000, True),
                        DefaultParams(0.99, .01,
                                      1, 1, 1, 1, 1,
                                      [(-1., 1.), (-1., 1.)]))


def run_and_train(env_config: EnvConfig,
                  agent: Agent,  # TODO: need more specific interface for discrete/update agent
                  num_runs: int = 300,
                  show_progress: bool = True,
                  seed_runs: int = 20,
                  step_per_train: int = 1,
                  step_per_copy: int = 1,
                  runs_per_display: int = 5,
                  debug_viz: bool = False):
    # NOTE: should work in discrete or continious case
    # run params
    run_length = env_config.run_length
    # number of seeds per 
    num_runs = num_runs

    # render modes:
    #   None(default): no render
    #   "human": continuously render in current display
    #   "rgb_array", "ansi", and a few others
    env_run = gym.make(env_config.env_name, **env_config.kwargs)
    if show_progress:
        env_disp = gym.make(env_config.env_name, render_mode="human",
                            **env_config.kwargs)
    else:
        env_disp = env_run

    rews = []
    # different seed for each run:
    for i in range(num_runs):
        if i % runs_per_display == 0:  # use display
            active_env = env_disp
            debug_set = debug_viz
            train_mode = False
        else:
            active_env = env_run
            debug_set = False
            train_mode = True

        if i < seed_runs:
            train_mode = False

        active_env.reset(seed=npr.randint(num_runs))
        # gather run data
        rewards = runner(active_env, agent, run_length,
                            debug=debug_set, train_mode=train_mode,
                            step_per_train=step_per_train,
                            step_per_copy=step_per_copy)
        rews.append(np.sum(np.array(rewards)))

        print(rews[-1])

        # signal end of epoch
        agent.end_epoch()

    env_run.close()
    env_disp.close()
    return rews


if __name__ == "__main__":
    (env_config, def_params) = Envs.cartpole.value
    # (env_config, def_params) = Envs.lunar.value
    # (env_config, def_params) = Envs.acrobot.value
    # (env_config, def_params) = Envs.pendulum.value
    # (env_config, def_params) = Envs.lunar_continuous.value

    # run
    if env_config.continuous:
        agent = build_dense_qagent_cont(action_bounds=def_params.action_bounds,
                                        num_observations=env_config.num_obs,
                                        layer_sizes=[128, 64],
                                        drop_rate=0.05,
                                        gamma=def_params.gamma,
                                        tau=def_params.tau,
                                        num_batch_sample=def_params.num_batch_sample,
                                        train_epoch=def_params.train_epoch,
                                        batch_size=def_params.batch_size,
                                        )
    else:  # discrete
        # TESTING: distributional approach
        agent = build_dense_qagent_distro(num_actions=env_config.num_actions,
                                   num_observations=env_config.num_obs,
                                   layer_sizes=[128, 64],
                                   drop_rate=0.05,
                                   gamma=def_params.gamma,
                                   tau=def_params.tau,
                                   num_batch_sample=def_params.num_batch_sample,
                                   train_epoch=def_params.train_epoch,
                                   batch_size=def_params.batch_size)
    reward_seq = run_and_train(env_config, agent, num_runs=200,
                               seed_runs=5,
                               step_per_train=def_params.step_per_train,
                               step_per_copy=def_params.step_per_copy,
                               debug_viz=True)
    print(reward_seq)
