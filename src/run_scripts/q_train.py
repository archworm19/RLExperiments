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


# TODO: this is no longer used
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
    cartpole = EnvConfig("CartPole-v1", {}, 2, 4, 1000, False)
    lunar = EnvConfig("LunarLander-v2", {}, 4, 8, 1000, False)
    acrobot = EnvConfig("Acrobot-v1", {}, 3, 6, 200, False)
    pendulum = EnvConfig('Pendulum-v1', {}, 1, 3, 500, True)
    lunar_continuous = EnvConfig("LunarLander-v2", {"continuous": True},
                                  4, 8, 1000, True)


# TODO: what's the right design pattern for this?
# Idea: have this point to builder functions with variables supplied
# TODO: move agent name to enum
def build_agent(agent_name: str, action_dims: int, state_dims: int,
                min_reward: float, max_reward: float,
                action_bounds = None):
    if agent_name == "dqn":
        return build_dense_qagent(num_actions=action_dims,
                                        num_observations=state_dims,
                                        layer_sizes=[128, 64],
                                        drop_rate=0.05,
                                        gamma=0.99,
                                        tau=.05,
                                        train_epoch=1,
                                        batch_size=64,
                                        min_qerr=0.,
                                        max_qerr=(max_reward - min_reward)**2.,
                                        alpha=5.)
    elif agent_name == "dqn_cont":
        return build_dense_qagent_cont(action_bounds=action_bounds,
                                        num_observations=state_dims,
                                        layer_sizes=[128, 64],
                                        drop_rate=0.05,
                                        gamma=0.99,
                                        tau=.05,
                                        num_batch_sample=1,
                                        train_epoch=1,
                                        batch_size=64)
    elif agent_name == "dqn_distro":
        return build_dense_qagent_distro(num_actions=action_dims,
                                         num_observations=state_dims,
                                         num_atoms=51,
                                         Vmin=-1 * reward_scale,
                                         Vmax=reward_scale,
                                         layer_sizes=[128, 64],
                                         drop_rate=0.05,
                                         gamma=0.99,
                                         tau=.05,
                                         num_batch_sample=1,
                                         train_epoch=1,
                                         batch_size=64)


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
    env_config = Envs.cartpole.value
    min_reward = 0.
    max_reward = env_config.run_length
    # env_config = Envs.lunar.value
    # min_reward = -200.
    # max_reward = 200.
    # (env_config, def_params) = Envs.acrobot.value
    # (env_config, def_params) = Envs.pendulum.value
    # (env_config, def_params) = Envs.lunar_continuous.value

    agent = build_agent("dqn", env_config.num_actions, env_config.num_obs,
                        min_reward=min_reward, max_reward=max_reward)
    reward_seq = run_and_train(env_config, agent, num_runs=200,
                               seed_runs=5,
                               step_per_train=1,
                               step_per_copy=1,
                               debug_viz=False)
    print(reward_seq)
