"""
    Deep Q Learning + Open AI Gym

"""
import gymnasium as gym
import numpy as np
import numpy.random as npr
from dataclasses import dataclass
from enum import Enum
from frameworks.agent import Agent

from run_scripts.runner import runner
from run_scripts.utils import build_dense_qagent


@dataclass
class EnvConfig:
    env_name: str
    num_actions: int
    num_obs: int


@dataclass
class DefaultParams:
    gamma: float  # discount factor
    tau: float  # weight for model copying
    num_batch_sample: int  # number of batches pulled for training
    batch_size: int
    train_epoch: int
    step_per_train: int
    step_per_copy: int


# NOTE: tau = 1 --> regular DQN


class Envs(Enum):
    cartpole = (EnvConfig("CartPole-v1", 2, 4),
                DefaultParams(0.87, 1.,
                              1, 32,
                              1,
                              5, 5))
    lunar = (EnvConfig("LunarLander-v2", 4, 1),
             DefaultParams(0.99, 1.,
                           1, 32,
                           1,
                           5, 5))


def run_and_train(env_config: EnvConfig,
                  agent: Agent,  # TODO: need more specific interface for discrete/update agent
                  run_length: int = 1000,
                  num_runs: int = 300,
                  show_progress: bool = True,
                  seed_runs: int = 20,
                  step_per_train: int = 1,
                  step_per_copy: int = 1):
    # run params
    # decay_rate^num_runs = .01
    rap_decay_rate = .01 ** (1. / num_runs)
    run_length = run_length
    # number of seeds per 
    num_runs = num_runs

    agent.run_iface.rand_act_prob = 1.

    # render modes:
    #   None(default): no render
    #   "human": continuously render in current display
    #   "rgb_array", "ansi", and a few others
    env_run = gym.make(env_config.env_name)
    if show_progress:
        env_disp = gym.make(env_config.env_name, render_mode="human")
    else:
        env_disp = env_run

    rews = []
    # different seed for each run:
    for i in range(num_runs):
        if i % 20 == 0:  # use display
            active_env = env_disp
            train_mode = False
            debug_set = False
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

        # decay:
        agent.run_iface.rand_act_prob *= rap_decay_rate

    env_run.close()
    env_disp.close()
    return rews


if __name__ == "__main__":

    # cartpole
    (env_config, def_params) = Envs.cartpole.value
    # lunar lander
    # (env_config, def_params) = Envs.lunar.value

    # run
    agent = build_dense_qagent(num_actions=env_config.num_actions,
                               num_observations=env_config.num_obs,
                               layer_sizes=[128, 64],
                               drop_rate=0.05,
                               gamma=def_params.gamma,
                               tau=def_params.tau,
                               num_batch_sample=def_params.num_batch_sample,
                               train_epoch=def_params.train_epoch,
                               batch_size=def_params.batch_size)
    reward_seq = run_and_train(env_config, agent,
                               run_length=1000, num_runs=200,
                               seed_runs=20,
                               step_per_train=def_params.step_per_train,
                               step_per_copy=def_params.step_per_copy)
    print(reward_seq)
