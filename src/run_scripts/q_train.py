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
from run_scripts.utils import build_dense_qagent, build_dense_qagent_cont


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
                DefaultParams(0.99, 1.,
                              8, 128,
                              2,
                              10, 10))
    lunar = (EnvConfig("LunarLander-v2", 4, 8),
             DefaultParams(0.99, 1.,
                           16, 128,
                           3,
                           10, 10))
    acrobot = (EnvConfig("Acrobot-v1", 3, 6),
               DefaultParams(0.99, 1.,
                              8, 128,
                              2,
                              10, 10))


def run_and_train(env_config: EnvConfig,
                  agent: Agent,  # TODO: need more specific interface for discrete/update agent
                  run_length: int = 1000,
                  num_runs: int = 300,
                  show_progress: bool = True,
                  seed_runs: int = 20,
                  step_per_train: int = 1,
                  step_per_copy: int = 1,
                  runs_per_display: int = 5,
                  debug_viz: bool = False):
    # NOTE: should work in discrete or continious case
    # run params
    run_length = run_length
    # number of seeds per 
    num_runs = num_runs

    # render modes:
    #   None(default): no render
    #   "human": continuously render in current display
    #   "rgb_array", "ansi", and a few others
    # TODO: need better design for specifying continuous
    env_run = gym.make(env_config.env_name)
    if show_progress:
        env_disp = gym.make(env_config.env_name, render_mode="human")
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

    """
    # cartpole
    (env_config, def_params) = Envs.cartpole.value
    run_length = 1000
    # lunar lander
    # (env_config, def_params) = Envs.lunar.value
    # run_length = 1000
    # (env_config, def_params) = Envs.acrobot.value
    # run_length = 500

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
                               run_length=run_length, num_runs=200,
                               seed_runs=5,
                               step_per_train=def_params.step_per_train,
                               step_per_copy=def_params.step_per_copy)
    print(reward_seq)
    """

    # continuous testing (with pendulum)
    # pendulum works pretty well tho there's still some instability
    env_config = EnvConfig('Pendulum-v1', 1, 3)
    action_bounds = [(-2., 2.)]
    num_observations=3
    run_length = 500
    agent = build_dense_qagent_cont(action_bounds=action_bounds,
                                    num_observations=num_observations,
                                    tau=0.1, num_batch_sample=1,
                                    train_epoch=1,
                                    gamma=0.99,
                                    batch_size=64,
                                    drop_rate=.05)
    reward_seq = run_and_train(env_config, agent,
                               run_length=run_length, num_runs=200,
                               seed_runs=5,
                               step_per_train=1,
                               step_per_copy=1,
                               debug_viz=False)
    print(reward_seq)

    """
    # continuous testing
    # TODO: yet another design problem
    env_config = EnvConfig("LunarLander-v2", 2, 8)
    run_length=1000
    # from paper: tau = .001 (I believe)
    agent = build_dense_qagent_cont(tau=0.1, num_batch_sample=1,
                                    train_epoch=1,
                                    gamma=0.99,
                                    batch_size=64,
                                    drop_rate=.05)
    reward_seq = run_and_train(env_config, agent,
                               run_length=run_length, num_runs=200,
                               seed_runs=5,
                               step_per_train=1,
                               step_per_copy=1,
                               debug_viz=False)
    print(reward_seq)
    """
