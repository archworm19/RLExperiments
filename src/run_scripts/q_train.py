"""
    Deep Q Learning + Open AI Gym

"""
import gymnasium as gym
import numpy as np
import numpy.random as npr
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Pool
from functools import partial
from frameworks.agent import RunData, Agent

from run_scripts.runner import runner
from run_scripts.utils import build_dense_qagent, purge_run_data


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
    run_per_train: int   # runs per training session
    run_per_copy: int   # runs per copy session
    train_epoch: int


class Envs(Enum):
    cartpole = (EnvConfig("CartPole-v1", 2, 4),
                DefaultParams(0.87, 0.25, 8,
                              1, 3, 1))
    lunar = (EnvConfig("LunarLander-v2", 4, 8),
             DefaultParams(0.99, 1., 24,
                           1, 1, 2))


def _generate_seeds(num_seeds: int):
    v = npr.randint(0, 1000, num_seeds)
    return [int(vi) for vi in v]


def _append_rundat(struct: RunData, add_struct: RunData):
    return RunData(np.concatenate([struct.states, add_struct.states], axis=0),
                   np.concatenate([struct.states_t1, add_struct.states_t1], axis=0),
                   np.concatenate([struct.actions, add_struct.actions], axis=0),
                   np.concatenate([struct.rewards, add_struct.rewards], axis=0),
                   np.concatenate([struct.termination, add_struct.termination], axis=0))


def run_and_train(env_config: EnvConfig,
                  def_params: DefaultParams,
                  agent: Agent,  # TODO: need more specific interface for discrete/update agent
                  run_length: int = 200,
                  num_runs: int = 300,
                  show_progress: bool = True):
    # run params
    # decay_rate^num_runs = .01
    rap_decay_rate = .01 ** (1. / num_runs)
    run_length = run_length
    # number of seeds per 
    num_runs = num_runs
    run_per_train = def_params.run_per_train
    run_per_copy = def_params.run_per_copy

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

    struct = None
    rews = []

    # different seed for each run:
    seeds = _generate_seeds(num_runs)
    for i, seed in enumerate(seeds):
        if i % 20 == 0:  # use display
            active_env = env_disp
            debug_set = False
        else:
            active_env = env_run
            debug_set = False

        active_env.reset(seed=seed)
        # gather run data
        add_struct = runner(active_env, agent, run_length,
                            debug=debug_set)
        rews.append(np.sum(add_struct.rewards))

        # append
        if struct is None:
            struct = add_struct
        else:
            struct = _append_rundat(struct, add_struct)

        # train
        if i % run_per_train == 0:
            agent.train(struct, debug=False)

        # copy
        if i % run_per_copy == 0:
            agent._copy_model(debug=False)

        # purge struct:
        struct = purge_run_data(struct, 500000)

        # decay:
        agent.run_iface.rand_act_prob *= rap_decay_rate

    env_run.close()
    env_disp.close()
    return rews


def build_and_run(params: DefaultParams,
                env_config: EnvConfig,
                run_length: int,
                num_runs: int):
    agent = build_dense_qagent(num_actions=env_config.num_actions,
                            num_observations=env_config.num_obs,
                            layer_sizes=[64, 32],
                            drop_rate=0.05,
                            gamma=params.gamma,
                            tau=params.tau,
                            num_batch_sample=params.num_batch_sample,
                            train_epoch=params.train_epoch)
    rews = run_and_train(env_config, params, agent,
                    run_length=run_length, num_runs=num_runs,
                    show_progress=False)
    return np.mean(rews)


def param_search_random(env_config: EnvConfig,
                        params_lb: DefaultParams,
                        params_ub: DefaultParams,
                        num_rand: int = 40,
                        run_length: int = 300,
                        num_runs: int = 300):
    def sel_param(v_lo, v_hi):
        return v_lo + (v_hi - v_lo) * npr.random()

    # TODO: this will not generalize!
    def gen_params(p_lo: DefaultParams, p_hi: DefaultParams):
        return DefaultParams(sel_param(p_lo.gamma, p_hi.gamma),
                             sel_param(p_lo.tau, p_hi.tau),
                             int(sel_param(p_lo.num_batch_sample, p_hi.num_batch_sample)),
                             int(sel_param(p_lo.run_per_train, p_hi.run_per_train)),
                             int(sel_param(p_lo.run_per_copy, p_hi.run_per_copy)),
                             int(sel_param(p_lo.train_epoch, p_hi.train_epoch)))
    # generate params
    params = [gen_params(params_lb, params_ub) for _ in range(num_rand)]

    with Pool(processes=4) as p:
        rew_sums = p.map(partial(build_and_run, env_config=env_config,
                                                run_length=run_length,
                                                num_runs=num_runs),
                         params)
    return params, rew_sums


if __name__ == "__main__":

    # cartpole
    (env_config, def_params) = Envs.cartpole.value
    # lunar lander
    (env_config, def_params) = Envs.lunar.value


    # run
    agent = build_dense_qagent(num_actions=env_config.num_actions,
                               num_observations=env_config.num_obs,
                               layer_sizes=[128, 64],
                               drop_rate=0.05,
                               gamma=def_params.gamma,
                               tau=def_params.tau,
                               num_batch_sample=def_params.num_batch_sample,
                               train_epoch=def_params.train_epoch)
    reward_seq = run_and_train(env_config, def_params, agent,
                               run_length=500, num_runs=400)
    print(reward_seq)


    """
    # optimize
    params_lb = DefaultParams(0.95, .1, 8, 1, 2, 1)
    params_ub = DefaultParams(0.99, 0.25, 12, 3, 3, 3)
    params, rews = param_search_random(env_config,
                                        params_lb, params_ub,
                                        num_rand=16,
                                        num_runs=250,
                                        run_length=500)
    max_ind = np.argmax(rews)
    print(np.amax(rews))
    print(params[max_ind])
    """

