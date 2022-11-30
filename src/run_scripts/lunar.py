"""Gymnasium[Box2D] lunar lander


    TODO: once we've done some experiments --> we can probably
        figure out how to make some runner functions that can be used
        across systems

"""
import gymnasium as gym
import numpy as np
import numpy.random as npr
from typing import Callable, Tuple, Dict
from random import randint


def runner(env: gym.Env, action_model: Callable,
           max_step: int = 200,
           init_action: int = 0):
    # state-action model
    # s_t --> model --> a_t --> env --> s_{t+1}, r_{t+1}
    #
    # action_model must keep track of (memory)
    #   1. previous observations, 2. previous actions
    # action model must take in env.step output
    action = init_action
    obs, actions, rewards, other = [], [], [], []
    for _ in range(max_step):
        # yields: s_{t+1}, r_{t+1}
        step_output = env.step(action)
        # yields: a_{t+1}
        action = action_model(step_output)
        obs.append(step_output[0])
        actions.append(action)
        rewards.append(step_output[1])
        other.append(step_output[2:])
    # return alignment: s_t, a_t, r_{t+1}
    # where: s_t -> a_t -> r_{t+1}
    return obs[:-1], actions[:-1], rewards[1:]


# epochs


def run_epoch(env: gym.Env, action_model: Callable,
              max_step: int,
              run_iters: int,
              p: float, rng: npr.Generator):
    # TODO: currently resetting struct every epoch

    # samping/processing wrapper for run output
    def pkg_and_sample(v):
        [obs, act, rew] = [np.array(vi) for vi in v]
        sel = rng.random(np.shape(obs)[0]) <= p
        return {"OBS": obs[sel], "ACT": act[sel], "REW": rew[sel]}

    env.reset(seed=42)
    struct = pkg_and_sample(runner(env, action_model, max_step))
    for _ in range(run_iters - 1):
        env.reset(seed=42)
        add_struct = pkg_and_sample(runner(env, action_model, max_step))
        struct = {k: np.concatenate([struct[k], add_struct[k]], axis=0) for k in struct}

    # TODO: training?
    # Design? just take in a training function/method...
    # BETTER: define an interface that supports
    #       1. select_action, 2. errors

    return struct


# Strats


def random_strat(step_output: Tuple):
    return randint(0, 3)


if __name__ == "__main__":
    # render modes:
    #   None(default): no render
    #   "human": continuously render in current display
    #   "rgb_array", "ansi", and a few others
    # lunar lander
    # env = gym.make("LunarLander-v2", render_mode="human")
    env = gym.make("LunarLander-v2")
    struct = run_epoch(env, random_strat, 100, 3, .05, npr.default_rng(42))
    env.close()

    for k in struct:
        print(k)
        print(np.shape(struct[k]))
