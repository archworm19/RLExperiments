"""Gymnasium[Box2D] lunar lander


    TODO: once we've done some experiments --> we can probably
        figure out how to make some runner functions that can be used
        across systems

"""
import gymnasium as gym
from typing import Callable, Tuple, List


def runner(env: gym.Env, action_model: Callable,
           max_step: int = 200,
           init_action: int = 0):
    # action_model must keep track of (memory)
    #   1. previous observations, 2. previous actions
    # action model must take in env.step output
    action = init_action
    for _ in range(max_step):
        step_output = env.step(action)
        action = action_model(step_output)


if __name__ == "__main__":
    # render modes:
    #   None(default): no render
    #   "human": continuously render in current display
    #   "rgb_array", "ansi", and a few others
    # lunar lander
    env = gym.make("LunarLander-v2", render_mode="human")
    env.reset(seed=42)
    env.close()