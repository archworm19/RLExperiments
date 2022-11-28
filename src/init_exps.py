"""Temporary File for messing around with gymnasium
    Eventually: useful pieces should be factored apart"""
import numpy as np
import numpy.random as npr
import gymnasium as gym
from typing import Callable, Tuple, List
from functools import partial


def runner(env: gym.Env, action_model: Callable,
           max_step: int = 200,
           init_action: int = 0):
    # action_model must keep track of (memory)
    #   1. previous observations, 2. previous actions
    # action model must take in env.step output
    _observation, _info = env.reset(seed=42)
    action = init_action
    for _ in range(max_step):
        step_output = env.step(action)
        action = action_model(step_output)
    env.close()


# Policies


def semi_random_policy(step_output: Tuple, rng: npr.Generator):
    # this is a nonsense policy created for Lunar Lander
    obs = step_output[0]
    y = obs[1]
    if y < 1.2:
        # if drops too low --> fire main thruster
        return 2
    # else: randomly fire left/right thruster
    if rng.random() < 0.5:
        return 1
    return 3


class PID:
    # 1 variable pid control

    def __init__(self, target_val: float,
                 K_prop: float,
                 K_gral: float,
                 K_deriv: float,
                 dt: float = 1.):
        self.target_val = target_val
        self.K_prop = K_prop
        self.K_gral = K_gral
        self.K_deriv = K_deriv
        self.dt = dt
        self.prev_err = 0.
        self.running_integral = 0.

    def _update_terms(self, new_err: float):
        # proportional, integral, and derivative terms
        #   operates on error --> ASSUMES: error is updated
        # also outputs u(t)
        prop = new_err
        self.running_integral += new_err * self.dt
        deriv = (new_err - self.prev_err) * (1. / self.dt)
        u_t = (self.K_prop * prop + self.K_gral * self.running_integral
               + self.K_deriv * deriv)
        return prop, self.running_integral, deriv, u_t

    def control_step(self, y: float):
        # returns u(t)
        new_err = self.target_val - y
        _, _, _, ut = self._update_terms(new_err)
        self.prev_err = new_err
        return ut


class SimpleControlLunar():

    def __init__(self):
        # simple goal:
        #   keep y at 1.4
        #   keep theta at 0
        self.pid_y = PID(1.4, 0.3, 0.5, 0.0)
        self.pid_theta = PID(0., 0.1, 0.5, 0.0)
        self.rng = npr.default_rng(42)

    def __call__(self, step_output: Tuple):
        obs = step_output[0]
        # x = obs[0]
        y = obs[1]
        # dx = obs[2]
        # dy = obs[3]
        theta = obs[4]
        # dtheta = obs[5]
        u_y = self.pid_y.control_step(y)
        u_theta = self.pid_theta.control_step(theta)

        print(u_y)
        print(u_theta)

        # randomly select based on u magnitude
        p = np.fabs(u_y) / (np.fabs(u_y) + np.fabs(u_theta))

        print(p)
        input("cont?")

        if self.rng.random() < p:
            # main booster
            if u_y > 0.:
                return 2
            return 0
        else:
            # TODO: not sure about angles
            if u_theta < 0.:
                return 3
            return 1


if __name__ == "__main__":
    # lunar lander
    env = gym.make("LunarLander-v2", render_mode="human")
    # runner(env, partial(semi_random_policy, rng = npr.default_rng(42)))
    runner(env, SimpleControlLunar())
