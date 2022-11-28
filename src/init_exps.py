"""Temporary File for messing around with gymnasium
    Eventually: useful pieces should be factored apart"""
from random import seed
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
    action = init_action
    for _ in range(max_step):
        step_output = env.step(action)
        action = action_model(step_output)


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


def simple_control_lunar(step_output: Tuple, rng: npr.Generator):
    # problem formulation?
    # target: x_targ, y_targ, sigma_targ
    #       = (0, 1.4, 0)
    # there are 2 vectors of interest for this formulation
    #   1. direction vector for rocket: v_dir
    #   2. vector from rocket x,y to correct x,y: v_diff
    #
    # top-level goal: align the 2 vectors
    #       minimize angle between them
    #
    # v_diff = [dx, dy] = [x_targ - x, y_targ - y] (x, y)
    # v_dir = [1, theta] (assumes r = 1 for direction vector) (polar)
    #       = [cos(theta), sin(theta)]  (x, y)

    # NOTE on theta: theta = 0 when rocket is upright
    #   --> adjust theta (theta_adj) to make math work

    # NOTE: super lazy stop gap
    #   > set y_targ very high
    #   > turn off booster if rocket gets out of screen
    #   == should prevent flipping as rocket will not get above y_targ

    x_targ = 0.
    y_targ = 5.

    obs = step_output[0]
    x = obs[0]
    y = obs[1]

    # stopgap
    if y > 1.5:
        return 0

    # dx = obs[2]
    # dy = obs[3]
    theta = obs[4]
    v_diff = np.array([x_targ - x, y_targ - y])
    theta_adj = theta + (np.pi / 2.)
    v_dir = np.array([np.cos(theta_adj), np.sin(theta_adj)])
    v_adj = v_diff - v_dir

    # sampling: up vs. side thrust
    up_mag = 1.5 - y
    theta_mag = np.fabs(v_adj[0])
    theta_prob = theta_mag / (theta_mag + up_mag)

    if rng.random() < theta_prob:
        # left booster:
        if v_adj[0] > 0:
            return 3
        return 1
    return 2


def _simple_lunar_project(x, y, dx, dy, theta, dtheta,
                          k_main, k_theta_right, k_theta_left,
                          dt = 0.1):
    # simplified x, y model = only main thruster yields x, y change
    #       at current step
    #   x(t + t0) = x(t0) + dx/dt * dt
    #   dx/dt(t + t0) = dx/dt(t0) + k_main * cos(theta) * dt
    #   y(t + t0) = y(t0) + dy/dt * dt
    #   dy/dt(t + t0) = dy/dt(t0) + k_main * sin(theta) * dt
    #   theta(t + t0) = theta(t0) + dtheta/dt * dt
    #   dtheta/dt(t + t0) = dtheta/dt + [k_theta_right - k_theta_left] * dt
    num_step = 1. / dt
    for _ in range(int(num_step)):
        x += dx * dt
        dx += k_main * np.cos(theta) * dt
        y += dy * dt
        dy += k_main * np.sin(theta) * dt
        theta += dtheta * dt
        dtheta += (k_theta_right - k_theta_left) * dt
    return x, y, dx, dy, theta, dtheta
   

def lunar_control_infer(step_output: Tuple, rng: npr.Generator,
                        k_r: float = 0.1,
                        k_theta: float = 0.1):
    # lunar control inference model
    # forward projection taylor series
    #   theta(t) = theta(t0) + dtheta/dt * t + 1/2 * dtheta^2 / dt^2 * t^2
    #           theta(t0) = theta measured
    #           dtheta/dt = dtheta measured
    #           0.5 * dtheta^2/dt^2 = k_theta * [right booster] - k_theta * [left booster]
    # simplified x, y model
    #   only main thruster yields x, y changes
    #   main thruster force = magnitude and direction = theta -->
    #   x(t) = x(t0) + dx/dt * dt + F/2 cos(theta) * dt^2
    #   y(t) = y(t0) + dy/dt * dt + F/2 sin(theta) * dt^2

    # TODO: make these params:
    x_target = 0
    y_target = 1.25
    # Assuming standard unit circle
    theta_target = np.pi / 2.
    v_target = np.array([x_target, y_target, theta_target])

    # unpack
    obs = step_output[0]
    [x, y, dx, dy, theta, dtheta, _, _] = obs

    # adjust theta to standard unit circle
    theta += np.pi / 2.

    # factor for adjusting dx and dy to dx/dt and dy/dt
    # NOTE: these are just approximations
    dx = dx * 0.01
    dy = dy * 0.01
    dtheta = dtheta * 0.01

    # action -> force mapping
    #   action enum --> k_r, k_theta_rightboost, k_theta_left_boost
    action_map = [[0, 0, 0],  # no boost
                  [0, k_theta, 0],  # right boost
                  [k_r, 0, 0],  # main boost
                  [0, 0, k_theta]]  # left boost

    # project each possible action:
    errs = []
    for forces in action_map:
        [_k_r, _k_theta_right, _k_theta_left] = forces
        xp, yp, dxp, dyp, thetap, dthetap = _simple_lunar_project(x, y, dx, dy,
                                                                  theta, dtheta,
                                                                  _k_r, _k_theta_right, _k_theta_left)
        # TODO: more work needed:
        errs.append(np.sum(np.fabs(v_target - np.array([xp, yp, thetap]))))
    return np.argmin(errs)


if __name__ == "__main__":
    # lunar lander
    env = gym.make("LunarLander-v2", render_mode="human")
    #env.reset(seed=42)
    #runner(env, partial(semi_random_policy, rng = npr.default_rng(42)))
    #env.reset(seed=42)
    #runner(env, partial(simple_control_lunar, rng = npr.default_rng(42)),
    #        max_step=500)
    env.reset(seed=42)
    runner(env, partial(lunar_control_infer, rng=npr.default_rng(42),
                        k_r = .005, k_theta = .005),
           max_step=500)
    env.close()
