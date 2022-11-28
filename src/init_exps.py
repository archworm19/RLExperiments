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


def lunar_control_look(step_output: Tuple, rng: npr.Generator,
                       k3: float = 0.1):
    # TODO: add paramter for t --> how far ahead to predict

    # simple lunar model with look-ahead projection
    # Taylor Series approximation
    #   theta(t) = theta(t0) + k1 * t + k2/2 * t^2
    # Apply this approximation at each timestep + assume k2
    #       constant assumption = constant force from thrust
    # additional assumptions:
    #   1. work in t-steps = 1
    #   2. k1 given by model's dtheta
    # --> theta(t+1) = theta + dtheta + k3
    # where we'll infer k3
    x_targ = 0.
    y_targ = 5.

    obs = step_output[0]
    x = obs[0]
    y = obs[1]
    # stopgap
    if y > 1.4:
        return 0
    theta = obs[4]
    dtheta = obs[5]

    # vector comparison
    v_diff = np.array([x_targ - x, y_targ - y])
    theta_adj = theta + (np.pi / 2.)

    # Counterfactual
    # predict new theta as a function of different thrusts
    #   choose the option with the lower v_adj x magnitude
    actions = (1, 3)
    scores = []
    for k_force in (k3, -k3):
        theta_taylor = theta_adj + dtheta + k_force
        v_dir = np.array([np.cos(theta_taylor), np.sin(theta_taylor)])
        v_adj = v_diff - v_dir
        scores.append(v_adj[0])

    if rng.random() < 0.35:
        if np.fabs(scores[0]) < np.fabs(scores[1]):
            return actions[0]
        return actions[1]
    return 2


# Full On Inference system!
# improvement on above system:
# project x_new, y_new as well as theta_new
# > for each thrust possibility
# > > use taylor series to project new vector = v_new
# > take thrust option that minimizes f(v_new, v_target)
# ... need a good choice for f (probably just use a weighted average to start)


if __name__ == "__main__":
    # lunar lander
    env = gym.make("LunarLander-v2", render_mode="human")
    #env.reset(seed=42)
    #runner(env, partial(semi_random_policy, rng = npr.default_rng(42)))
    #env.reset(seed=42)
    #runner(env, partial(simple_control_lunar, rng = npr.default_rng(42)),
    #        max_step=500)
    for k in [.005, .01]:
        env.reset(seed=42)
        runner(env, partial(lunar_control_look, rng = npr.default_rng(42), k3=k),
                max_step=1000)
    env.close()
