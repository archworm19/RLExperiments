"""Proximal Policy Update Framework (TRPO, PPO)
    PPO paper: Proximal Policy Optimization Algorithms
        Schulman et al, 2017
"""
import tensorflow as tf
from typing import List
from frameworks.layer_signatures import ScalarStateModel

# TODO
# > clipped surrogate objective ~ eqn 7 DONE
#       no model;
#       just take in
#           r ~ probability ratio
#           A ~ advantage estimate
#           eta
# > sample average KL estimate (doesn't need own function)
#       between old policy and new
# > value target calculation; TODO: should we just use following formulation?
#       scipy.signal.lfilter
# > value target calculation + fixed T
#       use lfilter for T steps + value estimate for end
#       NOTE: this should probably just replace previous
# > generalized advantage calculation + fixed T
#       eqn 11
#       NOTE: definitely stop gradient through v (this is used for policy update)
# > higher order function ~ do all of the work given agents
# > TODO: parallelization system?
#       as far as I understand --> single valuation network + N parallel actors
#       ... could probably use 1 actor with stochasticity as well...


def clipped_surrogate_loss(probability_ratio: tf.Tensor,
                           advantage: tf.Tensor,
                           eta: float):
    """Clipped surrogate loss from PPO paper
        = E_t [ min(r_t(theta) A_t,
                    clip(r_t(theta), 1 - eta, 1 + eta) A_t) ]

    Args:
        probability_ratio (tf.Tensor): r_t(theta)
            = pi_{theta}(a_t | s_t) / pi_{theta_old}(a_t | s_t)
            shape = batch_size
        advantage (tf.Tensor): advantage estimate A_t
            shape = batch_size
        eta (float): step size

    Returns:
        tf.Tensor: surrogate loss for each sample
            shape = batch_size
    """
    rt_clip = tf.clip_by_value(probability_ratio, 1. - eta, 1. + eta)
    # --> batch_size x 2
    r2 = tf.stack([probability_ratio, rt_clip], axis=1)
    return tf.math.reduce_min(r2 * tf.expand_dims(advantage, 1), axis=1)


# TODO: calculate advantage separately?

# TODO: I'm missing some stuff
# 1. pi needs a distribution
#   > in the paper, actor model outputs --> mean of gaussian distro (with variable variance)
#   ez to sample from
#   calculate pi using the gaussian formulation ~ ratio of 2 gaussians?
# 2. transform sequences to samples?
#   A_t = series
#   if collect N sequences of length T for given episode segment
#   Oh! I think I get it now ~
#       run for T timesteps --> calc advantage / value estimate --> no train
#   ... after a certain number of collection runs --> optimize
#   ... hmmm... that doesn't seem to be the case for algorithm 1
#
#   2 corrected:
#       look back at advantage function + consider A_T
#       A_T = V(S_T)
#       --> so, we do calculate advantage in a CONVOLUTION
#       ... presumably value_target is calculated in a convolution as well
#
#
# TODO: how to do convolution?
#   input data = batch_size x T
#   > something like convolution with step function (0 to gamma * lambda)
#   one system: pad sequence right with 0s + use constant filter of gamma * lambda


# TODO: who uses this right_conv?
# > advantage
#       v = delta
#       k = gamma * lambda
#
# > value
#       v = concat(reward, V from final step)
#       k = gamma

def _right_conv(v: tf.Tensor,
                k: float):
    # v: shape = batch_size
    v_pad = tf.concat([v, 0. * v], axis=0)
    x = tf.concat([[1.],
                   tf.ones(tf.shape(v)[0], dtype=v_pad.dtype) * k],
                  axis=0)
    f = tf.math.cumprod(x, axis=0)[:-1]
    v_fin = tf.nn.conv1d(tf.reshape(v_pad, [1, -1, 1]),
                         tf.reshape(f, [-1, 1, 1]),
                         1,
                         "VALID",
                         data_format="NWC")
    return v_fin[0, :-1, 0]


def advantage_conv():
    pass

def ppo_actor_loss(pi: ScalarStateModel,
                   pi_old: ScalarStateModel,
                   value_model: ScalarStateModel,
                   states: List[tf.Tensor],
                   reward: tf.Tensor,
                   gamma: float,
                   lam: float):
    # TODO: this is totally wrong!
    """PPO actor loss
        L^{LKPEN}(theta) = eqn 8 from PPO paper


    Args:
        pi (ScalarStateModel): new model
        pi_old (ScalarStateModel): old actor model
        value_model (ScalarStateModel):
        states (List[tf.Tensor]):
            each tensor has shape batch_size x T x ...
        reward (tf.Tensor):
            shape = batch_size x T
        gamma (float): discount factor
        lam (float): discount factor for generalized advantage function
    """
    # t_i = tf.ones([T], dtype=reward.dtype)

    # unpack states
    states_t = [si[:-1] for si in states]
    states_t1 = [si[1:] for si in states]

    # advantage diffs
    #   = r_t + gamma * V(s_{t+1}) - V(s_t)
    adv_diffs = reward[:-1] + gamma * value_model(states_t1) - value_model(states_t)
    # generalized advantage
    #   = adv_diff(0) + (gamma * lam) * adv_diff(1) + ... + (gamma * lam)^(T - 1) * adv_diff(T)
    gen_adv = tf.cum_prod(t_i * gamma * lam)
    pass


if __name__ == "__main__":
    # where does gradient exist for surrogate loss?
    # should exist for all ones
    # case 1: prob ratio inside clip boundary --> gradient = advantage
    # case 2: prob ratio outside boundary + performance is better --> gradient = 0
    # case 3: prob ratio outside boundary + performance is worse --> gradient = advantage
    # case 4: prob ratio outside boundary + performance is worse --> gradient = advantage (neg version)
    xs = [tf.ones([5]), tf.ones([5]) * 10, tf.ones([5]) * -10, tf.ones([5]) * 10]
    Ats = [tf.ones([5]) * 2., tf.ones([5]) * 2., tf.ones([5]) * 2., tf.ones([5]) * -2.]
    exp_g = [tf.ones([5]) * 2., tf.ones([5]) * 0., tf.ones([5]) * 2., tf.ones([5]) * -2.]
    for x, At, eg in zip(xs, Ats, exp_g):
        with tf.GradientTape() as g:
            g.watch(x)
            y = clipped_surrogate_loss(x, At, 0.2)
        assert tf.math.reduce_all(g.gradient(y, x) == eg)
