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
    # v: shape = batch_size x T
    # return shape = batch_size x T
    v_pad = tf.concat([v, 0. * v], axis=1)
    f = tf.math.pow(k, tf.cast(tf.range(tf.shape(v)[1]), v.dtype))
    v_fin = tf.nn.conv1d(tf.expand_dims(v_pad, 2),
                         tf.reshape(f, [-1, 1, 1]),
                         1,
                         "VALID",
                         data_format="NWC")
    return v_fin[:, :-1, 0]


def advantage_conv(V: tf.Tensor,
                   reward: tf.Tensor,
                   gamma: float,
                   lam: float):
    """Generalized advantage calculation (convolution)
        eqn 11 in PPO paper

    Args:
        V (tf.Tensor): value_model(state) --> V
            shape = batch_size x T
        reward (tf.Tensor):
            shape = batch_size x T
        gamma (float): discount factor
        lam (float): generalized discount factor

    Returns:
        tf.Tensor: generalized advantage series
            shape = batch_size x (T - 1)
    """
    delta = reward[:, :-1] + gamma * V[:, 1:] - V[:, :-1]
    return _right_conv(delta, gamma * lam)


def value_conv(V: tf.Tensor,
               reward: tf.Tensor,
               gamma: float):
    """target Value calculation (convolution)
        = r_t + gamma * r_{t+1} + ... + gamma^T * V(T)

    Args:
        V (tf.Tensor): value_model(state) --> V
            shape = batch_size x T
        reward (tf.Tensor):
            shape = batch_size x T
        gamma (float): discount factor

    Returns:
        tf.Tensor: value estimate
            shape = batch_size x (T + 1)
    """
    # --> batch_size x T + 1
    rv = tf.concat([reward, V[:, -1:]], axis=1)
    return _right_conv(rv, gamma)


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

    # advantage calculation testing
    # if lam = 1 --> should become reward sum
    V = tf.ones([3, 20], dtype=tf.float32)
    reward = tf.ones([3, 20], dtype=tf.float32)
    gamma = 0.9
    At = advantage_conv(V, reward, gamma, 1.)

    for i in range(19):
        a_i = -1. * V[0, i].numpy()
        gfactor = 1.
        for j in range(i, 20):
            a_i += reward[0, j].numpy() * gfactor
            gfactor *= gamma
        assert tf.math.reduce_all(tf.round(At[:, i] * 100) ==
                                  tf.round(tf.constant(a_i, tf.float32) * 100))

    # value estimate?
    val = value_conv(V, reward, gamma)
    for i in range(21):
        v_i = 0.
        gfactor = 1.
        for j in range(i, 20):
            v_i += reward[0, j].numpy() * gfactor
            gfactor *= gamma
        v_i += V[0, -1].numpy() * gfactor
        assert tf.math.reduce_all(tf.round(val[:, i] * 100) ==
                                  tf.round(tf.constant(v_i, tf.float32) * 100))
