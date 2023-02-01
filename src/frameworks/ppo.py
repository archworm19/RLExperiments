"""Proximal Policy Update Framework (TRPO, PPO)
    PPO paper: Proximal Policy Optimization Algorithms
        Schulman et al, 2017
"""
import tensorflow as tf
import numpy as np
from typing import List, Dict
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


def _right_conv(v: np.ndarray,
                k: float):
    # v: shape = T
    # return shape = T
    v_pad = np.concatenate((v, 0. * v), axis=0)
    f = k ** np.arange(np.shape(v)[0])
    return np.convolve(v_pad, f[::-1], mode="valid")[:-1]


def advantage_conv(V: np.ndarray,
                   reward: np.ndarray,
                   gamma: float,
                   lam: float,
                   terminated: bool = False):
    """Generalized advantage calculation (convolution)
        eqn 11 in PPO paper

    Args:
        V (np.ndarray): value_model(state) --> V
            shape = T
        reward (np.ndarray):
            shape = T
        gamma (float): discount factor
        lam (float): generalized discount factor
        terminated (bool): set to true if sequence
            terminates (assumed to terminate at end of seq)
            V(s_{T+1}) = 0

    Returns:
        np.ndarray: generalized advantage series
            shape = (T - 1)
    """
    delta = reward[:-1] + gamma * V[1:] - V[:-1]
    if terminated:
        delta[-1] = reward[-2] - V[-2]
    return _right_conv(delta, gamma * lam)


def value_conv(V: np.ndarray,
               reward: np.ndarray,
               gamma: float,
               terminated: bool = False):
    """target Value calculation (convolution)
        = r_t + gamma * r_{t+1} + gamma^2 * r_{t+1} + ...
        later timesteps will involve fewer reward values
    
        sequence assumption:
            s_t + a_t --> r_t, s_{t+1}
            so, since we want to use V(s_T) as proxy for
                rewards following reward ndarray -->
            replace last reward with V[-1]
                or 0 if terminated sequence

    Args:
        V (np.ndarray): value_model(state) --> V
            shape = T
        reward (np.ndarray):
            shape = T
        gamma (float): discount factor
        terminated (bool): set to true if sequence
            terminates (assumed to terminate at end of seq)
            V(s_{T+1}) = 0

    Returns:
        np.ndarray: value estimate
            shape = T
    """
    if terminated:
        rv = np.concatenate((reward[:-1], [0.]), axis=0)
    else:
        rv = np.concatenate((reward[:-1], V[-1:]), axis=0)
    return _right_conv(rv, gamma)


# TODO: there are separate datasets!
# analysis of gradients
# > L_VF only involves V(s_t) = critic model
# > L_CLIP involves critic model V(s_t) and
#       actor model pi
#       ... tho, I suppose you could freeze critic for this
#       YES!!! V(s_t) should be frozen for this error
#       ... should probably be handled through stop_gradient... might still want single dataset!
#
# background seq: s_t + a_t --> r_t, s_{t+1}
#
# Option 1: single dataset ~ yeah, this is the way
# > lop off extra value (that's fine; this is just V(s) / no reward anyway)
# > > concat all sequences
# > > shuffle
# > > batch
# > each sample = s_t, value_t, advantage_t
# > train on summed losses


def package_dataset(states: Dict[str, List[np.ndarray]],
                    V: List[np.ndarray],
                    reward: List[np.ndarray],
                    terminated: List[bool],
                    gamma: float,
                    lam: float,
                    adv_name: str = "adv",
                    val_name: str = "val"):
    """package dataset for ppo training actor and/or critic

    Args:
        states (Dict[str, List[np.ndarray]]): mapping from state names to
            state vectors. Each dict entry is a different state.
            Each list is a different trajectory.
            states[k0][i] matches up with states[k1][i]
        V (List[np.ndarray]): V(s_t) = critic evaluation of states
            Each list is a different trajectory.
            Each ndarray has shape T x ...
        reward (List[np.ndarray]):
            Each list is a different trajectory.
            Each ndarray has shape T x ...
        terminated (List[bool]): whether each trajectory
            was terminated or is still running
        gamma (float): discount factor
        lam (float): generalized advantage factor
        adv_name (str, optional): advantage field name. Defaults to "adv".
        val_name (str, optional): value field name. Defaults to "val".
            value = estimated reward value = r_t + gamma * r_t1 + ...

    Returns:
        tf.data.Dataset: shuffled dataset with named fields
            fields:
                > one for each state
                > advantage (adv_name)
                > value (val_name)
    """
    advs = [advantage_conv(vi, ri, gamma, lam, termi) for vi, ri, termi in
            zip(V, reward, terminated)]
    # NOTE: last value doesn't involve reward --> no training signal
    vals = [value_conv(vi, ri, gamma, termi)[:-1] for vi, ri, termi in
            zip(V, reward, terminated)]

    # package into dataset
    d = {k: np.concatenate([ski[:-1] for ski in states[k]], axis=0)
         for k in states}
    d[adv_name] = np.concatenate(advs, axis=0)
    d[val_name] = np.concatenate(vals, axis=0)
    dset = tf.data.Dataset.from_tensor_slices(d)
    return dset.shuffle(np.shape(d[adv_name])[0])


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
    V = np.ones((20,))
    reward = np.ones((20,))
    gamma = 0.9
    At = advantage_conv(V, reward, gamma, 1.)

    for i in range(19):
        a_i = -1. * V[i]
        gfactor = 1.
        for j in range(i, 20):
            a_i += reward[j] * gfactor
            gfactor *= gamma
        assert np.round(a_i, 4) == np.round(At[i], 4)

    # value estimate?
    val = value_conv(V, reward, gamma)
    for i in range(20):
        v_i = 0.
        gfactor = 1.
        for j in range(i, 19):
            v_i += reward[j] * gfactor
            gfactor *= gamma
        v_i += V[-1] * gfactor
        assert np.round(v_i, 4) == np.round(val[i], 4)

    # TODO: termination tests

    # package dset:
    # test with 2 sequences of different lengths
    s1 = np.zeros((10, 2))
    s2 = np.zeros((5, 2))
    v1 = np.ones((10,))
    v2 = np.ones((5,))
    r1 = np.ones((10,))
    r2 = np.ones((5,))
    terminated = [False, True]
    dset = package_dataset({"s": [s1, s2]}, [v1, v2], [r1, r2], terminated, 0.9, 1.)
    for v in dset.batch(4):
        print(v)


