"""Proximal Policy Update Framework (TRPO, PPO)
    PPO paper: Proximal Policy Optimization Algorithms
        Schulman et al, 2017
"""
import tensorflow as tf
import numpy as np
from typing import List, Dict


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
            shape = T - 1
        gamma (float): discount factor
        lam (float): generalized discount factor
        terminated (bool): set to true if sequence
            terminates (assumed to terminate at end of seq)
            V(s_{T+1}) = 0

    Returns:
        np.ndarray: generalized advantage series
            shape = (T - 1)
    """
    delta = reward + gamma * V[1:] - V[:-1]
    if terminated:
        delta[-1] = reward[-1] - V[-2]
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
            shape = T - 1
        gamma (float): discount factor
        terminated (bool): set to true if sequence
            terminates (assumed to terminate at end of seq)
            V(s_{T+1}) = 0

    Returns:
        np.ndarray: value estimate
            shape = T
    """
    if terminated:
        rv = np.concatenate((reward, [0.]), axis=0)
    else:
        rv = np.concatenate((reward, V[-1:]), axis=0)
    return _right_conv(rv, gamma)


def package_dataset(states: List[Dict[str, np.ndarray]],
                    V: List[np.ndarray],
                    reward: List[np.ndarray],
                    actions: List[np.ndarray],
                    terminated: List[bool],
                    gamma: float,
                    lam: float,
                    adv_name: str = "adv",
                    val_name: str = "val",
                    action_name: str = "action"):
    """package dataset for ppo training actor and/or critic

        Assumed ordering: s_t + a_t --> r_t, s_{t+1}
        so, 0th element of each array = function of state s0

    Args:
        states (List[Dict[str, np.ndarray]]):
            outer list = different trajectories
            inner dict = mapping from state names to state values
            Each ndarray has shape (T + 1) x ...
        V (List[np.ndarray]): V(s_t) = critic evaluation of states
            Each list is a different trajectory.
            Each ndarray has shape (T + 1) x ...
        reward (List[np.ndarray]):
            Each list is a different trajectory.
            Each ndarray has shape T x ...
        actions (List[np.ndarray]): where len of each state
            trajectory is T --> len of reward/action trajectory = T-1
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
                > actions (action_name)
    """
    advs = [advantage_conv(vi, ri, gamma, lam, termi) for vi, ri, termi in
            zip(V, reward, terminated)]
    # NOTE: last value term = V(s_T) ~ no training signal there --> lop it off
    vals = [value_conv(vi, ri, gamma, termi)[:-1] for vi, ri, termi in
            zip(V, reward, terminated)]

    # package into dataset
    d = {}
    for k in states[0]:
        d[k] = np.concatenate([st[k][:-1] for st in states])
    d[action_name] = np.concatenate(actions, axis=0)
    d[adv_name] = np.concatenate(advs, axis=0)
    d[val_name] = np.concatenate(vals, axis=0)
    dset = tf.data.Dataset.from_tensor_slices(d)
    return dset.shuffle(np.shape(d[adv_name])[0])


def ppo_loss_multiclass(pi_old_distro: tf.Tensor, pi_new_distro: tf.Tensor,
                        critic_pred: tf.Tensor,
                        action: tf.Tensor,
                        advantage: tf.Tensor,
                        value_target: tf.Tensor,
                        eta: float,
                        vf_scale: float = 1.,
                        entropy_scale: float = 0.):
    """ppo loss for multiclass distribution actors
        = L^CLIP + L^VF from ppo paper for actor error
            as well as critic error (V(s_t) - V_targ)^2

        NOTE: this loss trains actor and critic simultaneously
        NOTE: using fixed advantage --> actor is trained on
            old value model

    Args:
        pi_old (tf.Tensor): 
        pi_new (tf.Tensor):
            old and new probability distros (softmaxed)
            output by actor models
            shape = batch_size x num_actions
        critic_pred (tf.Tensor):
            critic prediction ~ should approximate value calc
            shape = batch_size
        action (tf.Tensor): one-hot actions
            shape = batch_size x num_actions
        advantage (tf.Tensor): estimated advantage
            shape = batch_size
        value_target (tf.Tensor): critic target
            shape = batch_size
        eta (float): allowable step size; used by clipped surrogate
        vf_scale (float): scale on L^VF term
            c1 from 
        entropy_scale (float): c2 from ppo paper

    Returns:
        tf.Tensor: loss for each sample
            shape = batch_size
    """
    prob_old = tf.stop_gradient(tf.math.reduce_sum(pi_old_distro * action, axis=1))
    prob_new = tf.math.reduce_sum(pi_new_distro * action, axis=1)
    prob_ratio = tf.math.divide(prob_new, prob_old)
    l_clip = clipped_surrogate_loss(prob_ratio, advantage, eta)
    l_vf = tf.math.pow(critic_pred - value_target, 2.)
    negentropy = tf.math.reduce_sum(pi_new_distro * tf.math.log(pi_new_distro), axis=1)
    return l_vf - vf_scale * l_clip + entropy_scale * negentropy


def _gauss_prob_ratio(x: tf.Tensor,
                      mu_num: tf.Tensor, prec_num: tf.Tensor,
                      mu_denom: tf.Tensor, prec_denom: tf.Tensor):
    # num = numerator; denom = denominator; prec=precision
    # all shapes assumed to be batch_size x action_dims
    # det(var) for diagonal = product of diagonals
    pre_term = tf.math.sqrt(tf.math.reduce_prod(tf.math.divide(prec_num, prec_denom), axis=1))
    # inside the exponent
    diff_num = x - mu_num
    v_num = tf.math.reduce_sum(diff_num * prec_num * diff_num, axis=1)
    diff_denom = x - mu_denom
    v_denom = tf.math.reduce_sum(diff_denom * prec_denom * diff_denom, axis=1)
    exp_term = tf.math.exp(-0.5 * (v_num - v_denom))
    return pre_term * exp_term


def ppo_loss_gauss(pi_old_mu: tf.Tensor, pi_old_precision: tf.Tensor,
                   pi_new_mu: tf.Tensor, pi_new_precision: tf.Tensor,
                   critic_pred: tf.Tensor,
                   action: tf.Tensor,
                   advantage: tf.Tensor,
                   value_target: tf.Tensor,
                   eta: float,
                   vf_scale: float = 1.,
                   entropy_scale: float = 0.):
    """ppo loss for gaussian distribution actors
        = L^CLIP - L^VF + entropy from ppo paper for actor error
            as well as critic error (V(s_t) - V_targ)^2

        NOTE: this loss trains actor and critic simultaneously
        NOTE: using fixed advantage --> actor is trained on
            old value model

    Args:
        pi_old_mu (tf.Tensor): mean outputs by old actor
            shape = batch_size x action_dims
        pi_old_precision (tf.Tensor): 1/variance output by old actor
            shape = batch_size x action_dims = diagonal covar
        pi_new_mu (tf.Tensor):
        pi_new_precision (tf.Tensor):
        critic_pred (tf.Tensor):
            critic prediction ~ should approximate value calc
            shape = batch_size
        action (tf.Tensor): one-hot actions
            shape = batch_size x num_actions
        advantage (tf.Tensor): estimated advantage
            shape = batch_size
        value_target (tf.Tensor): critic target
            shape = batch_size
        eta (float): allowable step size; used by clipped surrogate
        vf_scale (float): scale on L^VF term
            c1 from ppo paper
        entropy_scale (float): scale term for entropy
            c2 from ppo paper
    """
    prob_ratio = _gauss_prob_ratio(action,
                                   pi_new_mu, pi_new_precision,
                                   tf.stop_gradient(pi_old_mu),
                                   tf.stop_gradient(pi_old_precision))
    l_clip = clipped_surrogate_loss(prob_ratio, advantage, eta)
    l_vf = tf.math.pow(critic_pred - value_target, 2.)
    # entropy for gaussian with diagonal covar = k * log(det(var))
    #       = k * log(prod(1 / prec_i)) = -k2 * sum(prec_i)
    neg_ent = tf.math.reduce_sum(pi_new_precision, axis=1)
    return l_vf - vf_scale * l_clip + entropy_scale * neg_ent
