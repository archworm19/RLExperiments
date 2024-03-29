"""Proximal Policy Update Framework (TRPO, PPO)
    PPO paper: Proximal Policy Optimization Algorithms
        Schulman et al, 2017
"""
import tensorflow as tf
import numpy as np
from typing import List, Dict


def clipped_surrogate_likelihood(probability_ratio: tf.Tensor,
                           advantage: tf.Tensor,
                           eta: float):
    """Clipped surrogate likelihood from PPO paper
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
        tf.Tensor: surrogate likelihood for each sample
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


def value_conv(Vend: float,
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
        Vend (float): value_model(state) --> V
            full shape of V = T
            here, we need V(T)
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
        rv = np.concatenate((reward, [Vend]), axis=0)
    return _right_conv(rv, gamma)


# TODO: requires testing
def package_dataset_critic(states: List[Dict[str, np.ndarray]],
                           reward: List[np.ndarray],
                           Vpred_end: List[float],
                           terminated: List[float],
                           gamma: float,
                           val_name: str = "val"):
    # ASSUMES: states and values have T as 0 axis
    #   NOTE: only need the last critic prediction for each trajectory
    # get values
    vals = [value_conv(vi, ri, gamma, termi) for vi, ri, termi in
            zip(Vpred_end, reward, terminated)]
    d = {}
    for k in states[0]:
        d[k] = np.concatenate([st[k] for st in states])
    d[val_name] = np.concatenate(vals, axis=0)
    dset = tf.data.Dataset.from_tensor_slices(d)
    return dset.shuffle(np.shape(d[val_name])[0])


def package_dataset(states: List[Dict[str, np.ndarray]],
                    Vpred: List[np.ndarray],
                    reward: List[np.ndarray],
                    actions: List[np.ndarray],
                    terminated: List[bool],
                    gamma: float,
                    lam: float,
                    adv_name: str = "adv",
                    val_name: str = "val",
                    action_name: str = "action"):
    """package dataset for ppo training actor

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
            zip(Vpred, reward, terminated)]
    # NOTE: last value term = V(s_T) ~ no training signal there --> lop it off
    # TODO: could this be the problem? is critic not learning termination; i don't think so
    vals = [value_conv(vi[-1], ri, gamma, termi)[:-1] for vi, ri, termi in
            zip(Vpred, reward, terminated)]

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
                        action: tf.Tensor,
                        advantage: tf.Tensor,
                        eta: float):
    """ppo loss for multiclass distribution actors
        = returns the various losses from ppo paper for actor error
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

    Returns:
        tf.Tensor: clipped surrogate actor loss for each sample (-1 * likelihood)
        tf.Tensor: negentropy
            all shapes = batch_size
    """
    prob_old = tf.stop_gradient(tf.math.reduce_sum(pi_old_distro * action, axis=1))
    prob_new = tf.math.reduce_sum(pi_new_distro * action, axis=1)
    prob_ratio = tf.math.divide(prob_new, prob_old)
    l_clip = clipped_surrogate_likelihood(prob_ratio, advantage, eta)
    negentropy = tf.math.reduce_sum(pi_new_distro * tf.math.log(pi_new_distro), axis=1)
    return -1.*l_clip, negentropy


def value_loss(critic_pred: tf.Tensor, value_target: tf.Tensor):
    # value function loss for each sample
    return tf.math.pow(critic_pred - value_target, 2.)


def _calc_precision(log_std_dev: tf.Tensor):
    # log_std_dev = log(standard deviation)
    #  shape must be:
    #       batch_size x d (diagonal covariance case)
    # convert to precision (1 / covariance)
    std_dev = tf.math.exp(log_std_dev)
    covar = tf.math.pow(std_dev, 2.)
    return tf.math.divide(1., covar)


def _gauss_prob_ratio2(x: tf.Tensor,
                       mu_num: tf.Tensor, log_std_dev_num: tf.Tensor,
                       mu_denom: tf.Tensor, log_std_dev_denom: tf.Tensor):
    # num = numerator; denom = denominator;
    # mu = mean = must be batch_size x action_dims
    # log_std_dev = must be batch_size x action_dims
    # NOTE: computes exp(log(gaussian ratios))
    prec_num = _calc_precision(log_std_dev_num)
    prec_denom = _calc_precision(log_std_dev_denom)
    log_det = 0.5 * tf.math.reduce_sum(tf.math.log(prec_num) - tf.math.log(prec_denom), axis=1)
    diff_num = x - mu_num
    log_exp_num = (-0.5) * tf.math.reduce_sum(diff_num * prec_num * diff_num, axis=1)
    diff_denom = x - mu_denom
    log_exp_denom = (-0.5) * tf.math.reduce_sum(diff_denom * prec_denom * diff_denom, axis=1)
    return tf.math.exp(log_det + log_exp_num - log_exp_denom)


def ppo_loss_gauss(pi_old_mu: tf.Tensor, pi_old_log_std_dev: tf.Tensor,
                   pi_new_mu: tf.Tensor, pi_new_log_std_dev: tf.Tensor,
                   action: tf.Tensor,
                   advantage: tf.Tensor,
                   eta: float):
    """ppo loss for gaussian distribution actors

    Args:
        pi_old_mu (tf.Tensor): mean outputs by old actor
            shape = batch_size x action_dims
        pi_old_log_std_dev (tf.Tensor): log(standard deviation) output by old actor
            shape = batch_size x action_dims = diagonal covar
        pi_new_mu (tf.Tensor):
        pi_new_log_std_dev (tf.Tensor):
        action (tf.Tensor): one-hot actions
            shape = batch_size x num_actions
        advantage (tf.Tensor): estimated advantage
            shape = batch_size
        eta (float): allowable step size; used by clipped surrogate

    Returns:
        tf.Tensor: clipped surrogate loss for each point
        tf.Tensor: negative entropy for each point
        tf.Tensor: prob ratio for each point
            all shapes = batch_size
    """
    prob_ratio = _gauss_prob_ratio2(action,
                                    pi_new_mu, pi_new_log_std_dev,
                                    tf.stop_gradient(pi_old_mu),
                                    tf.stop_gradient(pi_old_log_std_dev))
    l_clip = clipped_surrogate_likelihood(prob_ratio, advantage, eta)
    # entropy for gaussian with diagonal covar = k * log(det(var))
    #       = k * log(prod(var)) cuz diagonal
    #       = k * log(prod(std_dev^2)) cuz diagonal
    #       = k * sum [ log(std_dev_i ^ 2) ]
    #       = 2k * sum [ log(std_dev_i) ]
    ent = tf.math.reduce_sum(pi_new_log_std_dev, axis=1)
    return -1. * l_clip, -1. * ent, prob_ratio
