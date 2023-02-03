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


def package_dataset(states: Dict[str, List[np.ndarray]],
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
    d = {k: np.concatenate([ski[:-1] for ski in states[k]], axis=0)
         for k in states}
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
                        vf_scale: float = 1.):
    # TODO: entropy term?
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

    Returns:
        tf.Tensor: loss for each sample
            shape = batch_size
    """
    prob_old = tf.stop_gradient(tf.math.reduce_sum(pi_old_distro * action, axis=1))
    prob_new = tf.math.reduce_sum(pi_new_distro * action, axis=1)
    prob_ratio = tf.math.divide(prob_new, prob_old)
    l_clip = clipped_surrogate_loss(prob_ratio, advantage, eta)
    l_vf = tf.math.pow(critic_pred - value_target, 2.)
    return l_vf - vf_scale * l_clip


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
                   vf_scale: float = 1.):
    # TODO: entropy term?
    """ppo loss for gaussian distribution actors
        = L^CLIP + L^VF from ppo paper for actor error
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
            c1 from 
    """
    prob_ratio = _gauss_prob_ratio(action,
                                   pi_new_mu, pi_new_precision,
                                   tf.stop_gradient(pi_old_mu),
                                   tf.stop_gradient(pi_old_precision))
    l_clip = clipped_surrogate_loss(prob_ratio, advantage, eta)
    l_vf = tf.math.pow(critic_pred - value_target, 2.)
    return l_vf - vf_scale * l_clip


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
    reward = np.ones((19,))
    gamma = 0.9
    At = advantage_conv(V, reward, gamma, 1.)

    for i in range(19):
        a_i = -1. * V[i]
        gfactor = 1.
        for j in range(i, 19):
            a_i += reward[j] * gfactor
            gfactor *= gamma
        a_i += gfactor * V[-1]
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
    s1 = np.zeros((11, 2))
    s2 = np.zeros((6, 2))
    v1 = np.ones((11,))
    v2 = np.ones((6,))
    r1 = np.ones((10,))
    r2 = np.ones((5,))
    a1 = np.ones((10, 3))
    a2 = np.ones((5, 3))
    terminated = [False, True]
    dset = package_dataset({"s": [s1, s2]}, [v1, v2], [r1, r2], [a1, a2], terminated, 0.9, 1.)
    for v in dset.batch(4):
        print(v)

    # TODO: gaussian tests
    from scipy.stats import multivariate_normal
    import numpy.random as npr
    rng = npr.default_rng(42)
    # seems like multivariate_normal can figure out how to use diagonal covar
    N = 10
    mu = rng.random((N, 2))
    var = rng.random((N, 2))
    mu2 = rng.random((N, 2))
    var2 = rng.random((N, 2))
    x = rng.random((N, 2))
    tf_ratio = _gauss_prob_ratio(tf.constant(x, dtype=tf.float32),
                                 tf.constant(mu, dtype=tf.float32),
                                 1. / tf.constant(var, dtype=tf.float32),
                                 tf.constant(mu2, dtype=tf.float32),
                                 1. / tf.constant(var2, dtype=tf.float32))
    for i in range(N):
        ratio = multivariate_normal.pdf(x[i:i+1], mu[i], var[i]) / multivariate_normal.pdf(x[i:i+1], mu2[i], var2[i])
        diff = np.fabs(tf_ratio[i].numpy() - ratio)
        assert diff < .001

    # TODO: ppo loss multiclass test
    # 3 "models"
    # > base model
    # > model that is very close to base model + improves advantage scale (should be best performance)
    # > model that is very far from base model + improves advantage scale (should be same as best)
    critic_pred = tf.zeros([8], dtype=tf.float32)
    action_np = np.zeros((8, 2))
    action_np[:4, 0] = 1.
    action_np[4:, 1] = 1.
    action = tf.constant(action_np, dtype=tf.float32)
    value_target = tf.zeros([8], dtype=tf.float32)
    advantage = tf.constant([1., -1.] * 2 + [-1., 1.] * 2, dtype=tf.float32)
    eta = 0.2  # step size
    # base model
    pi_base = tf.constant([[0.5, 0.5] for _ in range(8)], dtype=tf.float32)

    # Q? when do we get clipped?
    #   upper: x / base_prob = 1 + eta --> x_up = (1 + eta) * base_prob
    #   lower: x / base_prob = 1 - eta --> x_low = (1 - eta) * base_prob
    # here: base_prob = 0.5
    x_up = (1 + eta) * 0.5
    x_lo = (1 - eta) * 0.5

    # best model
    pi_best = tf.reshape(tf.constant([x_up, x_lo, x_lo, x_up] * 4, dtype=tf.float32), [8, 2])
    # wrong way
    pi_bad = 1. - pi_best
    # should be same as best
    pi_big = tf.reshape(tf.constant([0.99, 0.01, 0.01, 0.99] * 4, dtype=tf.float32), [8, 2])

    losses = []
    for v in [pi_best, pi_bad, pi_big]:
        loss = ppo_loss_multiclass(pi_base, v,
                                   critic_pred,
                                   action,
                                   advantage,
                                   value_target,
                                   eta)
        assert np.shape(loss.numpy()) == (8,)
        losses.append(tf.math.reduce_mean(loss).numpy())
    assert losses[0] < losses[1]
    assert np.round(losses[0], 4) == np.round(losses[2], 4)
