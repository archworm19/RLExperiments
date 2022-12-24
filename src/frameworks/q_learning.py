"""Deep Q Learning Frameworks
"""
import tensorflow as tf
from typing import List
from tensorflow.keras.layers import Layer
from frameworks.layer_signatures import ScalarModel, ScalarStateModel


# Model-Agnostic Q Error


def calc_q_error(q_t: tf.Tensor, max_q_t1: tf.Tensor,
                 reward_t1, gamma: float):
    """Q error

        Q error = (Y_t - Q(S_t, A_t; sigma_t))^2
            where Y_t = R_{t+1} + gamma * max_a[ Q(S_{t + 1}, a; sigma_t^-1 ]

        NOTE: this can be used with q learning or double q learning

    Args:
        q_t (tf.Tensor): Q(s_{t}, a_{t})
            shape = batch_size
        max_q_t1 (tf.Tensor): max_a' [Q(s_{t+1}, a')]
            shape = batch_size
        reward_t1 (_type_): reward at time t+1
            shape = batch_size
        gamma (float): discount factor

    Returns:
        tf.Tensor: Q error
        tf.Tensor: Y_t = target
            both have shape = batch_size
    """
    Y_t = reward_t1 + gamma * max_q_t1
    return tf.math.pow(Y_t - q_t, 2.), Y_t


def calc_q_error_huber(q_t: tf.Tensor, max_q_t1: tf.Tensor,
                       reward_t1, gamma: float,
                       huber_delta: float = 1.):
    # same as above but uses Huber instead of L2
    Y_t = reward_t1 + gamma * max_q_t1
    di = Y_t - q_t
    # Huber
    abs_di = tf.math.abs(di)
    # small mag: if |a| <= delta --> 1/2 * a^2
    small_mag = 0.5 * tf.math.pow(di, 2.)
    # large mag: if |a| > delta --> delta * (|a| - .5 delta)
    large_mag = huber_delta * (abs_di - 0.5 * huber_delta)
    small_mask = tf.cast(abs_di <= huber_delta, small_mag.dtype)
    err = small_mask * small_mag + (1. - small_mask) * large_mag
    return err, Y_t


# Model Helpers


def _greedy_select(selection_model: ScalarModel,
                   num_action: int,
                   state_t1: List[tf.Tensor]):
    """greedy action selection using selection model

    Args:
        selection_model (ScalarModel):
        num_action (int):
        state_t1 (List[tf.Tensor]):
            see calc_q_error_sm

    Returns:
        tf.Tensor: idx of maximizing action for each
            batch_sample
            shape = batch_size
        tf.Tensor: q scores for all options
            shape = batch_size x num_actions
    """
    qsels = []
    for i in range(num_action):
        # --> shape = num_action
        v_hot = tf.one_hot(i, num_action)
        # --> shape = batch_size x num_action
        v = tf.tile(tf.expand_dims(v_hot, 0),
                    [tf.shape(state_t1[0])[0], 1])
        # qt1: shape = batch_size
        qt1 = selection_model(v, state_t1)
        qsels.append(qt1)
    # indices of greedily selected actions
    # --> shape = batch_size
    qsels = tf.stack(qsels, axis=1)
    max_a = tf.argmax(qsels, axis=1)
    return max_a, qsels


# Model-Dependent Q calculation


def calc_q_error_sm(q_model: ScalarModel,
                    selection_model: ScalarModel,
                    eval_model: ScalarModel,
                    action_t: tf.Tensor,
                    reward_t1: tf.Tensor,
                    state_t: List[tf.Tensor],
                    state_t1: List[tf.Tensor],
                    termination: tf.Tensor,
                    num_action: int,
                    gamma: float,
                    huber: bool = True):
    """Calculate the Q error given 3 scalar models
        Flexibile enough to be used by multiple kinds of Q learning

        Q = q_model
        Q_s = selection model
        Q_e = evaluation model

        Q_err = f(Q - Y_t)
        Y_t = r_t + gamma * max_a'[Q_e(s_{t+1}, a')]
        where a' is greedily chosen by Q_s

    Args:
        q_model (ScalarModel): model that outputs Q(a_t, s_t)
        selection_model (ScalarModel): model that selects action a'
        eval_model (ScalarModel): model that calculates Q values for selected action
            = Q(a', s_{t+1})
            scalar_model assumption ~ applies to q/selection/eval models:
                call has the following signature:
                    call(action_t: tf.Tensor, state_t: List[tf.Tensor])
        action_t (tf.Tensor): action at time t
        reward_t1 (tf.Tensor): reward at time t+1 (follows from action_t)
        state_t (List[tf.Tensor]): state at time t
        state_t1 (List[tf.Tensor]): state at time t+1
            state_t + actiom_t --> reward_t1, state_t1 
        termination (tf.Tensor): binary array indicating whether
            samples is a terminal bit
            shape = batch_size
        num_actions (int): number of actions available to agent
            only works for discrete action space
        gamma (float): q learning discount factor
        huber (bool): whether to use huber loss

    Returns:
        tf.Tensor: Q error
        tf.Tensor: Y_t = target
            both have shape = batch_size
    """
    max_a, _ = _greedy_select(selection_model, num_action, state_t1)
    # evaluate --> max_a[ Q(t+1) ] using eval model
    max_q_t1 = tf.stop_gradient(eval_model(tf.one_hot(max_a, num_action), state_t1))
    # termination:
    # if not term --> max_a[ Q(t+1)]
    # else --> 0
    max_q_t1 = (1. - termination) * max_q_t1
    q_t = q_model(action_t, state_t)
    if huber:
        return calc_q_error_huber(q_t, max_q_t1, reward_t1, gamma)
    return calc_q_error(q_t, max_q_t1, reward_t1, gamma)


def calc_q_error_critic(q_model: ScalarModel,
                        qprime_model: ScalarModel,
                        piprime_model: ScalarStateModel,
                        action_t: tf.Tensor,
                        reward_t1: tf.Tensor,
                        state_t: List[tf.Tensor],
                        state_t1: List[tf.Tensor],
                        termination: tf.Tensor,
                        gamma: float,
                        huber: bool = True):
    """continuous-space Q error (DDPG)

    Q = q_model
    pi = pi_model

    Q_err = f(Q(a_t, s_t) - Y_t)
        where Y_t = r_t1 + gamma * Q'(pi'(s_t1), s_t1)
    difference from continuous space?
        > no greedy ~ selects actions based on action model pi
    KEY: prime models are frozen here ~ just optimizing Q

    Args:
        q_model (ScalarModel):
        qprime_model (ScalarModel):
        piprime_model (ScalarStateModel):
            q model call signature:
                call(action: tf.Tensor, state: List[tf.Tensor]) -->
                    tf.Tensor with shape = batch_size
            prime model call signature:
                call(state: List[tf.Tensor]) -->
                    tf.Tensor with shape = batch_size

        action_t (tf.Tensor): action at time t
        reward_t1 (tf.Tensor): reward at time t+1 (follows from action_t)
        state_t (List[tf.Tensor]): state at time t
        state_t1 (List[tf.Tensor]): state at time t+1
            state_t + actiom_t --> reward_t1, state_t1 
        termination (tf.Tensor): binary array indicating whether
            samples is a terminal bit
            shape = batch_size
        gamma (float): q learning discount factor
        huber (bool): whether to use huber loss

    Returns:
        tf.Tensor: Q error
        tf.Tensor: Y_t = target
            both have shape = batch_size
    """
    # Q
    Qval = q_model(action_t, state_t)
    # Q'(state_t1, pi(state_t1))
    Q_prime = tf.stop_gradient(qprime_model(piprime_model(state_t1),
                                            state_t1))
    # termination:
    # if not term --> Q'
    # else --> 0
    Q_prime = (1. - termination) * Q_prime
    if huber:
        return calc_q_error_huber(Qval, Q_prime, reward_t1, gamma)
    return calc_q_error(Qval, Q_prime, reward_t1, gamma)


def calc_q_error_actor(q_model: ScalarModel,
                       pi_model: ScalarStateModel,
                       state_t: List[tf.Tensor]):
    """deterministic policy gradient for action model pi
    return the gradient wrt parameters of model pi
    can use this gradient to perform gradient ascent

    grad J approx= (1/N) sum_i [grad_{a} Q grad_{theta} mu(s)]
        = grad_{theta} mean(Q)
    error = negative of mean(Q)
    NOTE: no gradient blocking is applied
        --> will have to restrict trainable variables downstream

    Args:
        q_model (ScalarModel): no gradient thru critic
        pi_model (ScalarStateModel):
        state_t (List[tf.Tensor]): start at time t
            where each 

    Returns:
        tf.Tensor: -1 * Q() ~ scalar
    """
    return -1. * tf.math.reduce_mean(q_model(pi_model(state_t), state_t))


# Distributional Approach


def _redistribute_weight(Vmin: float, Vmax: float,
                         atoms_probs: tf.Tensor,
                         reward_t1: tf.Tensor, gamma: float):
    """> apply distributional Bellman operator (T z_j)
       > redistribute probability of T z_j
           follows Bellamere et al, 2017

    Args:
        Vmin (float):
        Vmax (float):
            inclusive
        num_atoms (int):
        atoms_probs (tf.Tensor): atom component in probability space
            batch_size x num_atoms
            together --> they describe a distribution
            (can be vizd with histogram)
        reward_t1 (tf.Tensor): reward at timestep t+1
            shape = batch_size
        gamma (float): discount factor

    Returns:
        tf.Tensor: weights in atom-z space for each batch sample
            = targets for (weighted) cross-entropy
            batch_size x num_atoms
    """
    num_atoms = tf.cast(tf.shape(atoms_probs)[1], tf.float32)
    dz = tf.math.divide(Vmax - Vmin, num_atoms - 1.)
    z = tf.range(Vmin, Vmax + dz, dz)

    # T z_j <-- r_t1 + gamma * z_j
    # --> batch_size x num_atoms
    Tz = tf.clip_by_value(tf.expand_dims(reward_t1, axis=1) + gamma * tf.expand_dims(z, axis=0),
                          Vmin, Vmax)

    # calculate distance from each atom in T z_j
    # from each atom in z
    # --> batch_size x num_atoms x num_atoms
    diff = tf.expand_dims(Tz, 2) - tf.reshape(z, [1, 1, -1])
    dist = tf.math.abs(diff)  # TODO: not sure I'll need this

    # for atom in T z_j
    # contributing weight = dz - dist
    # NOTE: only care about the 2 closest atoms
    #   to each atom in T z   
    #   > (dz - dist) >= 0 for 2 closest (given earlier clipping)
    #   > (dz - dist) < 0 for all other atoms
    # so: just clip the contributing weights
    contrib = tf.clip_by_value(dz - dist, 0., dz)

    # scale contributions of each atom in T z_j
    # by atom's probability
    weights = tf.math.reduce_sum(contrib * tf.expand_dims(atoms_probs, 1),
                                 axis=1)
    return weights


if __name__ == "__main__":
    # TODO: move these to tests

    # trivial example: state to same state due to gamma = 1.
    # remember: atoms_static (z) is in reward space
    Vmin = 1.
    Vmax = 3.
    atoms_probs = tf.constant([[0.0, 0.0, 1.0],
                               [1.0, 0.0, 0.0]], dtype=tf.float32)
    reward = tf.constant([0., 0.])
    weights = _redistribute_weight(Vmin, Vmax, atoms_probs, reward, 1.)
    print(weights)
    target = tf.constant([[0, 0, 1],
                          [1, 0, 0]], dtype=tf.float32)
    assert tf.math.reduce_all(tf.round(100. * weights) ==
                              tf.round(100. * target))

    # extreme case: extreme rewards that saturate at Vmin or Vmax
    v = 1. / 3.
    atoms_probs = tf.constant([[v, v, v],
                               [v, v, v]], dtype=tf.float32)
    reward = tf.constant([-50., 50.])
    weights = _redistribute_weight(Vmin, Vmax, atoms_probs, reward, 1.)
    print(weights)
    target = tf.constant([[1, 0, 0],
                          [0, 0, 1]], dtype=tf.float32)
    assert tf.math.reduce_all(tf.round(100. * weights) ==
                              tf.round(100. * target))
