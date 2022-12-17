"""Deep Q Learning Frameworks
"""
import tensorflow as tf
from typing import List
from tensorflow.keras.layers import Layer


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


def _greedy_select(selection_model: Layer,
                   num_action: int,
                   state_t1: List[tf.Tensor]):
    """greedy action selection using selection model

    Args:
        selection_model (Layer):
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


def calc_q_error_sm(q_model: Layer,
                    selection_model: Layer,
                    eval_model: Layer,
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

        TODO: expand this explanation

        Q = q_model
        Q_s = selection model
        Q_e = evaluation model

        Q_err = f(Q - Y_t)
        Y_t = r_t + gamma * max_a'[Q_e(s_{t+1}, a')]
        where a' is greedily chosen by Q_s

    Args:
        q_model (Layer): model that outputs Q(a_t, s_t)
        selection_model (Layer): model that selects action a'
        eval_model (Layer): model that calculates Q values for selected action
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


def calc_q_error_cont(q_model: Layer,
                      pi_model: Layer,
                      qprime_model: Layer,
                      piprime_model: Layer,
                      reward_t1: tf.Tensor,
                      state_t: List[tf.Tensor],
                      state_t1: List[tf.Tensor],
                      termination: tf.Tensor,
                      gamma: float,
                      huber: bool = True):
    """continuous-space Q error (DDPG)
        goal: find Q*, pi* such that Q*(s_t, pi*(s_t)) = r_t1 + gamma * Q*(s_t1, pi*(s_t1))
            if you've found such models for all of state space -->
            if can be shown that they accurately estimate expected reward over horizon
        Q learning approach:
            > for subset of state space and Q^i, pi^i --> find Q^{i+1}, pi^{i+1} that makes
                the above eqn hold: Q^{i+1}(pi^{i+1}) = r + gamma * Q^i(pi^i)
            > (not performed here): do this procedure iteratively, update Q and pi
            > to find Q^{i+1}: minimize Q_error (calculated here)
                Q_error = d(Q(s_t, pi(s_t)), r_t1 + gamma * Q'(s_t1, pi'(s_t1))
                    where d is some distance function (commonly L2)
        How is this different from discrete?
            have an actor model pi(a | s). In discrete case, only 1 model is needed
            and actions are chosen greedily by what action maximizes Q

    Args:
        q_model (Layer):
        pi_model (Layer):
            Q = Q_model(state_t, pi_model(state_t))
        qprime_model (Layer):
        piprime_model (Layer):
            Y_t = Qprime_model(state_t1, piprime_model(state_t1))
            these models are treated as fixed

            q model call signature:
                call(action: tf.Tensor, state: List[tf.Tensor]) -->
                    tf.Tensor with shape = batch_size
            prime model call signature:
                call(state: List[tf.Tensor]) -->
                    tf.Tensor with shape = batch_size

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
    # Q(state_t, pi(state_t))
    Qval = q_model(pi_model(state_t), state_t)
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
