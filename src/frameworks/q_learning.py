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


# Model-Dependent Q calculation


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
                    stop_selection_grad: bool = True):
    """Calculate the Q error given 2 scalar models
        This is based on the idea of double Q learning
            but can be used for regular Q learning
            by passing in one model for both args
        Idea:
            > selection: greedily select an action using
                the selection model
            > evaluation: calculate max_a[Q(t+1)]
                using the evaluation model

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

    Returns:
        tf.Tensor: Q error
        tf.Tensor: Y_t = target
            both have shape = batch_size
    """
    max_a, _ = _greedy_select(selection_model, num_action, state_t1)
    # evaluate --> max_a[ Q(t+1) ] using eval model
    if stop_selection_grad:
        max_q_t1 = tf.stop_gradient(eval_model(tf.one_hot(max_a, num_action), state_t1))
    else:
        max_q_t1 = eval_model(tf.one_hot(max_a, num_action), state_t1)
    # termination:
    # if not term --> max_a[ Q(t+1)]
    # else --> 0
    max_q_t1 = (1. - termination) * max_q_t1
    q_t = q_model(action_t, state_t)
    return calc_q_error(q_t, max_q_t1, reward_t1, gamma)
