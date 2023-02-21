"""Deep Q Learning Frameworks
"""
import tensorflow as tf
from typing import List
from frameworks.layer_signatures import ScalarModel, ScalarStateModel, DistroModel


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
        qt1 = selection_model(v, state_t1)[0]  # NOTE: ignoring test tensors here
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

        TODO: doesn't return test tensor

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
    eval_tensor, eval_test = eval_model(tf.one_hot(max_a, num_action), state_t1)
    max_q_t1 = tf.stop_gradient(eval_tensor)
    # termination:
    # if not term --> max_a[ Q(t+1)]
    # else --> 0
    max_q_t1 = (1. - termination) * max_q_t1
    q_tensor, q_test = q_model(action_t, state_t)
    _test_tensor_bool = tf.math.logical_and(q_test, eval_test)
    if huber:
        return calc_q_error_huber(q_tensor, max_q_t1, reward_t1, gamma)
    return calc_q_error(q_tensor, max_q_t1, reward_t1, gamma)


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

    TODO: doesn't return test tensor

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
    Qval, q_test = q_model(action_t, state_t)
    # Q'(state_t1, pi(state_t1))
    Qp, qp_test = qprime_model(piprime_model(state_t1)[0],
                                             state_t1)
    Q_prime = tf.stop_gradient(Qp)
    # termination:
    # if not term --> Q'
    # else --> 0
    Q_prime = (1. - termination) * Q_prime
    _test_tensor_bool = tf.math.logical_and(q_test, qp_test)
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
    # TODO: ignores test tensors
    return -1. * tf.math.reduce_mean(q_model(pi_model(state_t)[0], state_t)[0])


# Distributional Approach


def _redistribute_weight(Vmin: float, Vmax: float,
                         atoms_probs: tf.Tensor,
                         reward_t1: tf.Tensor, gamma: float):
    """> apply distributional Bellman operator (T z_j)
       > redistribute probability of T z_j
           follows Bellamere et al, 2017

    Args:
        Vmin (float): minimum reward value
        Vmax (float): maximum reward value
            inclusive
        atoms_probs (tf.Tensor): atom component in probability space
            batch_size x num_atoms
            along with Vmin, Vmax --> describes a distribution
            (can be vizd with histogram)
        reward_t1 (tf.Tensor): reward at timestep t+1
            shape = batch_size
        gamma (float): discount factor

    Returns:
        tf.Tensor: weights in atom-z space for each batch sample
            = targets for (weighted) cross-entropy
            batch_size x num_atoms
            NOTE: will be approximately normalized across atoms
                but substantial possibility of numeric errors
    """
    # for illustrative purposes, we'll refer to
    # n_in = num atoms in; n_out = num atoms out
    # even tho n_in = n_out in this case

    num_atoms = tf.shape(atoms_probs)[1]
    # --> shape = n_out
    z = tf.cast(tf.linspace(Vmin, Vmax, num_atoms), atoms_probs.dtype)
    dz = z[1] - z[0]

    # T z_j = reward + gamma * z_j
    # reward --> batch_size x 1
    reward_exp = tf.expand_dims(reward_t1, axis=1)
    # --> batch_size x n_in
    Tz = tf.clip_by_value(reward_exp + gamma * tf.expand_dims(z, axis=0),
                          Vmin, Vmax)

    # calculate distance from each atom in T z_j
    # from each atom in z
    # --> batch_size x n_in x n_out
    diff = tf.expand_dims(Tz, 2) - tf.reshape(z, [1, 1, -1])
    dist = tf.math.abs(diff)

    # for atom in T z_j
    # contributing weight = (dz - dist) / dz
    # --> batch_size x n_in x n_out
    # NOTE: only care about the 2 closest atoms
    #   to each atom in T z   
    #   > (dz - dist) >= 0 for 2 closest (given earlier clipping)
    #   > (dz - dist) < 0 for all other atoms
    # so: just clip the contributing weights
    # KEY: normalized across axis 2
    contrib = tf.math.divide(tf.clip_by_value(dz - dist, 0., dz), dz)

    # scale contributions by probabilities
    # contrib = batch_size x n_in x n_out
    # atoms_probs = batch_size x n_in
    # --> weights = batch_size x n_out
    weights = tf.math.reduce_sum(contrib * tf.expand_dims(atoms_probs, 2),
                                 axis=1)
    return weights


def _calc_q_from_distro(Vmin: float, Vmax: float,
                        atoms_probs: tf.Tensor):
    """Q = Expected reward
            = sum_z [p(z) * z]
        where z = the value of the different atoms
            in reward space

    Args:
        Vmin (float): minimum reward value
        Vmax (float): maximum reward value
            inclusive
        atoms_probs (tf.Tensor): atom component in probability space
            batch_size x num_atoms
            along with Vmin, Vmax --> describes a distribution

    Returns:
        tf.Tensor: expected reward for each batch
            shape = batch_size

    """
    num_atoms = tf.shape(atoms_probs)[1]
    z = tf.cast(tf.linspace(Vmin, Vmax, num_atoms), atoms_probs.dtype)
    return tf.math.reduce_sum(tf.expand_dims(z, 0) * atoms_probs,
                              axis=1)


def calc_q_error_distro_discrete(q_model: DistroModel,
                                 selection_model: ScalarModel,
                                 eval_model: DistroModel,
                                 Vmin: float,
                                 Vmax: float,
                                 action_t: tf.Tensor,
                                 reward_t1: tf.Tensor,
                                 state_t: List[tf.Tensor],
                                 state_t1: List[tf.Tensor],
                                 termination: tf.Tensor,
                                 num_action: int,
                                 gamma: float,
                                 vector0: tf.Tensor):
    """distributional perspective Q error for discrete control
    follows Bellamere et al, 2017 (categorical model)
        > distribution representation: set of atoms parameterized
            by Vmin, Vmax (bounds in reward space) and probabilities
            output by DistroModels
        > Action Selection: Greedy (using select_model)
        > Q error calculation:
            applies Bellman operator (Tz) introduced in Bellamere
                atom probabilities come from eval_model
            Q error = cross_entropy(Tz, q_model(a_t, s_t))

    # TODO: ignores test tensors

    Args:
        q_model (DistroModel):
        selection_model (DistroModel): yields Q value for each
            batch sample ~ used to greedily select action
            NOTE: typically: take expectation over distribution
                model
        eval_model (DistroModel):
            NOTE: models are assumed to be unnormalized
            --> apply softmax to outputs to get
            probability space
        action_t (tf.Tensor): action at time t
        reward_t1 (tf.Tensor): rewards at time t+1
        state_t (List[tf.Tensor]): state at time t
        state_t1 (List[tf.Tensor]): state at time t+1
        termination (tf.Tensor): termination bit
            1 if terminal
            shape = batch_size
        num_action (int): number of available actions
        gamma (float): discount factor
        vector0 (tf.Tensor): 0 representation for atoms
            if termination --> return 0 vector for Tz
            shape = num_atoms x num_actions

    Returns:
        tf.Tensor: Q error
            shape = batch_size
        tf.Tensor: weights
            batch_size x num_atoms
        tf.Tensor: selected action tensor
            batch_size x num_action
    """
    # greedy selection
    max_a, _ = _greedy_select(selection_model, num_action, state_t1)
    max_a_vector = tf.one_hot(max_a, num_action)

    # apply Bellman operator --> target (weights of X-entropy)
    # --> batch_size x num_atoms
    # NOTE: if termination bit specified --> probability vector = vector0
    #       --> effective Tz = reward
    #       else: use model
    term_exp = tf.expand_dims(termination, axis=1)
    v0_exp = tf.expand_dims(vector0, axis=0)
    atoms_probs_target = (term_exp * v0_exp +
                          (1.0 - term_exp) * tf.nn.softmax(tf.stop_gradient(eval_model(max_a_vector, state_t1)[0]),
                                                              axis=1))
    weights = _redistribute_weight(Vmin, Vmax, atoms_probs_target, reward_t1, gamma)

    # x-entropy with weights as targets
    # model outputs v_i
    # softmax: p(v_i) = exp(v_i) / sum(exp(v_1) + ...)
    # log(p(v_i)) = v_i - log_sum_exp(V)
    # x-entropy: - sum_i [weight_i * log p(v_i)]
    #       = - sum_i [w_i * v_i - w_i * log_sum_exp(V)]
    #       = - sum_i [w_i * v_i] + log_sum_exp(V) sum_i [w_i]
    #       = log_sum_exp(V) - sum_i [ w_i * v_i]
    atoms_v_q = q_model(action_t, state_t)[0]
    return (tf.math.reduce_logsumexp(atoms_v_q, axis=1) -
            tf.math.reduce_sum(weights * atoms_v_q, axis=1)), weights, max_a_vector
