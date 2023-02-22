"""Forward Model Exploration Rewards (intrinsic motivation)
"""
import tensorflow as tf
from typing import List, Tuple
from frameworks.layer_signatures import VectorStateModel, MapperModel, VectorModel

# TODO: overarching design for frameworks
# > lowest level = tensors --> allows for more generalization
# > next level = models with certain signatures --> more foolproof


# TODO: forward model error ~ 'surprisal'
# > inputs?
#       1. states: x_t, x_{t+1}
#           TODO: worth it to generalize to more states?
#       2. action: a_t
# > networks?
#       1. embedding network: phi(x_t)
#       2. forward model: p(phi(x_{t+1}) | x_t, a_t)
#            TODO: should it be p(phi(x_{t+1}) | X, A)
#               whole action, state series
#               ... yes, at least give the option of state seqs
# > errors? start with this! (in terms of tensors)
#       1. Burda 2018 uses MSE between foreward model and phi(x_{t+1})


# TODO: rough thoughts on forward model?
# > p(phi(x_{t+1}) | x_{1...t}, a_{1...t})
# > do sequence to sequence models make sense? == does it make sense to predict a series?
#       ... also: forward models are trying to simulate the environment AND environment is likely stochastic
# > Q? do we need multi-step prediction or not?
#       if we want to do multi-step prediction/simulation --> probably yes (like in dream to control)
#       what about extrinsic reward?
#           would be interesting to try:
#                   reward_i = sum_{tau} [ forward_error(x_t | x_{1...t-tau}) ]


# TODO: encoding models? phi
# > random phi ~ no machinery needed here
# > Inverse Dynamics Features (IDF)
#       maths?
#           inverse model = p(a_t | phi(s_t), phi(s_{t+1}))
#       1. first level: errors using tensors ~ need separate for discrete and continuous action spaces
#       2. second level: wrap first level with function signatures!


def forward_surprisal(forward_model: VectorModel,
                      encoder: VectorStateModel,
                      state: List[tf.Tensor],
                      state_t1: List[tf.Tensor],
                      action: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """forward surprisal for single step prediction + continuous encoding
        NOTE: error only propagates through forward model ~ encoder is static
        NOTE: this is pretty basic --> would probably be better to include multiple
            timesteps or move to markovian predictions

        MSE(f(s_t, a_t), phi(s_{t+1})

    Args:
        forward_model (VectorModel): maps from action_t, state_t --> phi(state_t1)
        encoder (VectorStateModel): phi
        state (List[tf.Tensor]):
        state_t1 (List[tf.Tensor]):
        action (tf.Tensor):

    Returns:
        tf.Tensor: forward surprisal error; shape = batch_size
        tf.Tensor: test bit; true if signature tests pass
    """
    # --> batch_size x d
    phi1, phi_test = encoder(state_t1)
    y, for_test = forward_model(action, state)
    full_test = tf.math.logical_and(phi_test, for_test)
    return tf.math.reduce_sum(tf.pow(y - tf.stop_gradient(phi1), 2.), axis=1), full_test


def inverse_dynamics_error(encoder: VectorStateModel,
                           encode_to_action_layer: MapperModel,
                           state: List[tf.Tensor],
                           state_t1: List[tf.Tensor],
                           action: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """inverse dynamics error
        goal: train an encoder to only encode 'controllable features'
        encoder: phi(s)
        error = MSE(v(phi(s_t), phi(s_{t+1})), a_t)
            NOTE: this function uses MSE --> only appropriate for continuous
                action space

    Args:
        encoder (VectorStateModel): phi
        encode_to_action_layer (Layer): v
        state (List[tf.Tensor]): state at time t
        state_t1 (List[tf.Tensor]): state at time t+1
        action (tf.Tensor): action at time t

    Returns:
        tf.Tensor: error for each sample
            shape = batch_size
        tf.Tensor: test bit; true if signature tests pass
    """
    # --> batch_size x d
    phi0, phi_test = encoder(state)
    phi1, _ = encoder(state_t1)
    # --> batch_size x (2 * d)
    phi_combo = tf.concat([phi0, phi1], axis=1)
    # --> batch_size x action_dims
    apred, ea_test = encode_to_action_layer(phi_combo)
    full_test = tf.math.logical_and(phi_test, ea_test)
    return tf.math.reduce_sum(tf.pow(apred - action, 2.), axis=1), full_test
