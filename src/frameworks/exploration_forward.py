"""Forward Model Exploration Rewards (intrinsic motivation)
    Also includes random network distillation
"""
import tensorflow as tf
from typing import List, Tuple
from frameworks.layer_signatures import VectorStateModel, MapperModel, VectorModel


def randomnet_error(rand_encoder: VectorStateModel,
                    learned_encoder: VectorStateModel,
                    state: List[tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
    """Random Network Distillation Error in continuous space
        First published by Burda
        squared error between random_encoder(s_t) and learned_encoder(s_t)

    Args:
        rand_encoder (VectorStateModel): random, fixed network encoder
        learned_encoder (VectorStateModel): learned network encoder
        state (List[tf.Tensor]):

    Returns:
        tf.Tensor: random network error for each sample
            shape = batch_size
        tf.Tensor: check bit
    """
    vr, br = rand_encoder(state)
    vl, bl = learned_encoder(state)
    check_bit = tf.math.logical_and(br, bl)
    return tf.math.reduce_mean(tf.math.pow(tf.stop_gradient(vr) - vl, 2.), axis=1), check_bit


def forward_surprisal(forward_model: VectorModel,
                      encoder: VectorStateModel,
                      state: List[tf.Tensor],
                      state_t1: List[tf.Tensor],
                      action: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """forward surprisal for single step prediction + continuous encoding
        NOTE: error only propagates through forward model ~ encoder is static
        NOTE: this is pretty basic --> would probably be better to include multiple
            timesteps or move to markovian predictions

        squared_error(f(s_t, a_t), phi(s_{t+1})

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
        error = squared_error(v(phi(s_t), phi(s_{t+1})), a_t)
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
