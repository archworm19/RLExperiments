"""Forward Model Error Intrinsic Rewards

    Baseline errors are so simple --> don't need bottom-level functions
    Just provide top-level functions
"""
import tensorflow as tf
from typing import List, Tuple
from frameworks.layer_signatures import VectorModel, VectorTimeModel, VectorStateModel


def forward_error(forward_model: VectorModel,
                  encoder: VectorStateModel,
                  action_t: tf.Tensor,
                  state_t: List[tf.Tensor],
                  state_t1: List[tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
    """L2 forward error
        L2(forward_model(state_t), encoder(state_t1))
        Requirement: encoder and forward_model must map into the same space!

    Args:
        forward_model (VectorModel): forward prediction model
        encoder (VectorStateModel): encoder model; operates on state_t1
        action_t (tf.Tensor): action
            batch_size x ...
        state_t (List[tf.Tensor]): state(t)
        state_t1 (List[tf.Tensor]): state(t+1)
            both states:
                each tensor is batch_size x ...

        Returns:
            tf.Tensor: error for each sample; shape = batch_size
            tf.Tensor: single boolean ~ true if tests pass
    """
    yp, f_test_bit = forward_model(action_t, state_t)
    y, e_test_bit = encoder(state_t1)
    err = tf.math.reduce_sum(tf.math.pow(yp - y, 2.), axis=1)
    return err, tf.math.logical_and(f_test_bit, e_test_bit)


def forward_time_error(forward_model: VectorTimeModel,
                       encoder: VectorStateModel,
                       action_t: tf.Tensor,
                       state_t: List[tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
    """L2 forward error for temporal model ~ whole sequence computed at same time
        L2(forward_model(state_t), encoder(state_t1))
        Requirement: encoder and forward_model must map into the same space!

    Args:
        forward_model (VectorTimeModel): forward prediction model
        encoder (VectorStateModel): encoder model; operates on state_t1
        action_t (tf.Tensor): action
            batch_size x T x ...
        state_t (List[tf.Tensor]): state(t)
            each tensor has shape: batch_size x T x ...

        Returns:
            tf.Tensor: error for each sample; shape = batch_size
            tf.Tensor: single boolean ~ true if tests pass
    """
    batch_size = tf.shape(action_t)[0]
    T = tf.shape(action_t)[1]
    # --> batch_size x T x d
    yp, f_test_bit = forward_model(action_t, state_t)

    # combine the first 2 dims (encode in parallel):
    state_res = []
    for si in state_t:
        init_shape = tf.shape(si)
        new_sh0 = tf.constant([batch_size * T], dtype=init_shape.dtype)
        new_shape = tf.concat([new_sh0, init_shape[2:]], axis=0)
        state_res.append(tf.reshape(si, new_shape))
    # --> (batch_size * T) x d   
    y, e_test_bit = encoder(state_res)
    y = tf.reshape(y, [batch_size, T, -1])

    err = tf.math.reduce_sum(tf.math.pow(yp - y, 2.), axis=1)
    return err, tf.math.logical_and(f_test_bit, e_test_bit)
