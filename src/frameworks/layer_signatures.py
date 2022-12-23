"""Keras Layer Signatures ~ recommend more explicit call signatures
    than standard keras"""
import tensorflow as tf
from typing import List
from tensorflow.keras.layers import Layer


class ScalarModel(Layer):

    def __init__(self):
        super(ScalarModel, self).__init__()

    def call(action_t: tf.Tensor, state_t: List[tf.Tensor]) -> tf.Tensor:
        """
        Args:
            action_t (tf.Tensor): action
                batch_size x ...
            state_t (List[tf.Tensor]): set of states
                each tensor is:
                    batch_size x ...

        Returns:
            tf.Tensor: shape = batch_size
        """
        pass


class ScalarStateModel(Layer):

    def __init__(self):
        super(ScalarStateModel, self).__init__()

    def call(state_t: List[tf.Tensor]) -> tf.Tensor:
        """
        Args:
            state_t (List[tf.Tensor]): set of states
                each tensor is:
                    batch_size x ...

        Returns:
            tf.Tensor: shape = batch_size
        """
        pass


class DistroModel(Layer):
    # distributional model

    def __init__(self):
        super(DistroModel, self).__init__()

    def call(action_t: tf.Tensor, state_t: List[tf.Tensor]) -> tf.Tensor:
        """
        Args:
            action_t (tf.Tensor): action
                batch_size x ...
            state_t (List[tf.Tensor]): set of states
                each tensor is:
                    batch_size x ...

        Returns:
            tf.Tensor: shape = batch_size x d
                where d = number of elements representing distribution
        """
        pass