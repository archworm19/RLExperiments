"""Keras Layer Signatures ~ recommend more explicit call signatures
    than standard keras"""
import tensorflow as tf
from typing import List
from tensorflow.keras.layers import Layer


class ScalarModel(Layer):

    def __init__(self):
        super(ScalarModel, self).__init__()

    def call(self, action_t: tf.Tensor, state_t: List[tf.Tensor]) -> tf.Tensor:
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

    def call(self, state_t: List[tf.Tensor]) -> tf.Tensor:
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

    def call(self, action_t: tf.Tensor, state_t: List[tf.Tensor]) -> tf.Tensor:
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


class DistroStateModel(Layer):
    # distributional model

    def __init__(self):
        super(DistroStateModel, self).__init__()

    def call(self, state_t: List[tf.Tensor]) -> tf.Tensor:
        """
        Args:
            state_t (List[tf.Tensor]): set of states
                each tensor is:
                    batch_size x ...

        Returns:
            tf.Tensor: shape = batch_size x d
                where d = number of elements representing distribution
        """
        pass


class MapperModel(Layer):
    def __init__(self):
        super(MapperModel, self).__init__()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Args:
            x (tf.Tensor): batch_size x d1

        Returns:
            tf.Tensor: shape = batch_size x d2
        """
        pass


class VectorStateModel(Layer):
    # map states to continuous space

    def __init__(self):
        super(VectorStateModel, self).__init__()

    def call(self, state_t: List[tf.Tensor]) -> tf.Tensor:
        """
        Args:
            state_t (List[tf.Tensor]): set of states
                each tensor is:
                    batch_size x ...

        Returns:
            tf.Tensor: shape = batch_size x d
        """
        pass
