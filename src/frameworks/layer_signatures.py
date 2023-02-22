"""Keras Layer Signatures ~ recommend more explicit call signatures
    than standard keras
    
    # tensor wrappers ~ pseudo dataclass
    #   > wraps tensor
    #   > provides check bit = indicates whether tensor has right properties
    """
import tensorflow as tf
from typing import List
from tensorflow.keras.layers import Layer


class ScalarTensor():
    def __init__(self, x: tf.Tensor):
        # x: shape = batch_size
        self.tensor = x
        self.check_bit = tf.math.reduce_all(tf.shape(tf.shape(x)) == 1)


class VectorTensor():
    def __init__(self, x: tf.Tensor):
        # x: shape = batch_size x d
        self.tensor = x
        self.check_bit = tf.math.reduce_all(tf.shape(tf.shape(x)) == 2)


class DistroTensor():
    def __init__(self, x: tf.Tensor):
        # x: shape = batch_size x d
        #       and normalized across d
        self.tensor = x
        self.check_bit = tf.math.logical_and(tf.math.reduce_all(tf.shape(tf.shape(x)) == 2),
                                             tf.math.reduce_all(tf.math.abs(tf.math.reduce_sum(x, axis=1) - 1.) < .001))


class ScalarModel(Layer):

    def __init__(self):
        super(ScalarModel, self).__init__()

    def call(self, action_t: tf.Tensor, state_t: List[tf.Tensor]) -> ScalarTensor:
        """
        Args:
            action_t (tf.Tensor): action
                batch_size x ...
            state_t (List[tf.Tensor]): set of states
                each tensor is:
                    batch_size x ...

        Returns:
            ScalarTensor:
        """
        pass


class ScalarStateModel(Layer):

    def __init__(self):
        super(ScalarStateModel, self).__init__()

    def call(self, state_t: List[tf.Tensor]) -> ScalarTensor:
        """
        Args:
            state_t (List[tf.Tensor]): set of states
                each tensor is:
                    batch_size x ...

        Returns:
            ScalarTensor
        """
        pass


class DistroModel(Layer):
    # distributional model

    def __init__(self):
        super(DistroModel, self).__init__()

    def call(self, action_t: tf.Tensor, state_t: List[tf.Tensor]) -> DistroTensor:
        """
        Args:
            action_t (tf.Tensor): action
                batch_size x ...
            state_t (List[tf.Tensor]): set of states
                each tensor is:
                    batch_size x ...

        Returns:
            DistroTensor
        """
        pass


class DistroStateModel(Layer):
    # distributional model

    def __init__(self):
        super(DistroStateModel, self).__init__()

    def call(self, state_t: List[tf.Tensor]) -> DistroTensor:
        """
        Args:
            state_t (List[tf.Tensor]): set of states
                each tensor is:
                    batch_size x ...

        Returns:
            DistroTensor:
        """
        pass
