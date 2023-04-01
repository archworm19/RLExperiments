"""Keras Layer Signature wrappers

    # > test tensor design
    #       Sigs are not layers ~ they encapsulate layer
    #       __call__(...) Returns 2 things
    #           1. layer.call(...) -> target tensor of interest
    #           2. bool tensor = logical_and of all test cases
    #       How to use?
    #           > make a test version of model --> run test on first train...
    #           > implementing layers must match signatures and make tests pass
    #       Q? does this add a bunch of overhead?
    #           It shouldn't... cuz main model doesn't use the test tensors
    #        

    """
import tensorflow as tf
from typing import List, Tuple
from tensorflow.keras.layers import Layer


class ScalarModel():
    # layer must implement
    #   action_t + state_t --> scalar (shape = batch_size)

    def __init__(self, layer: Layer):
        self.layer = layer

    def __call__(self, action_t: tf.Tensor, state_t: List[tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Args:
            action_t (tf.Tensor): action
                batch_size x ...
            state_t (List[tf.Tensor]): set of states
                each tensor is:
                    batch_size x ...

        Returns:
            tf.Tensor: shape = batch_size
            tf.Tensor: single boolean ~ true if tests pass
        """
        v = self.layer(action_t, state_t)
        v_test = tf.math.reduce_all(tf.shape(tf.shape(v)) == 1)
        return v, v_test


class ScalarStateModel():
    # layer must implement
    #   state_t --> scalar (shape = batch_size)

    def __init__(self, layer: Layer):
        self.layer = layer

    def __call__(self, state_t: List[tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Args:
            state_t (List[tf.Tensor]): set of states
                each tensor is:
                    batch_size x ...

        Returns:
            tf.Tensor: shape = batch_size
            tf.Tensor: single boolean ~ true if tests pass
        """
        v = self.layer(state_t)
        v_test = tf.math.reduce_all(tf.shape(tf.shape(v)) == 1)
        return v, v_test


class DistroModel():
    # distributional model
    # layer must implement:
    #   action_t + state_t --> batch_size x d (normalized across d)

    def __init__(self, layer: Layer):
        self.layer = layer

    def __call__(self, action_t: tf.Tensor, state_t: List[tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
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
            tf.Tensor: single boolean ~ true if tests pass
        """
        v = self.layer(action_t, state_t)
        v_test = tf.math.logical_and(tf.math.reduce_all(tf.shape(tf.shape(v)) == 2),
                                     tf.math.reduce_all(tf.math.abs(tf.math.reduce_sum(v, axis=1) - 1.) < .001))
        return v, v_test


class DistroStateModel():
    # distributional model

    def __init__(self, layer: Layer):
        self.layer = layer

    def __call__(self, state_t: List[tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Args:
            state_t (List[tf.Tensor]): set of states
                each tensor is:
                    batch_size x ...

        Returns:
            tf.Tensor: shape = batch_size x d
                where d = number of elements representing distribution
            tf.Tensor: single boolean ~ true if tests pass
        """
        v = self.layer(state_t)
        v_test = tf.math.logical_and(tf.math.reduce_all(tf.shape(tf.shape(v)) == 2),
                                     tf.math.reduce_all(tf.math.abs(tf.math.reduce_sum(v, axis=1) - 1.) < .001))
        return v, v_test


class VectorModel():
    # aka: "single timepoint model"

    def __init__(self, layer: Layer):
        self.layer = layer

    def __call__(self, action_t: tf.Tensor, state_t: List[tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Args:
            action_t (tf.Tensor): action
                batch_size x ...
            state_t (List[tf.Tensor]): set of states
                each tensor is:
                    batch_size x ...

        Returns:
            tf.Tensor: shape = batch_size x d
            tf.Tensor: single boolean ~ true if tests pass
        """
        v = self.layer(action_t, state_t)
        v_test = tf.math.reduce_all(tf.shape(tf.shape(v)) == 2)
        return v, v_test


class VectorStateModel():
    # aka: "single timepoint model"

    def __init__(self, layer: Layer):
        self.layer = layer

    def __call__(self, state_t: List[tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Args:
            state_t (List[tf.Tensor]): set of states
                each tensor is:
                    batch_size x ...

        Returns:
            tf.Tensor: shape = batch_size x d
            tf.Tensor: single boolean ~ true if tests pass
        """
        v = self.layer(state_t)
        v_test = tf.math.reduce_all(tf.shape(tf.shape(v)) == 2)
        return v, v_test


class VectorTimeModel():
    # aka: "multiple timepoint model"

    def __init__(self, layer: Layer):
        self.layer = layer

    def __call__(self, action_t: tf.Tensor, state_t: List[tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Args:
            action_t (tf.Tensor): action
                batch_size x T x ...
            state_t (List[tf.Tensor]): set of states
                each tensor is:
                    batch_size x T x ...

        Returns:
            tf.Tensor: shape = batch_size x T x d
            tf.Tensor: single boolean ~ true if tests pass
        """
        v = self.layer(action_t, state_t)
        v_test0 = tf.math.reduce_all(tf.shape(tf.shape(v)) == 3)
        v_test1 = tf.math.reduce_all(tf.shape(action_t)[:2] == tf.shape(v)[:2])
        return v, tf.math.logical_and(v_test0, v_test1)


# TODO: are there fast tests for model parallelism?
# > don't think so... natural tests are:
# 1. using gradient (sloow),
# 2. manipulating one model --> seeing if others change (sloow)


class ParallelModels():
    # multiple single-timepoint models operating in parallel
    # NOTE: without parallel test --> this is identical to VectorModel

    def __init__(self, layer: Layer):
        self.layer = layer

    def __call__(self, action_t: tf.Tensor, state_t: List[tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Args:
            action_t (tf.Tensor): action
                batch_size x num_model x ...
            state_t (List[tf.Tensor]): set of states
                each tensor is:
                    batch_size x num_model x ...

        Returns:
            tf.Tensor: shape = batch_size x num_model x d
            tf.Tensor: single boolean ~ true if tests pass
        """
        v = self.layer(action_t, state_t)
        v_test0 = tf.math.reduce_all(tf.shape(tf.shape(v)) == 3)
        v_test1 = tf.math.reduce_all(tf.shape(action_t)[:2] == tf.shape(v)[:2])
        return v, tf.math.logical_and(v_test0, v_test1)


class ParallelTimeModels():
    # multiple temporal models operating in parallel

    def __init__(self, layer: Layer):
        self.layer = layer

    def __call__(self, action_t: tf.Tensor, state_t: List[tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Args:
            action_t (tf.Tensor): action
                batch_size x num_model x T x ...
            state_t (List[tf.Tensor]): set of states
                each tensor is:
                    batch_size x num_model x T x ...

        Returns:
            tf.Tensor: shape = batch_size x num_model x T x d
            tf.Tensor: single boolean ~ true if tests pass
        """
        v = self.layer(action_t, state_t)
        v_test0 = tf.math.reduce_all(tf.shape(tf.shape(v)) == 4)
        v_test1 = tf.math.reduce_all(tf.shape(action_t)[:3] == tf.shape(v)[:3])
        return v, tf.math.logical_and(v_test0, v_test1)
