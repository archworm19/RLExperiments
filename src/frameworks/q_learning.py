"""Deep Q Learning Frameworks"""
import tensorflow as tf
from typing import List
from abc import ABC
from tensorflow.keras.layers import Layer


class ScalarModel(ABC):

    def eval(self,
             action_t: tf.Tensor,
             state: List[tf.Tensor]):
        """evaluate scalar model

        Args:
            action_t (tf.Tensor): most recent action a_t
                shape = batch_size x action_dims
                typically one-hot
            state (List[tf.Tensor]): set of input tensors
                each with shape:
                    batch_size x ...
    
        Returns:
            tf.Tensor: scalar results
                tensor shape = batch_size
        """
        pass


# TODO: Q Learning interface?
# > select_action
# > evaluate_action


class QLearning(Layer):

    def __init__(self, model: ScalarModel, num_actions: int):
        # NOTE: currently assumes discrete action space
        super(QLearning, self).__init__()
        self.model = model
        self.num_actions = num_actions

    def _build_action_probes(self, state: List[tf.Tensor]):
        # helper function for action selection
        # returns all possible action-state combos
        #   for given timestep
        # ASSUMES: state elements are unbatched
        #       = single timepoint
        action_t = tf.eye(self.num_actions)
        tile_states = []
        for sti in state:
            num_dims = tf.shape(tf.shape(sti))[0]
            tile_sh = tf.concat([[self.num_actions],
                                 tf.ones(num_dims, tf.int32)],
                                 axis=0)
            tile_states.append(tf.tile(tf.expand_dims(sti, 0),
                                       tile_sh))
        return action_t, tile_states

    def select_action(self,
                      state: List[tf.Tensor]):
        """select action that maximizes Q
            Q(S_t, A_t; sigma_t) is implemented via the injected model
            the model outputs scalar values for each batch sample

            NOTE: this assumes discrete, exclusive action space
                to generalize --> inject action selections strategy

        Args:
            state (List[tf.Tensor]): set of unbatches input tensors
                each with shape:
                    ...

        Returns:
            tf.Tensor: index of action that yields largest Q evaluation
                shape = ()
        """
        action_t, tile_states = self._build_action_probes(state)
        # --> shape = num_action
        scores = self.model.eval(action_t, tile_states)
        return tf.math.argmax(scores)



    # TODO: evaluation


# TODO: Double Q Learning
# probably should take in 2 QLearning objects
# ... could also take in 2 ScalarModel objects


if __name__ == "__main__":
    class FakeModel(ScalarModel):
        def __init__(self):
            super(FakeModel, self).__init__()

        def eval(self, action_t: tf.Tensor, state: List[tf.Tensor]):
            return tf.range(tf.shape(action_t)[0])


    state = [tf.ones([3, 5])]
    QL = QLearning(FakeModel(), 4)
    action_t, tile_states = QL._build_action_probes(state)
    assert tf.math.reduce_all(tf.shape(action_t) ==
                              tf.constant([4, 4], dtype=tf.int32))
    assert tf.math.reduce_all(tf.shape(tile_states[0]) ==
                              tf.constant([4, 3, 5], dtype=tf.int32))
    max_action = QL.select_action(state)
    assert tf.math.reduce_all(max_action == tf.constant(3, dtype=tf.int64))
