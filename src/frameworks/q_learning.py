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

    def __init__(self, model: ScalarModel, num_actions: int,
                 gamma: float):
        # NOTE: currently assumes discrete action space
        assert gamma > 0. and gamma <= 1.
        super(QLearning, self).__init__()
        self.model = model
        self.num_actions = num_actions
        self.gamma = gamma

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

    def calc_target(self,
                    reward: tf.Tensor,
                    max_action: tf.Tensor,
                    state: List[tf.Tensor]):
        """Follows convention of original deep Q learning papers
            Mnih et al and Hasselt et al

            target = Y_t
            Y_t = R_{t+1} +
                  gamma * max_a [ Q(S_{t+1}, a; sigma_t)]

        Args:
            reward (tf.Tensor): reward at timestep t+1
                shape = batch_size (scalar reward)
            max_action (tf.Tensor): action that maximizes Q
                given state
                shape = batch_size x action_dims
                typically one-hot
            state (List[tf.Tensor]): state at time t+1
                set of input tensors
                each with shape:
                    batch_size x ...

        Returns:
            tf.Tensor: Y_t / target
                tensor shape = batch_size
                if underlying model is differentiable,
                    this output is differentiable
        """
        return (reward +
                self.gamma * self.model.eval(max_action, state))

    def call(self,
             reward: tf.Tensor,
             action_t: tf.Tensor,
             state_t: List[tf.Tensor],
             state_t1: List[tf.Tensor]):
        """Q error

            Q error = (Y_t - Q(S_t, A_t; sigma_t))^2
                where Y_t = R_{t+1} + gamma * max_a[ Q(S_{t + 1}, a; sigma_t^-1 ]

        Args:
            reward (tf.Tensor): reward at time t+1
                shape = batch_size
            action_t (tf.Tensor): action at time t
                shape = batch_size x num_action
            state_t (tf.Tensor): state at time t
                each tensor has shape batch_size x ...
            state_t1 (tf.Tensor): state at time t+1
                each tensor has shape batch_size x ...

        Returns:
            tf.Tensor: Q error (non-reduced)
                shape = batch_size
            tf.Tensor: Y_t
                shape = batch_size
        """
        # max action calc: get reward estimate
        # for each, for each batch elem
        scores = []
        for i in range(self.num_actions):
            batch_size = tf.shape(action_t)[0]
            action_t1 = tf.tile(tf.expand_dims(tf.one_hot(i, self.num_actions), 0),
                                [batch_size, 1])
            # Q(S_{t+1}, a_i)
            # --> shape = batch_size
            scores.append(self.model.eval(action_t1, state_t1))
        max_score = tf.math.reduce_max(tf.stack(scores, axis=0), axis=0)
        # bring it all together
        Y_t = reward + self.gamma * max_score
        # Q(S_t, A_t)
        Q_t = self.model.eval(action_t, state_t)
        return tf.math.pow(Y_t - Q_t, 2.), Y_t



# TODO: Double Q Learning
# probably should take in 2 QLearning objects
# ... could also take in 2 ScalarModel objects


if __name__ == "__main__":
    class FakeModel(ScalarModel):
        def __init__(self):
            super(FakeModel, self).__init__()

        def eval(self, action_t: tf.Tensor, state: List[tf.Tensor]):
            return tf.cast(tf.range(tf.shape(action_t)[0]), tf.float32)


    state = [tf.ones([3, 5])]
    QL = QLearning(FakeModel(), 4, 0.95)
    action_t, tile_states = QL._build_action_probes(state)
    assert tf.math.reduce_all(tf.shape(action_t) ==
                              tf.constant([4, 4], dtype=tf.int32))
    assert tf.math.reduce_all(tf.shape(tile_states[0]) ==
                              tf.constant([4, 3, 5], dtype=tf.int32))
    max_action = QL.select_action(state)
    assert tf.math.reduce_all(max_action == tf.constant(3, dtype=tf.int64))

    # testing q error
    action_t = tf.constant([[1, 0, 0, 0],
                            [0, 1, 0, 0],  # good action
                            [0, 0, 1, 0]], dtype=tf.float32)
    reward_t1 = tf.constant([0, 1, 0], dtype=tf.float32)
    Q_err, Y_t = QL.call(reward_t1, action_t, state, state)
    assert tf.argmax(Q_err).numpy() == 1
