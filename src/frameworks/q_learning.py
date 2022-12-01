"""Deep Q Learning Frameworks

    scalar_model assumption:
        call has the following signature:
            call(action_t: tf.Tensor, state_t: List[tf.Tensor])
"""
import tensorflow as tf
from typing import List
from tensorflow.keras.layers import Layer


class QLearning():

    def __init__(self, gamma: float):
        # NOTE: currently assumes discrete action space
        assert gamma > 0. and gamma <= 1.
        super(QLearning, self).__init__()
        self.gamma = gamma

    def calc_error(self,
                   num_actions: int,
                   scalar_model: Layer,
                   reward: tf.Tensor,
                   action_t: tf.Tensor,
                   state_t: List[tf.Tensor],
                   state_t1: List[tf.Tensor]):
        """Q error

            Q error = (Y_t - Q(S_t, A_t; sigma_t))^2
                where Y_t = R_{t+1} + gamma * max_a[ Q(S_{t + 1}, a; sigma_t^-1 ]

        Args:
            num_actions (int): number of possible actions
            scalar_model (ScalarModel): model that computes Q values
                = max expected reward
                must output scalars
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
        for i in range(num_actions):
            batch_size = tf.shape(action_t)[0]
            action_t1 = tf.tile(tf.expand_dims(tf.one_hot(i, num_actions), 0),
                                [batch_size, 1])
            # Q(S_{t+1}, a_i)
            # --> shape = batch_size
            scores.append(scalar_model(action_t1, state_t1))
        max_score = tf.math.reduce_max(tf.stack(scores, axis=0), axis=0)
        # bring it all together
        Y_t = reward + self.gamma * max_score
        # Q(S_t, A_t)
        Q_t = scalar_model(action_t, state_t)
        return tf.math.pow(Y_t - Q_t, 2.), Y_t



if __name__ == "__main__":
    class FakeModel(Layer):
        def __init__(self):
            super(FakeModel, self).__init__()

        def call(self, action_t: tf.Tensor, state: List[tf.Tensor]):
            return tf.cast(tf.range(tf.shape(action_t)[0]), tf.float32)

    state = [tf.ones([3, 5])]

    # testing q error
    action_t = tf.constant([[1, 0, 0, 0],
                            [0, 1, 0, 0],  # good action
                            [0, 0, 1, 0]], dtype=tf.float32)
    reward_t1 = tf.constant([0, 1, 0], dtype=tf.float32)
    Q_err, Y_t = QLearning(0.95).calc_error(tf.shape(action_t)[1],
                                           FakeModel(),
                                           reward_t1, action_t,
                                           state, state)
    # sample 1 has the largest error because
    #   1. the Q values are identical for all samples
    #   2. sample 1 has a reward -->
    #       thus, Q_t != Q_{t_1}
    #           (expected reward is different cuz of immediate reward)
    assert tf.argmax(Q_err).numpy() == 1
