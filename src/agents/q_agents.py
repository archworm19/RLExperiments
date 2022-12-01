"""Agents that use Q Learning

    from frameworks.q_learning:
        scalar_model assumption:
                call has the following signature:
                    call(action_t: tf.Tensor, state_t: List[tf.Tensor])
"""
import numpy as np
import tensorflow as tf
from typing import List
from tensorflow.keras.layers import Layer
from frameworks.agent import Agent, RunData
from frameworks.q_learning import QLearning
from frameworks.custom_model import CustomModel
from agents.utils import build_action_probes


class QAgent(Agent):

    def __init__(self,
                 scalar_model: Layer,
                 num_actions: int,
                 state_dims: int,
                 gamma: float = 0.95):
        """
        Args:
            scalar_model (Layer): keras layer where the call
                has the following signature:
                    call(action_t: )
            num_actions (int): number of actions available to
                the agent
            state_dims (int): number of dimensions in state
                assumes state can be easily represented by
                single tensor
            gamma (float): discount factor
        """
        super(QAgent, self).__init__()
        self.scalar_model = scalar_model
        self.num_actions = num_actions

        # TODO: is this really the right place
        # to build the custom model?
        inputs = [tf.keras.Input(shape=(num_actions,),
                                 name="action", dtype=tf.float32),
                  tf.keras.Input(shape=(),
                                 name="reward", dtype=tf.float32),
                  tf.keras.Input(shape=(state_dims,),
                                 name="state", dtype=tf.float32),
                  tf.keras.Input(shape=(state_dims,),
                                 name="state_t1", dtype=tf.float32)]
        outputs = {"loss": QLearning(gamma).calc_error(num_actions,
                                                       scalar_model,
                                                       inputs[1],
                                                       inputs[0],
                                                       inputs[2],
                                                       inputs[3])}
        self.kmodel = CustomModel("loss",
                                  inputs=inputs,
                                  outputs=outputs)
        self.kmodel.compile(optimizer=tf.keras.optimizers.Adam(.01))

    def select_action(self, state: List[np.ndarray]):
        """greedy action selection

        Args:
            state (List[np.ndarray]): set of unbatched input tensors
                each with shape:
                    ...

        Returns:
            int: index of selected action
        """
        # --> action_t = num_actions x num_actions
        # --> state_t = num_actions x ...
        action_t, state_t = build_action_probes(state, self.num_actions)
        # --> shape = num_actions
        scores = self.scalar_model(action_t, state_t)
        # greedy
        return tf.argmax(scores).numpy()

    def train(self, run_data: RunData, num_epoch: int):
        """train agent on run data

        Args:
            run_data (RunData):
            num_epoch (int):
        """
        # TODO: package into tf.dataset?
        # NOTE: keys must match keras input names
        dmap = {"action": run_data.actions,
                "reward": run_data.rewards,
                "state": run_data.states,
                "state_t1": run_data.states_t1}
        dset = tf.data.Dataset.from_tensor_slices(dmap)
        history = self.kmodel.fit(dset.shuffle(5000).batch(32))
        return history


if __name__ == "__main__":
    # test on fake dataset:
    #   sum[state] > 0 and action = 1 --> reward
    #   else --> no reward
    from numpy.random import npr
    rng = npr.default_rng(42)
    states = rng.random((100, 2)) - 0.5
    action0 = (rng.random((100,)) > 0.5) * 1
    actions = np.vstack((action0, 1. - action0)).T
    rewards = (1. * (np.sum(states, axis=1) > 0.)) * action0
    dat = RunData([states[:-1]], [states[1:]], actions[:-1], rewards[:-1])
