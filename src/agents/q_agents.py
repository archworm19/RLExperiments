"""Agents that use Q Learning

    from frameworks.q_learning:
        scalar_model assumption:
                call has the following signature:
                    call(action_t: tf.Tensor, state_t: List[tf.Tensor])
"""
import numpy as np
import numpy.random as npr
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
                 gamma: float = 0.95,
                 rand_act_prob: float = 0.25):
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
        self.rand_act_prob = rand_act_prob
        self.gamma = gamma
        self.rng = npr.default_rng(42)

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
        Qerr, _ = QLearning(gamma).calc_error(num_actions,
                                              scalar_model,
                                              inputs[1],
                                              inputs[0],
                                              [inputs[2]],
                                              [inputs[3]])
        outputs = {"loss": tf.math.reduce_mean(Qerr)}
        self.kmodel = CustomModel("loss",
                                  inputs=inputs,
                                  outputs=outputs)
        self.kmodel.compile(optimizer=tf.keras.optimizers.Adam(.001))

    def select_action(self, state: List[np.ndarray]):
        # TODO: missing randomness!!!
        """greedy action selection

        Args:
            state (List[np.ndarray]): set of unbatched input tensors
                each with shape:
                    ...

        Returns:
            int: index of selected action
        """
        if self.rng.random() < self.rand_act_prob:
            return self.rng.integers(0, self.num_actions)
        # --> action_t = num_actions x num_actions
        # --> state_t = num_actions x ...
        action_t, state_t = build_action_probes(state, self.num_actions)
        # --> shape = num_actions
        scores = self.scalar_model(action_t, state_t)
        # greedy
        return tf.argmax(scores).numpy()

    def _build_dset(self, run_data: RunData):
        # NOTE: keys must match keras input names
        dmap = {"action": run_data.actions,
                "reward": run_data.rewards,
                "state": run_data.states,
                "state_t1": run_data.states_t1}
        dset = tf.data.Dataset.from_tensor_slices(dmap)
        return dset.shuffle(25000).batch(32)

    def train(self, run_data: RunData, num_epoch: int):
        """train agent on run data

        Args:
            run_data (RunData):
            num_epoch (int):
        """
        dset = self._build_dset(run_data)
        history = self.kmodel.fit(dset,
                                  epochs=num_epoch)
        return history
