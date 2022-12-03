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
from frameworks.q_learning import calc_q_error_sm
from frameworks.custom_model import CustomModel
from agents.utils import build_action_probes


class QAgent(Agent):
    # double DQN
    # stateful: switches active model at the end of each train call

    def __init__(self,
                 model0: Layer,
                 model1: Layer,
                 num_actions: int,
                 state_dims: int,
                 gamma: float = 0.95,
                 rand_act_prob: float = 0.25):
        """
        Args:
            model0 (Layer):
            model1 (Layer):
                scalar_models
                keras layers with the following call signature
                    call(action_t: tf.Tensor, state_t: List[tf.Tensor])
                        --> tf.Tensor (with shape = batch_size)
            num_actions (int): number of actions available to
                the agent
            state_dims (int): number of dimensions in state
                assumes state can be easily represented by
                single tensor
            gamma (float): discount factor
        """
        super(QAgent, self).__init__()
        self.model0 = model0
        self.model1 = model1
        self.num_actions = num_actions
        self.rand_act_prob = rand_act_prob
        self.gamma = gamma
        self.rng = npr.default_rng(42)
        self.active_model_idx = 0
        self.modelz = [self.model0, self.model1]

        inputs = [tf.keras.Input(shape=(num_actions,),
                                 name="action", dtype=tf.float32),
                  tf.keras.Input(shape=(),
                                 name="reward", dtype=tf.float32),
                  tf.keras.Input(shape=(state_dims,),
                                 name="state", dtype=tf.float32),
                  tf.keras.Input(shape=(state_dims,),
                                 name="state_t1", dtype=tf.float32),]
        # need to duplicate losses and models to be able to switch
        Q_err0, _ = calc_q_error_sm(self.model0, self.model1,
                                    inputs[0], inputs[1], inputs[2], inputs[2],
                                    self.num_actions, self.gamma)
        Q_err1, _ = calc_q_error_sm(self.model1, self.model0,
                                    inputs[0], inputs[1], inputs[2], inputs[2],
                                    self.num_actions, self.gamma)        
        self.kmodel0 = CustomModel("loss",
                                  inputs=inputs,
                                  outputs={"loss": tf.math.reduce_mean(Q_err0)})
        self.kmodel0.compile(tf.keras.optimizers.RMSprop(.01))
        self.kmodel1 = CustomModel("loss",
                                  inputs=inputs,
                                  outputs={"loss": tf.math.reduce_mean(Q_err1)})
        self.kmodel1.compile(tf.keras.optimizers.RMSprop(.01))
        self.kmodelz = [self.kmodel0, self.kmodel1]


    def init_action(self):
        return self.rng.integers(0, self.num_actions)

    def select_action(self, state: List[np.ndarray], debug: bool = False):
        """greedy action selection

        Args:
            state (List[np.ndarray]): set of unbatched input tensors
                each with shape:
                    ...

        Returns:
            int: index of selected action
        """
        if self.rng.random() < self.rand_act_prob:
            if debug:
                print("rand select")
            return self.rng.integers(0, self.num_actions)
        # --> action_t = num_actions x num_actions
        # --> state_t = num_actions x ...
        action_t, state_t = build_action_probes(state, self.num_actions)
        # --> shape = num_actions
        scores = self.modelz[self.active_model_idx](action_t, state_t)

        if debug:
            print('action; state; scores')
            print(action_t)
            print(state_t)
            print(scores)
            print(tf.argmax(scores).numpy())

        # greedy
        return tf.argmax(scores).numpy()

    def _build_dset(self, run_data: RunData):
        # NOTE: keys must match keras input names
        dmap = {"action": run_data.actions,
                "reward": run_data.rewards,
                "state": run_data.states,
                "state_t1": run_data.states_t1}
        dset = tf.data.Dataset.from_tensor_slices(dmap)
        return dset.shuffle(1000000).batch(128)

    def train(self, run_data: RunData, num_epoch: int, debug: bool = False):
        """train agent on run data

        Args:
            run_data (RunData):
            num_epoch (int):
        """
        dset = self._build_dset(run_data)
        history = self.kmodelz[self.active_model_idx].fit(dset,
                                                          epochs=num_epoch)
        # stateful
        self.active_model_idx = 1 - self.active_model_idx
        return history
