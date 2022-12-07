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


class RunIface:
    # handles interfacing with simulation
    # build other models on top of this

    def __init__(self, action_model: Layer,
                 num_actions: int, rand_act_prob: float,
                 rng: npr.Generator):
        self.action_model = action_model
        self.num_actions = num_actions
        self.rand_act_prob = rand_act_prob
        self.rng = rng

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
        # TODO: not sure which model should be used here?!
        scores = self.action_model(action_t, state_t)

        if debug:
            print('action; state; scores')
            print(action_t)
            print(state_t)
            print(scores)
            print(tf.argmax(scores).numpy())

        # greedy
        return tf.argmax(scores).numpy()


class QAgent(Agent):
    # double DQN

    def __init__(self,
                 run_iface: RunIface,
                 free_model: Layer,
                 memory_model: Layer,
                 rng: npr.Generator,
                 num_actions: int,
                 state_dims: int,
                 gamma: float = 0.7,
                 tau: float = 0.01,
                 batch_size: int = 128,
                 num_batch_sample: int = 8,
                 train_epoch: int = 1):
        # TODO: eval_model and selection_model must be the same
        # underlying model (with different weights)
        # TODO/FIX: take in builder instead
        """
        Core Bellman Eqn:
            Q_f = free model
            Q_m = memory model == approximates Q table
            Q_m <- (1 - tau) * Q_m + tau * Q_f
        Learning Q_f:
            Find Q_f such that:
                Q_f(s_t, a_t) approx= r_t + gamma * max_a' [ Q_m(s_{t+1}, a') ]
                    where a' is chosen according to Q_f (double learning)

        Args:
            run_iface (RunIface): interface that implements the
                run strategy
            free/memory model (Layer):
                scalar_models
                keras layers with the following call signature
                    call(action_t: tf.Tensor, state_t: List[tf.Tensor])
                        --> tf.Tensor (with shape = batch_size)
            rng (npr.Generator):
            num_actions (int): number of actions available to
                the agent
            state_dims (int): number of dimensions in state
                assumes state can be easily represented by
                single tensor
            gamma (float): discount factor
            tau (float): update rate (often referred to as alpha in literature)
                after training eval, eval weights are copied to selection
                where update follows selection <- tau * eval + (1 - tau) * selection
            batch_size (int):
            num_batch_sample (int):
                number of batches to sample for a given training step
            train_epoch (int):
        """
        super(QAgent, self).__init__()
        self.run_iface = run_iface
        self.free_model = free_model
        self.memory_model = memory_model
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.num_batch_sample = num_batch_sample
        self.train_epoch = train_epoch
        self.rng = rng

        inputs = [tf.keras.Input(shape=(num_actions,),
                                 name="action", dtype=tf.float32),
                  tf.keras.Input(shape=(),
                                 name="reward", dtype=tf.float32),
                  tf.keras.Input(shape=(state_dims,),
                                 name="state", dtype=tf.float32),
                  tf.keras.Input(shape=(state_dims,),
                                 name="state_t1", dtype=tf.float32),
                  tf.keras.Input(shape=(),
                                 name="termination", dtype=tf.float32)]
        # need to duplicate losses and models to be able to switch
        Q_err, _ = calc_q_error_sm(self.free_model,
                                   self.free_model,
                                   self.memory_model,
                                   inputs[0], inputs[1],
                                   [inputs[2]], [inputs[3]],
                                   inputs[4],
                                   self.num_actions, self.gamma)       
        self.kmodel = CustomModel("loss",
                                  inputs=inputs,
                                  outputs={"loss": tf.math.reduce_mean(Q_err)})
        self.kmodel.compile(tf.keras.optimizers.Adam(.001))

    def init_action(self):
        return self.run_iface.init_action()

    def select_action(self, state: List[np.ndarray], debug: bool = False):
        """greedy action selection

        Args:
            state (List[np.ndarray]): set of unbatched input tensors
                each with shape:
                    ...

        Returns:
            int: index of selected action
        """
        return self.run_iface.select_action(state, debug=debug)

    def _copy_model(self, debug: bool = False):
        # copy weights from free_model to memory_model
        #   approximation of updating the Q table
        # according to: memory <- tau * free + (1 - tau) * memory
        #   NOTE: tau should probably be pretty small
        free_weights = self.free_model.get_weights()
        mem_weights = self.memory_model.get_weights()
        new_weights = []
        diffs = []
        for mem, fr in zip(mem_weights, free_weights):
            new_weights.append(self.tau * fr + (1. - self.tau) * mem)
            if debug:
                diffs.append(np.sum((mem - fr)**2.))
        if debug:
            print(np.sum(diffs))
        self.memory_model.set_weights(new_weights)

    def _draw_sample(self, run_data: RunData):
        inds = self.rng.integers(0, np.shape(run_data.actions)[0],
                                 self.num_batch_sample * self.batch_size)
        d = {"state": run_data.states[inds],
             "state_t1": run_data.states_t1[inds],
             "action": run_data.actions[inds],
             "reward": run_data.rewards[inds],
             "termination": run_data.termination[inds]}
        return tf.data.Dataset.from_tensor_slices(d)

    def train(self, run_data: RunData,
              debug: bool = False):
        """train agent on run data

        Args:
            run_data (RunData):
        """
        dset = self._draw_sample(run_data)
        history = self.kmodel.fit(dset.batch(self.batch_size),
                                  epochs=self.train_epoch)
        return history
