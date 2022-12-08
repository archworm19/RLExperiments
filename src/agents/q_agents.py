"""Agents that use Q Learning

    from frameworks.q_learning:
        scalar_model assumption:
                call has the following signature:
                    call(action_t: tf.Tensor, state_t: List[tf.Tensor])
"""
import numpy as np
import numpy.random as npr
import tensorflow as tf
from typing import List, Union
from tensorflow.keras.layers import Layer
from frameworks.agent import Agent
from frameworks.q_learning import calc_q_error_sm
from frameworks.custom_model import CustomModel
from agents.utils import build_action_probes


def _one_hot(x: np.ndarray, num_action: int):
    # array of indices --> num_sample x num_action one-hot array
    x_oh = np.zeros((np.shape(x)[0], num_action))
    x_oh[np.arange(np.shape(x)[0]), x] = 1.
    return x_oh


class RunIface:
    # handles interfacing with simulation
    # build other models on top of this
    # factored out of agent to allow for different strategies
    #       ... might not be worth it

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
        scores = self.action_model(action_t, state_t)

        if debug:
            print('action; state; scores')
            print(action_t)
            print(state_t)
            print(scores)
            print(tf.argmax(scores).numpy())

        # greedy
        return tf.argmax(scores).numpy()


class MemoryBuffer:

    def __init__(self, buffer_size: int, rng: npr.Generator):
        self.state_t = []
        self.state_t1 = []
        self.action = []
        self.reward = []
        self.termination = []
        self.buffer_size = buffer_size
        self.rng = rng

    def append(self, state_t: List[float], state_t1: List[float],
               action: int, reward: float, term: bool):
        self.state_t.append(state_t)
        self.state_t1.append(state_t1)
        self.action.append(action)
        self.reward.append(reward)
        self.termination.append(term)
        if len(self.state_t) > self.buffer_size:
            self.state_t = self.state_t[-self.buffer_size:]
            self.state_t1 = self.state_t1[-self.buffer_size:]
            self.action = self.action[-self.buffer_size:]
            self.reward = self.reward[-self.buffer_size:]
            self.termination = self.termination[-self.buffer_size:]

    def pull_sample(self, num_sample: int):
        # returns Tuple[List]
        inds = self.rng.integers(0, len(self.state_t), num_sample)
        state_t = [self.state_t[z] for z in inds]
        state_t1 = [self.state_t1[z] for z in inds]
        action = [self.action[z] for z in inds]
        reward = [self.reward[z] for z in inds]
        termination = [self.termination[z] for z in inds]
        return state_t, state_t1, action, reward, termination


class QAgent(Agent):
    # double DQN

    def __init__(self,
                 run_iface: RunIface,
                 mem_buffer: MemoryBuffer,
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
            mem_buffer (MemoryBuffer):
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
        self.mem_buffer = mem_buffer
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

        # align the models
        tau_hold = self.tau
        self.tau = 1.
        self._copy_model()
        self.tau = tau_hold

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

    def _draw_sample(self):
        (state_t, state_t1, action, reward, term) = self.mem_buffer.pull_sample(self.num_batch_sample
                                                                                * self.batch_size)
        d = {"state": np.array(state_t),
             "state_t1": np.array(state_t1),
             "action": _one_hot(np.array(action, dtype=np.int32), self.num_actions),
             "reward": np.array(reward),
             "termination": np.array(term) * 1.}
        return tf.data.Dataset.from_tensor_slices(d)

    def train(self, debug: bool = False):
        """train agent on run data

        Args:
            run_data (RunData):
        """
        dset = self._draw_sample()
        history = self.kmodel.fit(dset.batch(self.batch_size),
                                  epochs=self.train_epoch)
        return history

    def save_data(self,
                  state: List[List[float]],
                  state_t1: List[List[float]],
                  action: Union[int, float, List],
                  reward: float,
                  termination: bool):
        # NOTE: only saves a single step
        self.mem_buffer.append(state, state_t1, action, reward, termination)
