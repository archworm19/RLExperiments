"""Agents that use Q Learning

    from frameworks.q_learning:
        scalar_model assumption:
                call has the following signature:
                    call(action_t: tf.Tensor, state_t: List[tf.Tensor])
"""
import numpy as np
import numpy.random as npr
import tensorflow as tf
from typing import List, Union, Callable, Tuple, Dict
from tensorflow.keras.layers import Layer
from frameworks.agent import Agent, TrainOnLine
from frameworks.q_learning import (calc_q_error_sm, calc_q_error_critic, calc_q_error_actor,
                                   calc_q_error_distro_discrete, _calc_q_from_distro)
from frameworks.custom_model import CustomModel
from frameworks.layer_signatures import ScalarModel, ScalarStateModel, DistroModel
from agents.utils import build_action_probes
from replay_buffers.replay_buffs import MemoryBuffer


# utils


def _one_hot(x: np.ndarray, num_action: int):
    # array of indices --> num_sample x num_action one-hot array
    x_oh = np.zeros((np.shape(x)[0], num_action))
    x_oh[np.arange(np.shape(x)[0]), x] = 1.
    return x_oh


def copy_model(send_model: Layer, rec_model: Layer,
               tau: float):
    # copy weights from send_model to rec_model
    # rec_model <- (1 - tau) * rec_model + tau * send_model
    send_weights = send_model.get_weights()
    rec_weights = rec_model.get_weights()
    new_weights = []
    for send, rec in zip(send_weights, rec_weights):
        new_weights.append(tau * send + (1 - tau) * rec)
    rec_model.set_weights(new_weights)


# Run Interfaces


class RunIface:
    # epsilon-greedy run interface ~ anneal random act prob
    # handles interfacing with simulation
    # build other models on top of this
    # factored out of agent to allow for different strategies
    #       ... might not be worth it

    def __init__(self,
                 num_actions: int,
                 rng: npr.Generator,
                 min_rand_act_prob: float = 0.05,
                 T_rand_act_prob: float = 5e3):
        # anneal rand_act_prob from 1 to min_rand_act_prob
        #   in T_rand_act_prob action selection steps
        self.num_actions = num_actions
        self.rand_act_prob = 1.
        # min_rand_act_prob = mrap = x^(T)
        #   T log x = log(mrap) --> x = exp(log(mrap) / T)
        self.rand_act_decay = np.exp(np.log(min_rand_act_prob) / T_rand_act_prob)
        self.min_rand_act_prob = min_rand_act_prob
        self.rng = rng

    def init_action(self):
        return self.rng.integers(0, self.num_actions)

    def select_action(self, model: ScalarModel, state: List[np.ndarray],
                      test_mode: bool = False, debug: bool = False):
        """epsilon greedy action selection
            r = random float
            if r < rand_act_prob --> random action select
            else --> greedy choice

        Args:
            model (ScalarModel): model that outputs Q evaluations for state-action
                pairs
            state (List[np.ndarray]): set of unbatched input tensors
                each with shape:
                    ...

        Returns:
            int: index of selected action
        """
        if not test_mode:
            self.rand_act_prob = max(self.min_rand_act_prob,
                                     self.rand_act_decay * self.rand_act_prob)

        if self.rng.random() < self.rand_act_prob:
            if debug:
                print("rand select")
            return self.rng.integers(0, self.num_actions)
        # --> action_t = num_actions x num_actions
        # --> state_t = num_actions x ...
        action_t, state_t = build_action_probes(state, self.num_actions)
        # --> shape = num_actions
        scores = model(action_t, state_t)

        if debug:
            print('action; state; scores')
            print(action_t)
            print(state_t)
            print(scores)
            print(tf.argmax(scores).numpy())

        # greedy
        return tf.argmax(scores).numpy()


class RunIfaceCont:
    # continuous-domain run interface
    # pi_model: pi(a | s)
    # uses Ornstein-Uhlenbeck process for correlated noise

    def __init__(self,
                 bounds: List[Tuple],
                 theta: List[float],
                 sigma: List[float],
                 rng: npr.Generator):
        # bounds = list of (lower_bound, upper_bound) pairs
        #       for each action dim
        # theta: momentum term for each action dim
        # sigma: variance term for each action dim
        assert len(bounds) == len(theta)
        assert len(bounds) == len(sigma)
        self.bounds = np.array(bounds)
        self.theta = np.array(theta)
        self.sigma = np.array(sigma)
        self.mu = np.zeros((len(bounds)))
        self.cov = np.eye(len(bounds))
        self.rng = rng

        # initialize noise at 0
        self.x_noise = np.zeros((len(self.bounds)))

    def _noisify_and_clip(self, x):
        # update noise:
        self.x_noise += (self.sigma * np.sqrt(2. * self.theta) *
                         self.rng.multivariate_normal(self.mu, self.cov) -
                         self.theta * self.x_noise)
        return np.clip(x + self.x_noise, self.bounds[:,0], self.bounds[:,1])

    def init_action(self):
        # returns: List[float]
        # reset noise
        self.x_noise = np.zeros((len(self.bounds)))
        mid_pt = np.mean(self.bounds, axis=1)
        return self._noisify_and_clip(mid_pt).tolist()

    def select_action(self, model: ScalarStateModel, state: List[np.ndarray],
                      debug: bool = False, test_mode: bool = False):
        """
        Args:
            model (ScalarStateModel): action selection model
            state (List[np.ndarray]): set of unbatched input tensors
                each with shape:
                    ...

        Returns:
            List[float]
        """
        # NOTE: need to expand dims cuz unbatched
        state = [tf.expand_dims(si, 0) for si in state]
        # returns batch_size (1 here) x action_dims
        a = model(state)[0].numpy()
        a_noise = self._noisify_and_clip(a).tolist()
        if debug:
            print(self.x_noise)
            print(a)
            print(a_noise)
        return a_noise


# Agents


class QAgent(Agent, TrainOnLine):
    # double DQN

    def __init__(self,
                 run_iface: RunIface,
                 model_builder: Callable[[], ScalarModel],
                 rng: npr.Generator,
                 num_actions: int,
                 state_dims: Dict[str, Tuple[int]],
                 gamma: float = 0.7,
                 tau: float = 0.01,
                 batch_size: int = 128,
                 num_batch_sample: int = 8,
                 train_epoch: int = 1):
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
            state_dims (Dict[str, Tuple[int]]): mapping of state names
                to tensor shapes
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
        self.free_model = model_builder()
        self.memory_model = model_builder()
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
                  tf.keras.Input(shape=(),
                                 name="termination", dtype=tf.float32)]
        s0_names = [k for k in state_dims]
        s0_inputs = [tf.keras.Input(shape=state_dims[k], dtype=tf.float32, name=k)
                     for k in s0_names]
        s1_inputs = [tf.keras.Input(shape=state_dims[k], dtype=tf.float32, name=k+"t1")
                     for k in s0_names]
        inputs = inputs + s0_inputs + s1_inputs
        self.s0_names = s0_names
        self.s1_names = [inp.name for inp in s1_inputs]

        self.mem_buffer = MemoryBuffer(["action", "reward",
                                        "termination"] + self.s0_names + self.s1_names,
                                        rng,
                                        500000)
        # loss ~ discrete Q error framework
        Q_err, _ = calc_q_error_sm(self.free_model,
                                   self.free_model,
                                   self.memory_model,
                                   inputs[0], inputs[1],
                                   s0_inputs, s1_inputs,
                                   inputs[2],
                                   self.num_actions, self.gamma,
                                   huber=False)       
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
        ind = self.run_iface.init_action()
        v = np.zeros((self.num_actions,))
        v[ind] = 1.
        return v

    def select_action(self, state: Dict[str, np.ndarray], test_mode: bool, debug: bool) -> np.ndarray:
        """select 

        Args:
            state (Dict[str, np.ndarray]): mapping of state names to batched tensors
                each tensor = num_sample x ...
            test_mode (bool): are we in a 'test run' for the agent?
            debug (bool): debug mode

        Returns:
            np.ndarray: selected actions
                shape = num_sample x action_dims
        """
        # unpack states in correct order:
        state_upack = [state[k] for k in self.s0_names]
        return self.run_iface.select_action(self.memory_model, state_upack,
                                            test_mode=test_mode, debug=debug)

    def _copy_model(self, debug: bool = False):
        # copy weights from free_model to memory_model
        #   approximation of updating the Q table
        # according to: memory <- tau * free + (1 - tau) * memory
        #   NOTE: tau should probably be pretty small
        copy_model(self.free_model, self.memory_model, self.tau)

    def _draw_sample(self):
        d = self.mem_buffer.pull_sample(self.num_batch_sample * self.batch_size)
        d = {k: np.array(d[k]) for k in d}
        return tf.data.Dataset.from_tensor_slices(d)

    def train(self, debug: bool = False):
        """train agent on run data

        Args:
            run_data (RunData):
        """
        dset = self._draw_sample()
        history = self.kmodel.fit(dset.batch(self.batch_size),
                                  epochs=self.train_epoch,
                                  verbose=0)
        return history

    def save_data(self,
                  state: Dict[str, np.ndarray],
                  state_t1: Dict[str, np.ndarray],
                  action: np.ndarray,
                  reward: np.ndarray,
                  termination: np.ndarray) -> None:
        """send data to the model --> model will use
        it for training later

        Args:
            state (Dict[str, np.ndarray]): state(t)
                mapping from state name to num_samples x ...
            state_t1 (Dict[str, np.ndarray]): state(t + 1)
                mapping from state name to num_samples x ...
            action (np.ndarray): action(t)
                shape = num_sample x action_dims
            reward (np.ndarray): reward(t)
                shape = num_sample
            termination (np.ndarray): termination(t)
                shape = num_sample

        Returns:
            None
        """
        for z in range(len(reward)):
            d = {"action": action[z],
                "reward": reward[z],
                "termination": termination[z]}
            for k in self.s0_names:
                d[k] = state[k][z]
                d[k+"t1"] = state_t1[k][z]
            self.mem_buffer.append(d)

    def end_epoch(self):
        pass


class ExpModel(ScalarModel):
    # expectation model:
    # > wraps Distro Model
    # > Distro Model outputs unnormalized probabilities
    # > use probabilities + Vmin + Vmax to produce expected q value
    #       = expectation across distro

    def __init__(self, Vmin: float, Vmax: float, distro_model: DistroModel,
                 r_std_dev: float = 0.0):
        # r_std_dev: standard deviation of random value added to expectation score
        #       used to break ties in random fashion
        super(ExpModel, self).__init__()
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.model = distro_model
        self.r_std_dev = r_std_dev

    def call(self, action_t: tf.Tensor, state_t: List[tf.Tensor]) -> tf.Tensor:
        """
        Args:
            action_t (tf.Tensor): action
                batch_size x ...
            state_t (List[tf.Tensor]): set of states
                each tensor is:
                    batch_size x ...

        Returns:
            tf.Tensor: shape = batch_size
        """
        # --> batch_size x dim
        d = self.model(action_t, state_t)
        p = tf.nn.softmax(d, axis=1)
        q = _calc_q_from_distro(self.Vmin, self.Vmax, p)
        return q + tf.random.normal(tf.shape(q), stddev=self.r_std_dev)


class QAgent_distro(Agent):
    # distributional approach
    # discrete control

    def __init__(self,
                 run_iface: RunIface,
                 model_builder: Callable[[], DistroModel],
                 rng: npr.Generator,
                 num_actions: int,
                 state_dims: Dict[str, Tuple[int]],
                 vector0: tf.Tensor,
                 gamma: float = 0.95,
                 tau: float = 0.01,
                 Vmin: float = -20.,
                 Vmax: float = 20.,
                 batch_size: int = 128,
                 num_batch_sample: int = 1,
                 train_epoch: int = 1,
                 learning_rate: float = .001,
                 mem_buffer_size: int = 500000):
        # see QAgent docstring
        # TODO: additional fields: Vmin, Vmax, vector0
        super(QAgent_distro, self).__init__()
        self.run_iface = run_iface
        self.free_model = model_builder()
        self.memory_model = model_builder()
        self.exp_model = ExpModel(Vmin, Vmax, self.memory_model,
                                  r_std_dev=0.0)
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.num_batch_sample = num_batch_sample
        self.train_epoch = train_epoch
        self.rng = rng

        # multi-state system
        inputs = [tf.keras.Input(shape=(num_actions,),
                                 name="action", dtype=tf.float32),
                  tf.keras.Input(shape=(),
                                 name="reward", dtype=tf.float32),
                  tf.keras.Input(shape=(),
                                 name="termination", dtype=tf.float32)]
        s0_names = [k for k in state_dims]
        s0_inputs = [tf.keras.Input(shape=state_dims[k], dtype=tf.float32, name=k)
                     for k in s0_names]
        s1_inputs = [tf.keras.Input(shape=state_dims[k], dtype=tf.float32, name=k+"t1")
                     for k in s0_names]
        inputs = inputs + s0_inputs + s1_inputs
        self.s0_names = s0_names
        self.s1_names = [inp.name for inp in s1_inputs]

        # loss ~ discrete Q error framework
        Q_err, weights, act_vect = calc_q_error_distro_discrete(self.free_model, self.exp_model, self.memory_model,
                                                      Vmin, Vmax,
                                                      inputs[0], inputs[1],
                                                      s0_inputs,
                                                      s1_inputs,
                                                      inputs[2],
                                                      self.num_actions, self.gamma,
                                                      vector0)
        self.kmodel = CustomModel("loss",
                                  inputs=inputs,
                                  outputs={"loss": tf.math.reduce_mean(Q_err),
                                           "full_loss": Q_err,  # TODO: TESTING
                                           "weights": weights,  # TODO: TESTING
                                           "action_vector": act_vect})  # TODO: TESTING
        self.kmodel.compile(tf.keras.optimizers.Adam(learning_rate))
        # align the models
        tau_hold = self.tau
        self.tau = 1.
        self._copy_model()
        self.tau = tau_hold
        # simple memory buffer
        self.mem_buffer = MemoryBuffer((["action", "reward", "termination"]
                                            + self.s0_names + self.s1_names),
                                        rng,
                                        mem_buffer_size)

    def init_action(self):
        ind = self.run_iface.init_action()
        v = np.zeros((self.num_actions,))
        v[ind] = 1.
        return v

    def select_action(self, state: Dict[str, np.ndarray], test_mode: bool, debug: bool) -> np.ndarray:
        """select 

        Args:
            state (Dict[str, np.ndarray]): mapping of state names to batched tensors
                each tensor = num_sample x ...
            test_mode (bool): are we in a 'test run' for the agent?
            debug (bool): debug mode

        Returns:
            np.ndarray: selected actions
                shape = num_sample x action_dims
        """
        # unpack states in correct order:
        state_upack = [state[k] for k in self.s0_names]
        if debug:
            print("p(atom_j | a, state_t) for all a")
            action_t, state_t = build_action_probes(state_upack, self.run_iface.num_actions)
            prob = tf.nn.softmax(self.memory_model(action_t, state_t), axis=1).numpy()
            print(np.round(prob, decimals=2))
        return self.run_iface.select_action(self.exp_model, state_upack,
                                            test_mode=test_mode, debug=debug)

    def _copy_model(self, debug: bool = False):
        # copy weights from free_model to memory_model
        #   approximation of updating the Q table
        # according to: memory <- tau * free + (1 - tau) * memory
        #   NOTE: tau should probably be pretty small
        copy_model(self.free_model, self.memory_model, self.tau)

    def _draw_sample(self):
        d = self.mem_buffer.pull_sample(self.num_batch_sample * self.batch_size)
        d = {k: np.array(d[k]) for k in d}
        return tf.data.Dataset.from_tensor_slices(d)

    def train(self, debug: bool = False):
        """train agent on run data

        Args:
            run_data (RunData):
        """
        dset = self._draw_sample()
        history = self.kmodel.fit(dset.batch(self.batch_size),
                                  epochs=self.train_epoch,
                                  verbose=0)
        return history

    def save_data(self,
                  state: Dict[str, np.ndarray],
                  state_t1: Dict[str, np.ndarray],
                  action: np.ndarray,
                  reward: np.ndarray,
                  termination: np.ndarray) -> None:
        """send data to the model --> model will use
        it for training later

        Args:
            state (Dict[str, np.ndarray]): state(t)
                mapping from state name to num_samples x ...
            state_t1 (Dict[str, np.ndarray]): state(t + 1)
                mapping from state name to num_samples x ...
            action (np.ndarray): action(t)
                shape = num_sample x action_dims
            reward (np.ndarray): reward(t)
                shape = num_sample
            termination (np.ndarray): termination(t)
                shape = num_sample

        Returns:
            None
        """
        for z in range(len(reward)):
            d = {"action": action[z],
                "reward": reward[z],
                "termination": termination[z]}
            for k in self.s0_names:
                d[k] = state[k][z]
                d[k+"t1"] = state_t1[k][z]
            self.mem_buffer.append(d)

    def end_epoch(self):
        pass



# TODO: updated through here


# Continuous Agents


class QAgent_cont(Agent):
    # basic continuous Q agent
    #   drpg paper

    def __init__(self,
                 run_iface: RunIfaceCont,
                 q_model_builder: Callable[[], ScalarModel],
                 pi_model_builder: Callable[[], ScalarStateModel],
                 rng: npr.Generator,
                 action_bounds: List[Tuple[float]],
                 state_dims: List[Tuple[int]],
                 gamma: float = 0.7,
                 tau: float = 0.01,
                 batch_size: int = 128,
                 num_batch_sample: int = 8,
                 train_epoch: int = 1,
                 critic_lr: float = .001,
                 actor_lr: float = .001):
        """
        Args:
            run_iface (RunIfaceCont): interface that implements the
                run strategy ~ continuous space
            q_model_builder (Callable): function that builds a keras
                layer with the following call signature:
                    call(action: tf.Tensor, state: List[tf.Tensor])
                expected reward approximator
            pi_model_builder (Callable): function that builds a keras
                layer with the following call signature:
                    call(state: List[tf.Tensor])
                action generator
            rng (npr.Generator):
            action_bounds (List[Tuple[int]]): (lb, ub) pairs for each
                action dimension
            state_dims (List[List[int]]): shape of each state tensor
                Ex: if we want to supply base state + pixels -->
                    it's useful to subit them as separate state tensors
            gamma (float): discount factor
            tau (float): update rate (often referred to as alpha in literature)
                after training eval, eval weights are copied to selection
                where update follows selection <- tau * eval + (1 - tau) * selection
            batch_size (int):
            num_batch_sample (int):
                number of batches to sample for a given training step
            train_epoch (int):
        """
        super(QAgent_cont, self).__init__()
        self.run_iface = run_iface
        self.q_model = q_model_builder()
        self.qprime_model = q_model_builder()
        self.pi_model = pi_model_builder()
        self.piprime_model = pi_model_builder()
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.num_batch_sample = num_batch_sample
        self.train_epoch = train_epoch
        self.rng = rng
        action_dims = len(action_bounds)

        # multi-state system
        s0_names = ["state" + str(i) for i in range(len(state_dims))]
        s1_names = ["state" + str(i) + "_t1" for i in range(len(state_dims))]
        s0_inputs = [tf.keras.Input(shape=s, dtype=tf.float32, name=n)
                     for s, n in zip(state_dims, s0_names)]
        s1_inputs = [tf.keras.Input(shape=s, dtype=tf.float32, name=n)
                     for s, n in zip(state_dims, s1_names)]
        inputs = [tf.keras.Input(shape=(action_dims,),
                                 name="action", dtype=tf.float32),
                  tf.keras.Input(shape=(),
                                 name="reward", dtype=tf.float32),
                  tf.keras.Input(shape=(),
                                 name="termination", dtype=tf.float32)]
        inputs = inputs + s0_inputs + s1_inputs
        self.s0_names = s0_names
        self.s1_names = s1_names
        # loss ~ continuous Q error framework
        Q_err, _ = calc_q_error_critic(self.q_model,
                                     self.qprime_model,
                                     self.piprime_model,
                                     inputs[0], inputs[1],
                                     s0_inputs,
                                     s1_inputs,
                                     inputs[2],
                                     self.gamma,
                                     huber=False)
        self.kmodel = CustomModel("loss",
                                  inputs=inputs,
                                  outputs={"loss": tf.math.reduce_mean(Q_err)})
        self.kmodel.compile(tf.keras.optimizers.Adam(critic_lr))
        self.actor_opt = tf.keras.optimizers.Adam(actor_lr)

        # init weights for pi model:
        _ = self.pi_model(s0_inputs)
        # align the models
        tau_hold = self.tau
        self.tau = 1.
        self._copy_model()
        self.tau = tau_hold

        # simple memory buffer
        # simple memory buffer
        self.mem_buffer = MemoryBuffer((["action", "reward", "termination"]
                                            + self.s0_names + self.s1_names),
                                        rng,
                                        500000)

    def init_action(self):
        return self.run_iface.init_action()

    def select_action(self, state: List[np.ndarray],
                      test_mode: bool = False, debug: bool = False):
        """greedy action selection

        Args:
            state (List[np.ndarray]): set of unbatched input tensors
                each with shape:
                    ...

        Returns:
            int: index of selected action
        """
        act = self.run_iface.select_action(self.piprime_model, state,
                                           test_mode=test_mode, debug=debug)
        if debug:
            print(self.qprime_model(tf.expand_dims(tf.constant(act), 0),
                                    [tf.expand_dims(si, 0) for si in state]))
        return act

    def _copy_model(self, debug: bool = False):
        copy_model(self.q_model, self.qprime_model, self.tau)
        copy_model(self.pi_model, self.piprime_model, self.tau)

    def _draw_sample(self):
        d = self.mem_buffer.pull_sample(self.num_batch_sample * self.batch_size)
        d = {k: np.array(d[k]) for k in d}
        return tf.data.Dataset.from_tensor_slices(d)

    @tf.function
    def _update_actor_step(self, state_t: List[tf.Tensor]):
        with tf.GradientTape() as tape:
            loss = calc_q_error_actor(self.q_model, self.pi_model, state_t)
            # KEY: only differentiate wrt pi model's weights
            g = tape.gradient(loss, self.pi_model.trainable_weights)
        self.actor_opt.apply_gradients(zip(g, self.pi_model.trainable_weights))

    def train(self, debug: bool = False):
        """train agent on run data

        Args:
            run_data (RunData):
        """
        dset = self._draw_sample()
        # train critic
        history = self.kmodel.fit(dset.batch(self.batch_size),
                                  epochs=self.train_epoch,
                                  verbose=0)

        # update actor
        # TODO: this is a little diff cuz paper constructin
        # is not minibatched
        for v in dset.batch(self.batch_size):
            full_state = [v[k] for k in self.s0_names]
            self._update_actor_step(full_state)

        return history

    def save_data(self,
                  state: List[List[float]],
                  state_t1: List[List[float]],
                  action: Union[int, float, List],
                  reward: float,
                  termination: bool):
        d = {"action": action,
             "reward": reward,
             "termination": termination}
        for n, v in zip(self.s0_names, state):
            d[n] = v
        for n, v in zip(self.s1_names, state_t1):
            d[n] = v
        self.mem_buffer.append(d)

    def end_epoch(self):
        self.run_iface.x_noise = self.run_iface.x_noise * 0.
        pass
