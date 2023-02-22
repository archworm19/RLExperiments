"""Proximal Policy Optimization (PPO) agents"""
import os
import numpy as np
import numpy.random as npr
from src.frameworks.exploration_forward import forward_surprisal, inverse_dynamics_error
import tensorflow as tf
from typing import List, Tuple, Callable, Dict
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense
from frameworks.layer_signatures import DistroStateModel, ScalarStateModel, VectorStateModel, VectorModel, MapperModel
from frameworks.agent import Agent, TrainEpoch, WeightMate
from frameworks.ppo import package_dataset, ppo_loss_multiclass, ppo_loss_gauss
from frameworks.custom_model import CustomModel


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


class PPODiscrete(Agent, TrainEpoch, WeightMate):

    def __init__(self,
                 pi_model_builder: Callable[[], DistroStateModel],
                 value_model_builder: Callable[[], ScalarStateModel],
                 num_actions: int,
                 state_dims: Dict[str, Tuple[int]],
                 eta: float = 0.2,
                 vf_scale: float = 1.,
                 entropy_scale: float = .01,
                 gamma: float = 0.99,
                 lam: float = 1.,
                 train_batch_size: int = 32,
                 train_epoch: int = 8,
                 learning_rate: float = .001):
        """
        Args:
            pi_model_builder (Callable[[], DistroStateModel]): actor model builder
                actor model assumed to be normalized
            value_model_builder (Callable[[], ScalarStateModel]): critic model builder
            num_actions (int): number of available actions
            state_dims (Dict[str, Tuple[int]]): mapping of state names to state dims
            eta (float, optional): ppo clipping discount. Defaults to 0.2.
            vf_scale (float, optional): critic error regularization strength. Defaults to 1.
            entropy_scale (float, optional): entropy regularization strength. Defaults to .01.
            gamma (float, optional): discount factor. Defaults to 0.99.
            lam (float, optional): generalized discount scale. Defaults to 1..
            train_batch_size (int, optional): Defaults to 32.
            train_epoch (int, optional): Defaults to 8.
            learning_rate (float, optional): Defaults to .001.
        """
        super(PPODiscrete, self).__init__()
        self.rng = npr.default_rng(42)
        self.num_actions = num_actions
        self.gamma = gamma
        self.lam = lam
        self.train_batch_size = train_batch_size
        self.train_epoch = train_epoch

        # multi-state system
        # pulling out state names ensures consistent ordering for model calls
        self.state_names = list(state_dims.keys())
        s0_inputs = [tf.keras.Input(shape=state_dims[k], dtype=tf.float32, name=k)
                     for k in self.state_names]
        inputs = [tf.keras.Input(shape=(num_actions,),  # one-hot action encoding
                                 name="action", dtype=tf.float32),
                  tf.keras.Input(shape=(),  # advantage
                                 name="adv", dtype=tf.float32),
                  tf.keras.Input(shape=(),  # target value name
                                 name="val", dtype=tf.float32)]
        inputs = inputs + s0_inputs

        # ppo loss
        pi_new = pi_model_builder()
        pi_old = pi_model_builder()
        v_model = value_model_builder()
        # these lines set state ordering (ordering set internally by state names)
        pi_new_distro, pi_new_test = pi_new(s0_inputs)
        pi_old_distro, pi_old_test = pi_old(s0_inputs)
        critic_pred, critic_test = v_model(s0_inputs)
        loss = ppo_loss_multiclass(pi_old_distro, pi_new_distro,
                                   critic_pred,
                                   inputs[0], inputs[1], inputs[2],
                                   eta, vf_scale=vf_scale, entropy_scale=entropy_scale)
        # primary model
        self.kmodel = CustomModel("loss",
                                  inputs=inputs,
                                  outputs={"loss": tf.math.reduce_mean(loss)})
        self.kmodel.compile(tf.keras.optimizers.Adam(learning_rate))
        # test model ~ just for checking layer signatures are adhered to
        self.test_model = Model(inputs=inputs,
                                outputs={"pi_new_test": pi_new_test,
                                         "pi_old_test": pi_old_test,
                                         "critic_test": critic_test})
        self.lsig_test = False
        self.pi_new = pi_new
        self.pi_old = pi_old
        self.critic = v_model
        self._sname0 = s0_inputs[0].name

    def init_action(self) -> np.ndarray:
        """Initial action agent should take

        Returns:
            np.ndarray: len = dims in action space
        """
        ind = self.rng.integers(0, self.num_actions)
        v = np.zeros((1, self.num_actions))
        v[0, ind] = 1.
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
        # use state_names to sync order
        state_ord = [state[k] for k in self.state_names]
        # --> shape = num_sample x num_actions (normalized)
        pr = self.pi_new(state_ord)[0].numpy()
        # --> shape = num_sample x 1
        r = self.rng.random(size=(np.shape(pr)[0], 1))
        if debug:
            print('probability and random draw')
            print(pr)
            print(r)
        boolz = r <= np.cumsum(pr, axis=1)
        v = boolz * 1.
        return np.hstack((v[:,:1], v[:,1:] - v[:,:-1]))

    def _calculate_v(self, states: List[Dict[str, np.ndarray]]):
        # TODO: tf.function?
        # Returns: critic evals = List[np.ndarray]
        #               array for each trajectory
        V = []
        # iter thru trajectories:
        for traj in states:
            V_traj = []
            # iter thru windows within a trajectory:
            for win_start in range(0, np.shape(traj[self._sname0])[0], 32):
                # uses self.state_names to sync ordering
                state_wins = [traj[k][win_start:win_start+32] for k in self.state_names]
                V_traj.append(self.critic(state_wins)[0].numpy())
            V.append(np.concatenate(V_traj, axis=0))
        return V

    def _test_layersigs(self, dset: tf.data.Dataset):
        for v in dset.batch(16):
            vout = self.test_model(v)
            for k in vout:
                assert vout[k].numpy()
            break
        self.lsig_test = True

    def train(self,
              states: List[Dict[str, np.ndarray]],
              reward: List[np.ndarray],
              actions: List[np.ndarray],
              terminated: List[bool]) -> Dict:
        """train agent on data trajectories
        KEY: each element of outer list for each
            argument = different trajectory/run

        Args:
            states (List[Dict[str, np.ndarray]]):
                outer list = each distinct state
                inner dict = mapping from state names to
                    (T + 1) x ... arrays
            reward (List[np.ndarray]):
                Each list is a different trajectory.
                Each ndarray has shape T x ...
            actions (List[np.ndarray]): where len of each state
                Each list is a different trajectory.
                Each ndarray has shape T x ...
            terminated (List[bool]): whether each trajectory
                was terminated or is still running

        Returns:
            Dict: loss history
        """
        # filter out short trajectories
        states2, actions2, rewards2, terms2 = [], [], [], []
        for i in range(len(actions)):
            if np.shape(actions[i])[0] > 5:
                states2.append(states[i])
                actions2.append(actions[i])
                rewards2.append(reward[i])
                terms2.append(terminated[i])
        V = self._calculate_v(states2)
        dset = package_dataset(states2, V, rewards2, actions2, terms2,
                               self.gamma, self.lam,
                               adv_name="adv", val_name="val", action_name="action")
        if not self.lsig_test:
            self._test_layersigs(dset)
        history = self.kmodel.fit(dset.batch(self.train_batch_size),
                                  epochs=self.train_epoch,
                                  verbose=0)
        # copy update actor to old actor
        copy_model(self.pi_new.layer, self.pi_old.layer, 1.)
        return history

    def save_weights(self, directory_location: str):
        # get_weights --> List[np.ndarray]
        np.savez(os.path.join(directory_location, "actor_weights.npz"), *self.pi_new.layer.get_weights())
        np.savez(os.path.join(directory_location, "critic_weights.npz"), *self.critic.layer.get_weights())

    def load_weights(self, directory_location: str):
        d_actor = np.load(os.path.join(directory_location, "actor_weights.npz"))
        actor_weights = [d_actor["arr_" + str(i)] for i in range(len(d_actor))]
        self.pi_new.layer.set_weights(actor_weights)
        self.pi_old.layer.set_weights(actor_weights)  # TODO: necessary?
        d_critic = np.load(os.path.join(directory_location, "critic_weights.npz"))
        critic_weights = [d_critic["arr_" + str(i)] for i in range(len(d_critic))]
        self.critic.layer.set_weights(critic_weights)


class PPOContinuous(Agent, TrainEpoch):

    def __init__(self,
                 pi_model_builder: Callable[[], VectorStateModel],
                 value_model_builder: Callable[[], ScalarStateModel],
                 action_bounds: List[Tuple[float]],
                 state_dims: Dict[str, Tuple[int]],
                 eta: float = 0.2,
                 vf_scale: float = 1.,
                 entropy_scale: float = .01,
                 gamma: float = 0.99,
                 lam: float = 1.,
                 train_batch_size: int = 32,
                 train_epoch: int = 8,
                 learning_rate: float = .001):
        """Continuous PPO agent
        Args:
            pi_model_builder (Callable[[], VectorStateModel]): actor model builder
                ASSUMES: actor model outputs gaussian distribution
                        = batch_size x (2 * action_dims)
                            first set of action dims = mean
                            second set = diagonal of precision matrix
            value_model_builder (Callable[[], ScalarStateModel]): critic model builder
            action_bounds (List[Tuple[float]]): (lower bound, upper bound) pairs for
                each action dimension
            state_dims (Dict[str, Tuple[int]]): mapping of state names to state dims
            eta (float, optional): ppo clipping discount. Defaults to 0.2.
            vf_scale (float, optional): critic error regularization strength. Defaults to 1.
            entropy_scale (float, optional): entropy regularization strength. Defaults to .01.
            gamma (float, optional): discount factor. Defaults to 0.99.
            lam (float, optional): generalized discount scale. Defaults to 1..
            train_batch_size (int, optional): Defaults to 32.
            train_epoch (int, optional): Defaults to 8.
            learning_rate (float, optional): Defaults to .001.
        """
        super(PPOContinuous, self).__init__()
        self.rng = npr.default_rng(42)
        self.action_bounds = action_bounds
        self.gamma = gamma
        self.lam = lam
        self.train_batch_size = train_batch_size
        self.train_epoch = train_epoch

        # multi-state system
        # pulling out state names ensures consistent ordering for model calls
        self.state_names = list(state_dims.keys())
        s0_inputs = [tf.keras.Input(shape=state_dims[k], dtype=tf.float32, name=k)
                     for k in self.state_names]
        inputs = [tf.keras.Input(shape=(len(action_bounds),),
                                 name="action", dtype=tf.float32),
                  tf.keras.Input(shape=(),  # advantage
                                 name="adv", dtype=tf.float32),
                  tf.keras.Input(shape=(),  # target value name
                                 name="val", dtype=tf.float32)]
        inputs = inputs + s0_inputs

        # ppo loss
        pi_new = pi_model_builder()
        pi_old = pi_model_builder()
        v_model = value_model_builder()
        # these lines set state ordering (ordering set internally by state names)
        pi_new_distro, pi_new_test = pi_new(s0_inputs)
        pi_old_distro, pi_old_test = pi_old(s0_inputs)
        critic_pred, critic_test = v_model(s0_inputs)
        loss, pr_ratio = ppo_loss_gauss(pi_old_distro[:, :len(action_bounds)], pi_old_distro[:, len(action_bounds):],
                              pi_new_distro[:, :len(action_bounds)], pi_new_distro[:, len(action_bounds):],
                              critic_pred,
                              inputs[0], inputs[1], inputs[2],
                              eta, vf_scale, entropy_scale)
        # primary model
        self.kmodel = CustomModel("loss",
                                  inputs=inputs,
                                  outputs={"loss": tf.math.reduce_mean(loss),
                                           "pr_ratio": pr_ratio,
                                           "new_distro": pi_new_distro,
                                           "old_distro": pi_old_distro})
        self.kmodel.compile(tf.keras.optimizers.Adam(learning_rate))
        # testing model
        self.test_model = Model(inputs=inputs,
                                outputs={"pi_new_test": pi_new_test,
                                         "pi_old_test": pi_old_test,
                                         "critic_test": critic_test})
        self.lsig_test = False
        self.pi_new = pi_new
        self.pi_old = pi_old
        self.critic = v_model
        self._sname0 = s0_inputs[0].name

    def init_action(self) -> np.ndarray:
        """Initial action agent should take

        Returns:
            np.ndarray: len = dims in action space
        """
        # uniform distro within bounds
        ab = np.array(self.action_bounds)
        return (ab[:,1] - ab[:,0]) * self.rng.random(size=(1, len(self.action_bounds))) + ab[:,0]

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
        # use state_names to sync order
        state_ord = [state[k] for k in self.state_names]
        # --> shape = concat[mus, precisions]
        g = self.pi_new(state_ord)[0][0].numpy()
        mus = g[:len(self.action_bounds)]
        precs = g[len(self.action_bounds):]
        # since diagonal --> can just invert to get covar
        covar = 1. / precs
        # sample from gaussian
        sample = self.rng.normal(mus, np.sqrt(covar), size=(1, len(mus)))
        ab = np.array(self.action_bounds)
        sample = np.clip(sample, ab[:, 0], ab[:, 1])

        if debug:
            print('mean, covar, and sample')
            print(mus)
            print(covar)
            print(sample)

        return sample

    def _calculate_v(self, states: List[Dict[str, np.ndarray]]):
        # TODO: move to function outside of class?
        # Returns: critic evals = List[np.ndarray]
        #               array for each trajectory
        V = []
        # iter thru trajectories:
        for traj in states:
            V_traj = []
            # iter thru windows within a trajectory:
            for win_start in range(0, np.shape(traj[self._sname0])[0], 32):
                # uses self.state_names to sync ordering
                state_wins = [traj[k][win_start:win_start+32] for k in self.state_names]
                V_traj.append(self.critic(state_wins)[0].numpy())
            V.append(np.concatenate(V_traj, axis=0))
        return V

    def _test_layersigs(self, dset: tf.data.Dataset):
        for v in dset.batch(16):
            vout = self.test_model(v)
            print(vout)
            for k in vout:
                assert vout[k].numpy()
            break
        self.lsig_test = True

    def train(self,
              states: List[Dict[str, np.ndarray]],
              reward: List[np.ndarray],
              actions: List[np.ndarray],
              terminated: List[bool]) -> Dict:
        """train agent on data trajectories
        KEY: each element of outer list for each
            argument = different trajectory/run

        Args:
            states (List[Dict[str, np.ndarray]]):
                outer list = each distinct state
                inner dict = mapping from state names to
                    (T + 1) x ... arrays
            reward (List[np.ndarray]):
                Each list is a different trajectory.
                Each ndarray has shape T x ...
            actions (List[np.ndarray]): where len of each state
                Each list is a different trajectory.
                Each ndarray has shape T x ...
            terminated (List[bool]): whether each trajectory
                was terminated or is still running

        Returns:
            Dict: loss history
        """
        # filter out short trajectories
        states2, actions2, rewards2, terms2 = [], [], [], []
        for i in range(len(actions)):
            if np.shape(actions[i])[0] > 5:
                states2.append(states[i])
                actions2.append(actions[i])
                rewards2.append(reward[i])
                terms2.append(terminated[i])
        V = self._calculate_v(states2)
        dset = package_dataset(states2, V, rewards2, actions2, terms2,
                               self.gamma, self.lam,
                               adv_name="adv", val_name="val", action_name="action")
        if not self.lsig_test:
            self._test_layersigs(dset)
        history = self.kmodel.fit(dset.batch(self.train_batch_size),
                                  epochs=self.train_epoch,
                                  verbose=0)

        # copy update actor to old actor
        copy_model(self.pi_new.layer, self.pi_old.layer, 1.)
        return history

    def save_weights(self, directory_location: str):
        # get_weights --> List[np.ndarray]
        np.savez(os.path.join(directory_location, "actor_weights.npz"), *self.pi_new.layer.get_weights())
        np.savez(os.path.join(directory_location, "critic_weights.npz"), *self.critic.layer.get_weights())

    def load_weights(self, directory_location: str):
        d_actor = np.load(os.path.join(directory_location, "actor_weights.npz"))
        actor_weights = [d_actor["arr_" + str(i)] for i in range(len(d_actor))]
        self.pi_new.layer.set_weights(actor_weights)
        self.pi_old.layer.set_weights(actor_weights)  # TODO: necessary?
        d_critic = np.load(os.path.join(directory_location, "critic_weights.npz"))
        critic_weights = [d_critic["arr_" + str(i)] for i in range(len(d_critic))]
        self.critic.layer.set_weights(critic_weights)


class PPOContinuousExplo(Agent, TrainEpoch):

    def __init__(self,
                 pi_model_builder: Callable[[], VectorStateModel],
                 value_model_builder: Callable[[], ScalarStateModel],
                 encoder_model_builder: Callable[[], VectorStateModel],
                 forward_model_builder: Callable[[], VectorModel],
                 action_bounds: List[Tuple[float]],
                 state_dims: Dict[str, Tuple[int]],
                 eta: float = 0.2,
                 vf_scale: float = 1.,
                 entropy_scale: float = .01,
                 gamma: float = 0.99,
                 lam: float = 1.,
                 train_batch_size: int = 32,
                 train_epoch: int = 8,
                 learning_rate: float = .001):
        """Continuous PPO agent + Forward Exploration
        Args:
            pi_model_builder (Callable[[], VectorStateModel]): actor model builder
                ASSUMES: actor model outputs gaussian distribution
                        = batch_size x (2 * action_dims)
                            first set of action dims = mean
                            second set = diagonal of precision matrix
            value_model_builder (Callable[[], ScalarStateModel]): critic model builder
            encoder_model_builder (Callable[[], VectorStateModel]): builds phi
            forward_model_builder (Callable[[], VectorStateModel]): forward prediction model builder
                forward model maps from action_t + state_t --> phi(state_{t+1})
            action_bounds (List[Tuple[float]]): (lower bound, upper bound) pairs for
                each action dimension
            state_dims (Dict[str, Tuple[int]]): mapping of state names to state dims
            eta (float, optional): ppo clipping discount. Defaults to 0.2.
            vf_scale (float, optional): critic error regularization strength. Defaults to 1.
            entropy_scale (float, optional): entropy regularization strength. Defaults to .01.
            gamma (float, optional): discount factor. Defaults to 0.99.
            lam (float, optional): generalized discount scale. Defaults to 1..
            train_batch_size (int, optional): Defaults to 32.
            train_epoch (int, optional): Defaults to 8.
            learning_rate (float, optional): Defaults to .001.
        """
        super(PPOContinuous, self).__init__()
        self.rng = npr.default_rng(42)
        self.action_bounds = action_bounds
        self.gamma = gamma
        self.lam = lam
        self.train_batch_size = train_batch_size
        self.train_epoch = train_epoch

        # multi-state system
        # pulling out state names ensures consistent ordering for model calls
        self.state_names = list(state_dims.keys())
        s0_inputs = [tf.keras.Input(shape=state_dims[k], dtype=tf.float32, name=k)
                     for k in self.state_names]
        s1_inputs = [tf.keras.Input(shape=state_dims[k], dtype=tf.float32, name=k+"t1")
                     for k in self.state_names]
        inputs = [tf.keras.Input(shape=(len(action_bounds),),
                                 name="action", dtype=tf.float32),
                  tf.keras.Input(shape=(),  # advantage
                                 name="adv", dtype=tf.float32),
                  tf.keras.Input(shape=(),  # target value name
                                 name="val", dtype=tf.float32)]
        inputs = inputs + s0_inputs

        # ppo loss
        pi_new = pi_model_builder()
        pi_old = pi_model_builder()
        v_model = value_model_builder()
        # these lines set state ordering (ordering set internally by state names)
        pi_new_distro, pi_new_test = pi_new(s0_inputs)
        pi_old_distro, pi_old_test = pi_old(s0_inputs)
        critic_pred, critic_test = v_model(s0_inputs)
        loss, pr_ratio = ppo_loss_gauss(pi_old_distro[:, :len(action_bounds)], pi_old_distro[:, len(action_bounds):],
                              pi_new_distro[:, :len(action_bounds)], pi_new_distro[:, len(action_bounds):],
                              critic_pred,
                              inputs[0], inputs[1], inputs[2],
                              eta, vf_scale, entropy_scale)
        # TODO: forward model loss + encoder loss
        phi = encoder_model_builder()
        forward_model = forward_model_builder()
        converter_layer = MapperModel(Dense(len(action_bounds)))
        forward_error, f_test = forward_surprisal(forward_model, phi, s0_inputs, s1_inputs, inputs[0])
        inv_error, inv_test = inverse_dynamics_error(phi, converter_layer, s0_inputs, s1_inputs, inputs[0])
        # primary model
        self.kmodel = CustomModel("loss",
                                  inputs=inputs,
                                  outputs={"loss": tf.math.reduce_mean(loss),
                                           "pr_ratio": pr_ratio,
                                           "new_distro": pi_new_distro,
                                           "old_distro": pi_old_distro})
        self.kmodel.compile(tf.keras.optimizers.Adam(learning_rate))
        # forward + encoder training model
        self.fek_model = CustomModel("loss",
                                  inputs=s0_inputs + s1_inputs + inputs[:1],
                                  outputs={"loss": tf.math.reduce_mean(forward_error) + tf.math.reduce_mean(inv_error)})
        self.kmodel.compile(tf.keras.optimizers.Adam(learning_rate))
        # testing model
        self.test_model = Model(inputs=inputs,
                                outputs={"pi_new_test": pi_new_test,
                                         "pi_old_test": pi_old_test,
                                         "critic_test": critic_test,
                                         "f_test": f_test,
                                         "inv_test": inv_test})
        self.lsig_test = False
        self.pi_new = pi_new
        self.pi_old = pi_old
        self.critic = v_model
        self._sname0 = s0_inputs[0].name

    def init_action(self) -> np.ndarray:
        """Initial action agent should take

        Returns:
            np.ndarray: len = dims in action space
        """
        # uniform distro within bounds
        ab = np.array(self.action_bounds)
        return (ab[:,1] - ab[:,0]) * self.rng.random(size=(1, len(self.action_bounds))) + ab[:,0]

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
        # use state_names to sync order
        state_ord = [state[k] for k in self.state_names]
        # --> shape = concat[mus, precisions]
        g = self.pi_new(state_ord)[0][0].numpy()
        mus = g[:len(self.action_bounds)]
        precs = g[len(self.action_bounds):]
        # since diagonal --> can just invert to get covar
        covar = 1. / precs
        # sample from gaussian
        sample = self.rng.normal(mus, np.sqrt(covar), size=(1, len(mus)))
        ab = np.array(self.action_bounds)
        sample = np.clip(sample, ab[:, 0], ab[:, 1])

        if debug:
            print('mean, covar, and sample')
            print(mus)
            print(covar)
            print(sample)

        return sample

    def _calculate_v(self, states: List[Dict[str, np.ndarray]]):
        # TODO: move to function outside of class?
        # Returns: critic evals = List[np.ndarray]
        #               array for each trajectory
        V = []
        # iter thru trajectories:
        for traj in states:
            V_traj = []
            # iter thru windows within a trajectory:
            for win_start in range(0, np.shape(traj[self._sname0])[0], 32):
                # uses self.state_names to sync ordering
                state_wins = [traj[k][win_start:win_start+32] for k in self.state_names]
                V_traj.append(self.critic(state_wins)[0].numpy())
            V.append(np.concatenate(V_traj, axis=0))
        return V

    def _test_layersigs(self, dset: tf.data.Dataset):
        for v in dset.batch(16):
            vout = self.test_model(v)
            print(vout)
            for k in vout:
                assert vout[k].numpy()
            break
        self.lsig_test = True

    def train(self,
              states: List[Dict[str, np.ndarray]],
              reward: List[np.ndarray],
              actions: List[np.ndarray],
              terminated: List[bool]) -> Dict:
        """train agent on data trajectories
        KEY: each element of outer list for each
            argument = different trajectory/run

        Args:
            states (List[Dict[str, np.ndarray]]):
                outer list = each distinct state
                inner dict = mapping from state names to
                    (T + 1) x ... arrays
            reward (List[np.ndarray]):
                Each list is a different trajectory.
                Each ndarray has shape T x ...
            actions (List[np.ndarray]): where len of each state
                Each list is a different trajectory.
                Each ndarray has shape T x ...
            terminated (List[bool]): whether each trajectory
                was terminated or is still running

        Returns:
            Dict: loss history
        """
        # TODO: have to update reward with intrinsice motivation factor!!!!
        #   --> has to be done before updating forward model!!!!


        # filter out short trajectories
        states2, actions2, rewards2, terms2 = [], [], [], []
        states_t = []
        for i in range(len(actions)):
            if np.shape(actions[i])[0] > 5:
                actions2.append(actions[i])
                rewards2.append(reward[i])
                terms2.append(terminated[i])
                states2.append(states[i])
                # states with time offset for forward model training
                sadd = {k: states[i][k][:-1] for k in states}
                for k in states[i]:
                    sadd[k+"t1"] = states[i][k][1:]
                states_t.append(sadd)

        V = self._calculate_v(states2)
        dset = package_dataset(states2, V, rewards2, actions2, terms2,
                               self.gamma, self.lam,
                               adv_name="adv", val_name="val", action_name="action")
        if not self.lsig_test:
            self._test_layersigs(dset)
        history = self.kmodel.fit(dset.batch(self.train_batch_size),
                                  epochs=self.train_epoch,
                                  verbose=0)

        # TODO: this needs a bunch of testing!
        # package dset and train forward/encoder models
        d_fek = {}
        for k in states_t[0]:
            d_fek[k] = np.concatenate([st[k] for st in states_t], axis=0)
        d_fek["action"] = np.concatenate(actions2, axis=0)
        _ = self.kmodel.fit(tf.data.Dataset.from_tensor_slices(d_fek).batch(self.train_batch_size),
                            epochs=self.train_epoch,
                            verbose=1)

        # copy update actor to old actor
        copy_model(self.pi_new.layer, self.pi_old.layer, 1.)
        return history
