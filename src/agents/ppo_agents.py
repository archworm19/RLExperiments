"""Proximal Policy Optimization (PPO) agents"""
import os
import numpy as np
import numpy.random as npr
import tensorflow as tf
from typing import List, Tuple, Callable, Dict
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense
from frameworks.layer_signatures import DistroStateModel, ScalarStateModel, VectorStateModel, VectorModel, MapperModel
from frameworks.agent import Agent, TrainEpoch, WeightMate
from frameworks.ppo import package_dataset, ppo_loss_multiclass, ppo_loss_gauss
from frameworks.exploration_forward import forward_surprisal, inverse_dynamics_error
from frameworks.custom_model import CustomModel


# shared utilities


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


def test_layersigs(test_model, dset: tf.data.Dataset):
    # ensures that all test bits output by test_model are true
    for v in dset.batch(16):
        vout = test_model(v)
        for k in vout:
            assert vout[k].numpy()
        break


def calculate_v(critic_model,
                states: List[Dict[str, np.ndarray]],
                state_names: List[str]):
    # V = value = estimate reward over horizon
    # Returns: critic evals = List[np.ndarray]
    #               array for each trajectory
    V = []
    # iter thru trajectories:
    for traj in states:
        V_traj = []
        # iter thru windows within a trajectory:
        for win_start in range(0, np.shape(traj[state_names[0]])[0], 32):
            # uses state_names to sync ordering
            state_wins = [traj[k][win_start:win_start+32] for k in state_names]
            V_traj.append(critic_model(state_wins)[0].numpy())
        V.append(np.concatenate(V_traj, axis=0))
    return V


def filter_short_trajs(traj0: List, trajs: List, min_length: int = 5):
    # filter out short trajectories
    # NOTE: uses 
    ret_trajs = [[] for _ in range(1 + len(trajs))]
    for i in range(len(traj0)):
        if np.shape(traj0[i])[0] > min_length:
            ret_trajs[0].append(traj0[i])
            for j in range(1, len(ret_trajs)):
                ret_trajs[j].append(trajs[j-1][i])
    return ret_trajs


def save_weights(directory_location: str, layers: List[Layer], fn_roots: List[str]):
    for l, fnr in zip(layers, fn_roots):
        np.savez(os.path.join(directory_location, fnr + ".npz"), *l.get_weights())


def load_weights(directory_location: str, layers: List[Layer], fn_roots: List[str]):
    for l, fnr in zip(layers, fn_roots):
        d = np.load(os.path.join(directory_location, fn_roots+".npz"))
        l = [d["arr_"+str(i)] for i in range(len(d))]
        l.set_weights(l)



# models


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
        [actions2, states2, rewards2, terms2] = filter_short_trajs(actions, [states, reward, terminated])
        V = calculate_v(self.critic, states2, self.state_names)
        dset = package_dataset(states2, V, rewards2, actions2, terms2,
                               self.gamma, self.lam,
                               adv_name="adv", val_name="val", action_name="action")
        if not self.lsig_test:
            test_layersigs(self.test_model, dset)
            self.lsig_test = True
        history = self.kmodel.fit(dset.batch(self.train_batch_size),
                                  epochs=self.train_epoch,
                                  verbose=0)
        # copy update actor to old actor
        copy_model(self.pi_new.layer, self.pi_old.layer, 1.)
        return history

    def save_weights(self, directory_location: str):
        save_weights(directory_location,
                     [self.pi_new.layer, self.critic.layer],
                     ["actor_weights", "critic_weights"])

    def load_weights(self, directory_location: str):
        load_weights(directory_location,
                     [self.pi_new.layer, self.pi_old.layer, self.critic.layer],
                     ["actor_weights", "actor_weights", "critic_weights"])


class PPOContinuous(Agent, TrainEpoch, WeightMate):

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
        [actions2, states2, rewards2, terms2] = filter_short_trajs(actions, [states, reward, terminated])
        V = calculate_v(self.critic, states2, self.state_names)
        dset = package_dataset(states2, V, rewards2, actions2, terms2,
                               self.gamma, self.lam,
                               adv_name="adv", val_name="val", action_name="action")
        if not self.lsig_test:
            test_layersigs(self.test_model, dset)
            self.lsig_test = True
        history = self.kmodel.fit(dset.batch(self.train_batch_size),
                                  epochs=self.train_epoch,
                                  verbose=0)

        # copy update actor to old actor
        copy_model(self.pi_new.layer, self.pi_old.layer, 1.)
        return history

    def save_weights(self, directory_location: str):
        save_weights(directory_location,
                     [self.pi_new.layer, self.critic.layer],
                     ["actor_weights", "critic_weights"])

    def load_weights(self, directory_location: str):
        load_weights(directory_location,
                     [self.pi_new.layer, self.pi_old.layer, self.critic.layer],
                     ["actor_weights", "actor_weights", "critic_weights"])


class PPOContinuousExplo(Agent, TrainEpoch, WeightMate):

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
                 learning_rate: float = .001,
                 reward_scale: float = 1.,
                 exploration_reward_scale: float = 1.):
        """Continuous PPO agent + Forward Exploration
        Args:
            pi_model_builder (Callable[[], VectorStateModel]): actor model builder
                ASSUMES: actor model outputs gaussian distribution
                        = batch_size x (2 * action_dims)
                            first set of action dims = mean
                            second set = diagonal of precision matrix
            value_model_builder (Callable[[], ScalarStateModel]): critic model builder
            encoder_model_builder (Callable[[], VectorStateModel]): builds phi
            forward_model_builder (Callable[[], VectorModel]): forward prediction model builder
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
            reward_scale (float, optional): how much reward is weighted
            exploration_reward_scale (float, optional): how much exploration reward is weighted
        """
        super(PPOContinuousExplo, self).__init__()
        self.rng = npr.default_rng(42)
        self.action_bounds = action_bounds
        self.gamma = gamma
        self.lam = lam
        self.train_batch_size = train_batch_size
        self.train_epoch = train_epoch
        self.reward_scale = reward_scale
        self.exploration_reward_scale = exploration_reward_scale

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
        # forward model loss + encoder loss
        self.phi = encoder_model_builder()
        self.forward_model = forward_model_builder()
        converter_layer = MapperModel(Dense(len(action_bounds)))
        forward_error, f_test = forward_surprisal(self.forward_model, self.phi, s0_inputs, s1_inputs, inputs[0])
        inv_error, inv_test = inverse_dynamics_error(self.phi, converter_layer, s0_inputs, s1_inputs, inputs[0])
        # primary model
        self.kmodel = CustomModel("loss",
                                  inputs=inputs + s0_inputs,
                                  outputs={"loss": tf.math.reduce_mean(loss),
                                           "pr_ratio": pr_ratio,
                                           "new_distro": pi_new_distro,
                                           "old_distro": pi_old_distro})
        self.kmodel.compile(tf.keras.optimizers.Adam(learning_rate))
        # forward + encoder training model
        self.kencoder_model = CustomModel("loss",
                                         inputs=s0_inputs + s1_inputs + inputs[:1],
                                         outputs={"loss":tf.math.reduce_mean(inv_error)})
        self.kforward_model = CustomModel("loss",
                                         inputs=s0_inputs + s1_inputs + inputs[:1],
                                          outputs={"loss": tf.math.reduce_mean(forward_error),
                                                   "surprisal": forward_error})
        self.kencoder_model.compile(tf.keras.optimizers.Adam(learning_rate))
        self.kforward_model.compile(tf.keras.optimizers.Adam(learning_rate))
        # testing model
        self.test_model1 = Model(inputs=inputs + s0_inputs,
                                outputs={"pi_new_test": pi_new_test,
                                         "pi_old_test": pi_old_test,
                                         "critic_test": critic_test})
        self.test_model2 = Model(inputs=s0_inputs + s1_inputs + inputs[:1],
                                outputs={"f_test": f_test,
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

    def _make_offset_states(self, states: List[Dict[str, np.ndarray]]) -> Tuple[Dict[str, np.ndarray],
                                                                                Dict[str, np.ndarray]]:
        """converts state trajectories to time offset state trajectories
            traj_i --> t: traj_i[:-1], t+1: traj_i[1:]

        Args:
            states (List[Dict[str, np.ndarray]]):
                outer list = each distinct state
                inner dict = mapping from state names to
                    (T + 1) x ... arrays

        Returns:
            Dict[str, np.ndarray]] states_t
                traj_i --> traj_i[:-1]
            Dict[str, np.ndarray]: states_{t+1}
                traj_i --> traj_i[1:]
                for both states_t and states_{t+1},
                    stored as mapping from state names to concatenated trajectories
                    each trajectory length is shortened by 1 from original
        """
        states_t, states_t1 = {}, {}
        # iter through state names:
        for k in states[0]:
            # gather all trajectories for given name
            states_t[k] = np.concatenate([st_traj[k][:-1] for st_traj in states])
            states_t1[k] = np.concatenate([st_traj[k][1:] for st_traj in states])
        return states_t, states_t1

    def _surprisal_reward(self, explo_model: CustomModel, explo_dset: tf.data.Dataset,
                          explo_reward_name: str = "surprisal"):
        # intrinsic reward = 'forward surprisal' loss
        # TODO: normalization?
        explo_rewards = []
        for v in explo_dset.batch(32):
            vout = explo_model(v)
            explo_rewards.append(vout[explo_reward_name].numpy())
        return np.concatenate(explo_rewards, axis=0)

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
        [actions2, states2, rewards2, terms2] = filter_short_trajs(actions, [states, reward, terminated])
        # time offset states:
        states_t, states_t1 = self._make_offset_states(states2)

        # make exploration reward dataset (forward surpisal reward)
        explo_d = {k: states_t[k] for k in states_t}
        for k in states_t1:
            explo_d[k+"t1"] = states_t1[k]
        explo_d["action"] = np.concatenate(actions2)
        explo_dset = tf.data.Dataset.from_tensor_slices(explo_d)

        # intrinsic reward
        explo_reward = self._surprisal_reward(self.kforward_model,
                                              explo_dset,
                                              explo_reward_name="surprisal")

        # combine intrinsic reward with rewards2
        ind_st = 0
        for i in range(len(rewards2)):
            ind_end = ind_st + len(rewards2[i])
            rewards2[i] = self.reward_scale * rewards2[i] + self.exploration_reward_scale * explo_reward[ind_st:ind_end]
            ind_st = ind_end

        # train primary model
        V = calculate_v(self.critic, states2, self.state_names)
        dset = package_dataset(states2, V, rewards2, actions2, terms2,
                               self.gamma, self.lam,
                               adv_name="adv", val_name="val", action_name="action")
        if not self.lsig_test:
            test_layersigs(self.test_model1, dset)
            test_layersigs(self.test_model2, explo_dset)
            self.lsig_test = True
        history = self.kmodel.fit(dset.batch(self.train_batch_size),
                                  epochs=self.train_epoch,
                                  verbose=1)

        # train forward model and encoders
        # package dset and train forward/encoder models
        # NOTE: forward model depends on encoder; encoder does not depend on forward model
        _ = self.kencoder_model.fit(explo_dset.batch(self.train_batch_size),
                               epochs=self.train_epoch,
                               verbose=1)
        _ = self.kforward_model.fit(explo_dset.batch(self.train_batch_size),
                               epochs=self.train_epoch,
                               verbose=1)

        # copy update actor to old actor
        copy_model(self.pi_new.layer, self.pi_old.layer, 1.)
        return history

    def save_weights(self, directory_location: str):
        save_weights(directory_location,
                     [self.pi_new.layer, self.critic.layer, self.phi.layer, self.forward_model.layer],
                     ["actor_weights", "critic_weights", "encoder_weights", "forward_weights"])

    def load_weights(self, directory_location: str):
        load_weights(directory_location,
                     [self.pi_new.layer, self.pi_old.layer, self.critic.layer, self.phi.layer, self.forward_model.layer],
                     ["actor_weights", "actor_weights", "critic_weights", "encoder_weights", "forward_weights"])
