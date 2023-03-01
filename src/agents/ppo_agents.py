"""Proximal Policy Optimization (PPO) agents"""
import os
import numpy as np
import numpy.random as npr
import tensorflow as tf
from typing import List, Tuple, Callable, Dict
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from frameworks.layer_signatures import DistroStateModel, ScalarStateModel, VectorStateModel
from frameworks.agent import Agent, TrainEpoch, WeightMate
from frameworks.ppo import package_dataset, ppo_loss_multiclass, ppo_loss_gauss, value_conv, package_dataset_critic, value_loss
from frameworks.custom_model import CustomModel


# Shared Utilities


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


def calculate_vpred(critic: VectorStateModel, states: List[Dict[str, np.ndarray]],
                    state_names: List[str]):
    # TODO: tf.function?
    # Vpred = predicted value function = critic output
    # Returns: critic evals = List[np.ndarray]
    #               array for each trajectory
    V = []
    # iter thru trajectories:
    for traj in states:
        V_traj = []
        # iter thru windows within a trajectory:
        for win_start in range(0, np.shape(traj[state_names[0]])[0], 32):
            # uses self.state_names to sync ordering
            state_wins = [traj[k][win_start:win_start+32] for k in state_names]
            V_traj.append(critic(state_wins)[0].numpy())
        V.append(np.concatenate(V_traj, axis=0))
    return V


def calculate_vpred_end(critic: VectorStateModel, states: List[Dict[str, np.ndarray]],
                        state_names: List[str]):
    Vend = []
    for traj in states:
        state_end = [traj[k][-1:] for k in state_names]
        Vend.append(critic(state_end)[0][0].numpy())
    return Vend


def test_layersigs(test_model, dset: tf.data.Dataset):
    # check the test bits
    for v in dset.batch(16):
        vout = test_model(v)
        for k in vout:
            assert vout[k].numpy()
        break


def save_weights(directory_location: str, layers: List[Layer], names: List[str]):
    for ly, ln in zip(layers, names):
        np.savez(os.path.join(directory_location, ln + ".npz"), *ly.get_weights())


def load_weights(directory_location: str, layers: List[Layer], names: List[str]):
    for ly, ln in zip(layers, names):
        dw = np.load(os.path.join(directory_location, ln + ".npz"))
        wfin = [dw["arr_" + str(i)] for i in range(len(dw))]
        ly.set_weights(wfin)


# Models


class PPODiscrete(Agent, TrainEpoch, WeightMate):

    def __init__(self,
                 pi_model_builder: Callable[[], DistroStateModel],
                 value_model_builder: Callable[[], ScalarStateModel],
                 num_actions: int,
                 state_dims: Dict[str, Tuple[int]],
                 eta: float = 0.2,
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

        # ppo loss
        pi_new = pi_model_builder()
        pi_old = pi_model_builder()
        v_model = value_model_builder()
        # these lines set state ordering (ordering set internally by state names)
        pi_new_distro, pi_new_test = pi_new(s0_inputs)
        pi_old_distro, pi_old_test = pi_old(s0_inputs)
        critic_pred, critic_test = v_model(s0_inputs)
        loss_actor, negent_actor = ppo_loss_multiclass(pi_old_distro, pi_new_distro,
                                                       inputs[0], inputs[1],
                                                       eta)
        loss_critic = value_loss(critic_pred, inputs[2])
        # primary models
        self.kmodel_critic = CustomModel("loss",
                                         inputs=s0_inputs + inputs[2:],
                                         outputs={"loss": tf.math.reduce_mean(loss_critic)})
        self.kmodel_actor = CustomModel("loss",
                                         inputs=s0_inputs + inputs[:2],
                                         outputs={"loss": tf.math.reduce_mean(loss_actor) + entropy_scale*tf.math.reduce_mean(negent_actor)})
        self.kmodel_critic.compile(tf.keras.optimizers.Adam(learning_rate))
        self.kmodel_actor.compile(tf.keras.optimizers.Adam(learning_rate))
        # test model ~ just for checking layer signatures are adhered to
        self.test_model = Model(inputs=s0_inputs + inputs,
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
        boolz = r <= np.cumsum(pr, axis=1)
        v = boolz * 1.
        act_sel = np.hstack((v[:,:1], v[:,1:] - v[:,:-1]))
        if debug:
            print('probability and random draw')
            print(pr)
            print(r)
            print(self.critic(state_ord))
            print(act_sel)
        return act_sel

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

        # first: train critic
        Vpred_end = calculate_vpred_end(self.critic, states2, self.state_names)
        critic_dset = package_dataset_critic(states2, rewards2, Vpred_end, terms2, self.gamma, "val")
        # train critic
        self.kmodel_critic.fit(critic_dset.batch(self.train_batch_size),
                               epochs=self.train_epoch,
                               verbose=1)

        # second: train actor
        #   use the updated critic to give updated value function predictions
        V = calculate_vpred(self.critic, states2, self.state_names)
        dset = package_dataset(states2, V, rewards2, actions2, terms2,
                               self.gamma, self.lam,
                               adv_name="adv", val_name="val", action_name="action")
        if not self.lsig_test:
            test_layersigs(self.test_model, dset)
            self.lsig_test = True
        history = self.kmodel_actor.fit(dset.batch(self.train_batch_size),
                                        epochs=self.train_epoch,
                                        verbose=1)
        # copy update actor to old actor
        copy_model(self.pi_new.layer, self.pi_old.layer, 1.)
        return history

    def save_weights(self, directory_location: str):
        # get_weights --> List[np.ndarray]
        save_weights(directory_location,
                     [self.pi_new.layer, self.critic.layer],
                     ["actor_weights", "critic_weights"])

    def load_weights(self, directory_location: str):
        load_weights(directory_location,
                     [self.pi_new.layer, self.pi_old.layer, self.critic.layer],
                     ["actor_weights", "actor_weights", "critic_weights"])


class PPOContinuous(Agent, TrainEpoch):

    def __init__(self,
                 pi_model_builder: Callable[[], VectorStateModel],
                 value_model_builder: Callable[[], ScalarStateModel],
                 action_bounds: List[Tuple[float]],
                 state_dims: Dict[str, Tuple[int]],
                 eta: float = 0.2,
                 entropy_scale: float = .01,
                 gamma: float = 0.99,
                 lam: float = 1.,
                 train_batch_size: int = 32,
                 train_epoch: int = 8,
                 learning_rate: float = .001):
        """Continuous PPO agent
        Args:
            pi_model_builder (Callable[[], DistroStateModel]): actor model builder
                ASSUMES: actor model outputs gaussian distribution
                        = batch_size x (2 * action_dims)
                            first set of action dims = mean
                            second set = diagonal of precision matrix
            value_model_builder (Callable[[], ScalarStateModel]): critic model builder
            action_bounds (List[Tuple[float]]): (lower bound, upper bound) pairs for
                each action dimension
            state_dims (Dict[str, Tuple[int]]): mapping of state names to state dims
            eta (float, optional): ppo clipping discount. Defaults to 0.2.
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

        # ppo loss
        pi_new = pi_model_builder()
        pi_old = pi_model_builder()
        v_model = value_model_builder()
        pi_new_distro, pi_new_test = pi_new(s0_inputs)
        pi_old_distro, pi_old_test = pi_old(s0_inputs)
        critic_pred, critic_test = v_model(s0_inputs)
        loss_actor, negent, pr_ratio = ppo_loss_gauss(pi_old_distro[:, :len(action_bounds)], pi_old_distro[:, len(action_bounds):],
                                                      pi_new_distro[:, :len(action_bounds)], pi_new_distro[:, len(action_bounds):],
                                                      inputs[0], inputs[1], eta)
        loss_critic = value_loss(critic_pred, inputs[2])
        # primary model
        self.kmodel_actor = CustomModel("loss",
                                        inputs=s0_inputs + inputs[:2],
                                        outputs={"loss": tf.math.reduce_mean(loss_actor) + entropy_scale * tf.math.reduce_mean(negent),
                                                 "pr_ratio": pr_ratio,
                                                 "new_distro": pi_new_distro,
                                                 "old_distro": pi_old_distro})
        self.kmodel_actor.compile(tf.keras.optimizers.Adam(learning_rate))
        self.kmodel_critic = CustomModel("loss",
                                         inputs=s0_inputs + inputs[2:],
                                         outputs={"loss": tf.math.reduce_mean(loss_critic)})
        self.kmodel_critic.compile(tf.keras.optimizers.Adam(learning_rate))
        # testing model
        self.test_model = Model(inputs=s0_inputs + inputs,
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

    # TODO: finish below here!

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

        # first: train critic
        Vpred_end = calculate_vpred_end(self.critic, states2, self.state_names)
        critic_dset = package_dataset_critic(states2, rewards2, Vpred_end, terms2, self.gamma, "val")
        # train critic
        self.kmodel_critic.fit(critic_dset.batch(self.train_batch_size),
                               epochs=self.train_epoch,
                               verbose=1)

        # second: train actor
        #   use the updated critic to give updated value function predictions
        V = calculate_vpred(self.critic, states2, self.state_names)
        dset = package_dataset(states2, V, rewards2, actions2, terms2,
                               self.gamma, self.lam,
                               adv_name="adv", val_name="val", action_name="action")
        if not self.lsig_test:
            test_layersigs(self.test_model, dset)
            self.lsig_test = True
        history = self.kmodel_actor.fit(dset.batch(self.train_batch_size),
                                        epochs=self.train_epoch,
                                        verbose=1)
        # copy update actor to old actor
        copy_model(self.pi_new.layer, self.pi_old.layer, 1.)
        return history

    def save_weights(self, directory_location: str):
        # get_weights --> List[np.ndarray]
        save_weights(directory_location,
                     [self.pi_new.layer, self.critic.layer],
                     ["actor_weights", "critic_weights"])

    def load_weights(self, directory_location: str):
        load_weights(directory_location,
                     [self.pi_new.layer, self.pi_old.layer, self.critic.layer],
                     ["actor_weights", "actor_weights", "critic_weights"])
