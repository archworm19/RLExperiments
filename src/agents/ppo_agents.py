"""Proximal Policy Optimization (PPO) agents"""
import numpy as np
import numpy.random as npr
import tensorflow as tf
from typing import List, Tuple, Callable, Dict
from tensorflow.keras.layers import Layer
from frameworks.layer_signatures import DistroStateModel, ScalarStateModel
from frameworks.agent import AgentEpoch
from frameworks.ppo import package_dataset, ppo_loss_multiclass
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


class PPODiscrete(AgentEpoch):

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
        super(PPODiscrete, self).__init__()
        # TODO: docstring (gamma = discout, lam = generalized discount adjustment)
        #       ... state_dims = mapping from state names to shapes
        # TODO: assumes model builder is already normalized (put in docstring)

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
        # these lines fix state ordering (ordering set internally by state names)
        pi_new_distro = pi_new(s0_inputs)
        pi_old_distro = pi_old(s0_inputs)
        critic_pred = v_model(s0_inputs)
        loss = ppo_loss_multiclass(pi_old_distro, pi_new_distro,
                                   critic_pred,
                                   inputs[0], inputs[1], inputs[2],
                                   eta, vf_scale=vf_scale, entropy_scale=entropy_scale)
        self.kmodel = CustomModel("loss",
                                  inputs=inputs,
                                  outputs={"loss": tf.math.reduce_mean(loss)})
        self.kmodel.compile(tf.keras.optimizers.Adam(learning_rate))
        self.pi_new = pi_new
        self.pi_old = pi_old
        self.critic = v_model
        self._sname0 = s0_inputs[0].name

    def init_action(self):
        """Initial action agent should take

        Returns:
            Union[int, List[float]]:
                int if discrete action space
                List[float] if continuous
        """
        return self.rng.integers(0, self.num_actions)

    def select_action(self, state: Dict[str, np.ndarray],
                      test_mode: bool = False,
                      debug: bool = False):
        """select 

        Args:
            state (Dict[str, np.ndarray]]): mapping of state names to
                set of unbatched input tensors
                each with shape:
                    ...
            test_mode (bool): are we in a 'test run' for the agent?
            debug (bool): debug mode

        Returns:
            Union[int, List[float]]:
                int if discrete action space
                List[float] if continuous
        """
        # use state_names to sync order
        state_ord = [state[k][None] for k in self.state_names]
        # --> shape = num_actions (normalized)
        pr = self.pi_new(state_ord)[0].numpy()
        r = self.rng.random()
        return np.where(r <= np.cumsum(pr))[0][0]

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
                V_traj.append(self.critic(state_wins).numpy())
            V.append(np.concatenate(V_traj, axis=0))
        return V

    def train(self,
              states: List[Dict[str, np.ndarray]],
              reward: List[np.ndarray],
              actions: List[np.ndarray],
              terminated: List[bool]):
        """train agent on data trajectories

        Args:
            states (List[Dict[str, np.ndarray]]):
                outer list = different trajectories
                inner dict = mapping from state names to each
                    state
                all states = T x ...
            reward (List[np.ndarray]):
                Each list is a different trajectory.
                Each ndarray has shape T x ...
            actions (List[np.ndarray]): where len of each state
                trajectory is T --> len of reward/action trajectory = T-1
            terminated (List[bool]): whether each trajectory
                was terminated or is still running

        Returns:
            Dict: loss history
        """
        V = self._calculate_v(states)
        dset = package_dataset(states, V, reward, actions, terminated,
                               self.gamma, self.lam,
                               adv_name="adv", val_name="val", action_name="action")
        history = self.kmodel.fit(dset.batch(self.train_batch_size),
                                  epochs=self.train_epoch,
                                  verbose=1)
        # copy update actor to old actor
        copy_model(self.pi_new, self.pi_old, 1.)
        return history
