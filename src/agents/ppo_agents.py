"""Proximal Policy Optimization (PPO) agents"""
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Callable, Dict
from frameworks.layer_signatures import DistroStateModel, ScalarStateModel
from frameworks.agent import AgentEpoch
from frameworks.ppo import package_dataset, ppo_loss_multiclass
from frameworks.custom_model import CustomModel


class PPODiscrete(AgentEpoch):

    def __init__(self,
                 pi_model_builder: Callable[[], DistroStateModel],
                 value_model_builder: Callable[[], ScalarStateModel],
                 num_actions: int,
                 state_dims: List[Tuple[int]],
                 eta: float = 0.2,
                 vf_scale: float = 1.,
                 entropy_scale: float = .01):
        # TODO: assumes model builder is already normalized

        # multi-state system
        s0_names = ["state" + str(i) for i in range(len(state_dims))]
        s0_inputs = [tf.keras.Input(shape=s, dtype=tf.float32, name=n)
                     for s, n in zip(state_dims, s0_names)]
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
        self.kmodel.compile(tf.keras.optimizers.Adam(.001))

    def init_action(self):
        """Initial action agent should take

        Returns:
            Union[int, List[float]]:
                int if discrete action space
                List[float] if continuous
        """
        pass

    def select_action(self, state: List[np.ndarray], test_mode: bool, debug: bool):
        """select 

        Args:
            state (List[np.ndarray]): set of unbatched input tensors
                each with shape:
                    ...
            test_mode (bool): are we in a 'test run' for the agent?
            debug (bool): debug mode

        Returns:
            Union[int, List[float]]:
                int if discrete action space
                List[float] if continuous
        """
        pass

    def train(self,
              states: Dict[str, List[np.ndarray]],
              V: List[np.ndarray],
              reward: List[np.ndarray],
              actions: List[np.ndarray],
              terminated: List[bool],):
        """train agent on data trajectories

        Args:
            states (Dict[str, List[np.ndarray]]): mapping from state names to
                state vectors. Each dict entry is a different state.
                Each list is a different trajectory.
                states[k0][i] matches up with states[k1][i]
            V (List[np.ndarray]): V(s_t) = critic evaluation of states
                Each list is a different trajectory.
                Each ndarray has shape T x ...
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
        pass
