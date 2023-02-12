"""Agent interface protocols
    Exposes methods to allow 1. running, 2. training of an agent

"""
import numpy as np
from abc import abstractmethod
from typing import Protocol, List, Dict


# TODO: this is a good candidate for protocols / structural subtyping
# https://peps.python.org/pep-0544/#unions-and-intersections-of-protocols
# = most natural way to do type intersections
# see 'Unions and intersections of protocols'


class Actor(Protocol):

    @abstractmethod
    def init_action(self) -> np.ndarray:
        """Initial action agent should take

        Returns:
            np.ndarray: len = dims in action space
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError


class TrainOnLine(Protocol):
    # "OnLine" = model handles storage and train set sampling

    @abstractmethod
    def train(self, debug: bool) -> Dict:
        """train agent on saved data

        Args:

        Returns:
            Dict: loss history
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def end_epoch(self):
        """Signal to agent that epoch is over"""
        raise NotImplementedError


class TrainEpoch(Protocol):
    # "Epoch" = send training data to model after an epoch
    #   model does not necessarily handle any data storage

    @abstractmethod
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
        raise NotImplementedError
