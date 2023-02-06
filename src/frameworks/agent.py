"""Agent interface
    Exposes methods to allow 1. running, 2. training of an agent

"""
import numpy as np
from abc import ABC
from typing import List, Union, Dict


# TODO: this is a good candidate for protocols / structural subtyping
# https://peps.python.org/pep-0544/#unions-and-intersections-of-protocols
# = most natural way to do type intersections
# see 'Unions and intersections of protocols'


class Agent(ABC):

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

    def train(self, debug: bool):
        """train agent on saved data

        Args:

        Returns:
            Dict: loss history
        """
        pass

    def save_data(self,
                  state: List[List[float]],
                  state_t1: List[List[float]],
                  action: Union[int, float, List],
                  reward: float,
                  termination: bool):
        """add single entry to replay memory
            why here? data format is hard dependency for training

        Args:
            state (List[List[float]]):
            state_t1 (List[List[float]]):
                outer list = all the different states
                    could be different shapes
                inner list = dims for given state
            action (Union[int, float, List]): action taken
                int for discrete; float for continuous
                List for multi-pronged acrions
            reward (float): reward experienced
            termination (bool): done?
        """
        pass

    def end_epoch(self):
        """Signal to agent that epoch is over"""
        pass


class AgentEpoch(ABC):
    # agent that gets trained at every epoch
    #   doesn't need to keep track of data internally

    def init_action(self):
        """Initial action agent should take

        Returns:
            Union[int, List[float]]:
                int if discrete action space
                List[float] if continuous
        """
        pass

    def select_action(self, state: Dict[str, np.ndarray], test_mode: bool, debug: bool):
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
        pass

    def train(self,
              states: List[List[np.ndarray]],
              reward: List[np.ndarray],
              actions: List[np.ndarray],
              terminated: List[bool]):
        """train agent on data trajectories

        Args:
            states (List[List[np.ndarray]]):
                outer list = each distinct state
                inner list: elements = different trajectories
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
