"""Agent interface
    Exposes methods to allow 1. running, 2. training of an agent

"""
import numpy as np
from abc import ABC
from typing import List, Union
from dataclasses import dataclass


@dataclass
class RunData:
    # TODO: does anyone still use this?
    # TODO: for larger datasets --> this will need
    # to be changed to an interface
    states: List
    states_t1: List
    # typically one-hots
    actions: List
    rewards: List
    termination: List


class Agent(ABC):

    def init_action(self):
        """Initial action agent should take

        Returns:
            Union[int, List[float]]:
                int if discrete action space
                List[float] if continuous
        """
        pass

    def select_action(self, state: List[np.ndarray], debug: bool):
        """select 

        Args:
            state (List[np.ndarray]): set of unbatched input tensors
                each with shape:
                    ...

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
