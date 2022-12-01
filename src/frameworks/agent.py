"""Agent interface
    Exposes methods to allow 1. running, 2. training of an agent

    KEY: all core datatypes are numpy arrays
        models will have to convert to their desired formats
"""
import numpy as np
from abc import ABC
from typing import List
from dataclasses import dataclass


@dataclass
class RunData:
    # TODO: for larger datasets --> this will need
    # to be changed to an interface
    states: List[np.ndarray]
    states_t1: List[np.ndarray]
    # typically one-hots
    actions: np.ndarray
    rewards: np.ndarray


class Agent(ABC):

    def select_action(self, state: List[np.ndarray]):
        """select 

        Args:
            state (List[np.ndarray]): set of unbatched input tensors
                each with shape:
                    ...

        Returns:
            int: index of selected action
        """
        pass

    def train(self, run_data: RunData, num_epoch: int):
        """train agent on run data

        Args:
            run_data (RunData):
            num_epoch (int):

        Returns:
            Dict: loss history
        """
        pass
