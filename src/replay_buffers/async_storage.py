"""Asynchronous storage ~ useful for parallel agents"""
import numpy as np
from typing import List
from copy import deepcopy


class TrajectoryStore:
    # stores trajectories of some number of variables
    #   uses pointers to keep track of trajectory separations
    # NOTE: there's a subtle distinction between TrajectoryStore's end bit
    #       and overall termination bit
    #       if trial hits time lenth --> TrajectoryStore sent end signal
    #           but the trial didn't technically terminate
    def __init__(self, num_procs: int, T: int, dims: List[int]):
        """
        Args:
            num_procs (int): number of processes
            T (int): total number of timepoints
            dims (List[int]): len(dims) = number of variables
                dims[i] = dimensionality of variable i
        """
        self.num_procs = num_procs
        self._dstore = [np.zeros((num_procs, T, dimi), dtype=np.float32)
                        for dimi in dims]
        # active ptrs need to keept track of (t0, idx of last added item, has trial ended?)
        self._active_ptrs = np.full((num_procs, 3), -1, dtype=np.int32)
        self._active_ptrs[:, 0] = 0
        # old ptrs: (pid, t0, tend) ~ all processes get stored together
        self._old_ptrs = []  # (pid, t0, tend + 1)

    def add_datapt(self, process_idx: int, x: List[np.ndarray], end_bit: bool):
        """add a single datapoiont

        Args:
            process_idx (int): process index
            x (List[np.ndarray]): all variables for given datapoint
                requirement: len(x[i]) = self.dims[i]
            end_bit (bool): whether or not this is the last value in the trajectory
        """
        ptr = self._active_ptrs[process_idx]  # --> [t0, idx of last iterm, end bit?]
        if ptr[2] > 0.5:  # if previous trajectory ended
            # copy over old pointer
            v = [process_idx, ptr[0], ptr[1] + 1]
            self._old_ptrs.append(v)
            # modify active ptr:
            ptr[0] = v[-1]
            ptr[1] = v[-1]
            ptr[2] = int(1 * end_bit)
        else:
            ptr[1] = ptr[1] + 1
            ptr[2] = int(1 * end_bit)
        # add the new datapt
        for i, xi in enumerate(x):
            self._dstore[i][process_idx, ptr[1]] = xi

    def pull_trajectories(self) -> List[List[np.ndarray]]:
        """pull all trajectories; including active trajectories

        Returns:
            List[List[np.ndarray]]: outer list = the different variables
                inner list = the trajectories for given variable
                typical case: returns [state, reward] where state and reward
                    are composed of multiple trajectories 
        """
        # process active ptrs:
        _ta = deepcopy(self._active_ptrs)
        _ta[:,1] = _ta[:,1] + 1
        traj_adds = np.hstack((np.arange(self.num_procs)[:, None], _ta[:,:2]))
        all_ptrs = self._old_ptrs + [tai for tai in traj_adds]

        # outer list = different variables
        fr = []
        # iter thru vars:
        for i in range(len(self._dstore)):
            all_traj = []
            for ptr in all_ptrs:
                all_traj.append(self._dstore[i][ptr[0], ptr[1]:ptr[2]])
            fr.append(all_traj)
        return fr
