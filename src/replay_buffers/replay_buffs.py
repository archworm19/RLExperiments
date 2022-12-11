"""Basic in-memory replay buffers

    They should all implement:
        1. append
        2. pull_sample
    but the datatypes can be different

"""
import numpy.random as npr
from typing import List, Dict, Any


# TODO: make an interface?


class MemoryBuffer:

    def __init__(self, field_names: List[str], rng: npr.Generator,
                 buffer_size: int):
        self.dat = {fn: [] for fn in field_names}
        self._sample_key = field_names[0]
        self.rng = rng
        self.buffer_size = buffer_size

    def _purge(self):
        # keep only the last [buffer_size]
        for k in self.dat:
            self.dat[k] = self.dat[k][-self.buffer_size:]

    def append(self, new_dat: Dict[str, Any]):
        # add a single sample
        # will error out if you don't provide all
        #   field names
        for k in self.dat:
            self.dat[k].append(new_dat[k])
        if len(self.dat[self._sample_key]) > self.buffer_size:
            self._purge()

    def pull_sample(self, num_sample: int):
        # returns Dict[str, List]
        # keys = field names
        # 0th dim of list = sample dim
        #   shape = num_sample x ...
        inds = self.rng.integers(0, len(self.dat[self._sample_key]), num_sample)
        ret_dat = {}
        for k in self.dat:
            ret_dat[k] = [self.dat[k][z] for z in inds]
        return ret_dat


# TODO: N-step replay buffer

