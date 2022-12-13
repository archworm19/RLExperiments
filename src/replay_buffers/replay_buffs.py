"""Basic in-memory replay buffers

    They should all implement:
        1. append
        2. pull_sample
    but the datatypes can be different

"""
import numpy as np
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


# TODO: this needs a lot of testing
class MemoryBufferNstep:
    # each returned sample = contigious set of N steps
    # NOTE: this class ASSUMES that subsequent calls
    #   to append are contigious steps
    #   unless a termination bit intervened
    #   Also: termination values expected to be boolean
    #
    #   Methodology:
    #   store each set in a separate list
    #   keep track of legal t0s for each sub-list

    def __init__(self, N: int,
                 field_names: List[str], rng: npr.Generator,
                 buffer_size: int,
                 termination_field_name: str):
        self.dat = {fn: [[]] for fn in field_names}
        self.N = N
        self._sample_key = field_names[0]
        self.rng = rng
        self.buffer_size = buffer_size
        self.tfn = termination_field_name

    def _calc_legal_set_sizes(self):
        v = self.dat[self._sample_key]
        return [max(0, 1 + len(vi) - self.N) for vi in v]

    def _purge(self):
        # drop lists until get under buffer size
        legal_ss = self._calc_legal_set_sizes()
        psamples = sum(legal_ss)
        while psamples > self.buffer_size:
            self.dat = {k: self.dat[k][1:] for k in self.dat}
            legal_ss = legal_ss[1:]
            psamples = sum(legal_ss)

    def append(self, new_dat: Dict[str, Any]):
        # add a single sample
        # will error out if you don't provide all
        #   field names

        # add data ~ assumes pointing at correct set
        for k in self.dat:
            self.dat[k][-1].append(new_dat[k])

        if new_dat[self.tfn]:
            # point to new set
            for k in self.dat:
                self.dat[k].append([])

        # purge handles whether it is necessary
        self._purge()

    def pull_sample(self, num_sample: int):
        # returns Dict[str, List]
        # keys = field names
        # 0th dim of list = sample dim
        # 1st dim = N (window size / num contiguous steps)
        #   shape = num_sample x N x ...

        legal_ss = self._calc_legal_set_sizes()
        total_legal = sum(legal_ss)
        sample_inds = self.rng.integers(0, total_legal, num_sample)

        set_starts = np.concatenate([[0], np.cumsum(legal_ss)])[:-1]
        # --> num_set x num_sample
        offset = sample_inds[None] - set_starts[:, None]
        # set inds = last positive offset (also smallest positive)
        offset_bool = offset >= 0
        set_inds, set_offsets = [], []
        for i in range(np.shape(offset)[1]):
            set_inds.append(np.where(offset_bool[:, i])[0][-1])
            set_offsets.append(offset[set_inds[-1], i])

        # package:
        d = {}
        for k in self.dat:
            d[k] = [self.dat[k][sii][soi:soi+self.N] for
                    sii, soi in zip(set_inds, set_offsets)]
        return d
