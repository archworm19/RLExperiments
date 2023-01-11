"""Replay Buffers for Prioritized Experience Replay
    First introduced in Schaul et al, 2015
"""
import heapq
import numpy as np
import numpy.random as npr
from typing import List
from dataclasses import dataclass, field
from queue import PriorityQueue


@dataclass(order=True)
class PItem:
    priority: float
    item: List=field(compare=False)


def _down_swap(heap: List[PItem], target_idx: int):
    # swap with smallest child recursively till heap exhaustion
    # check if children exist
    lc = 2 * target_idx + 1
    rc = 2 * target_idx + 2
    min_child_idx = None
    if lc >= len(heap):
        return
    elif rc >= len(heap):
        min_child_idx = lc
    else:
        if heap[lc] <= heap[rc]:
            min_child_idx = lc
        else:
            min_child_idx = rc

    # basecase: element <= both children
    if heap[target_idx] <= heap[min_child_idx]:
        return

    # swap with min child
    v_hold = heap[target_idx]
    heap[target_idx] = heap[min_child_idx]
    heap[min_child_idx] = v_hold

    # descend into child ~ follow the target
    _down_swap(heap, min_child_idx)


def heap_pop_target(heap: List[PItem], target_idx: int):
    """pop target element of min-heap
    NOTE: this is pretty much the same procedure
        as pop --> heapq might support eventually

    Args:
        heap (List[PItem]): min-heap
            smallest values are prioritized
        target_idx (int): index to be popped
            (deleted and returned)
    """
    v = heap[target_idx]
    # copy last element value to target location --> down copy
    heap[target_idx] = heap[-1]
    _down_swap(heap, target_idx)
    # I assume python just decreases the end bounds for the arraylist
    #   else this would be inefficient
    heap.pop()
    return v


class MemoryBufferPQ:
    # heap-based priority queue memory buffer
    # keeps array sorted in terms of supplied priority score
    #   lower priority score pulled first
    # use rank-based stochastic sampling procedure
    #   introduced by Schaul et al, 2015
    # TODO: for importance sampling,
    #   should inserted probability be used or should
    #   segment probability be used?

    def __init__(self, rng: npr.Generator, buffer_size: int,
                 batch_size: int):
        # batch_size: number of samples to pull at a time
        # cumulative distro is broken up into
        #   [batch_size] segments of equal sum(probability)
        #   segments are tiled across underlying heapq array
        #       --> earlier segments will have fewer elements
        self.rng = rng
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self._segment_bounds = None
        self._hpq = []

    def _purge(self):
        self._hpq = self._hpq[:self.buffer_size]

    def append(self, sample_prob: float, new_dat: List):
        # sample_prob = sampling probability
        #   will produce odd results if not actually probability
        # NOTE: negate sampling prob -->
        #   highest prob will have lowest prority scores
        #   = pulled first
        v = PItem(priority=-1. * sample_prob, item=new_dat)
        heapq.heappush(self._hpq, v)
        self._purge()

    def _calculate_segment_bounds(self):
        # update segment bounds

        # expensive O(N) operations
        raw_probs = [-1. * v.priority for v in self._hpq]
        cprobs = np.cumsum(raw_probs) / np.sum(raw_probs)
        cprobs = np.vstack([0., cprobs])

        # find boundaries:
        pbounds = np.linspace(0., 1., self.batch_size+1)[:self.batch_size]
        segment_starts = [np.where(np.logical_and(pb >= cprobs[:-1],
                                                  pb < cprobs[1:]))[0][0]
                          for pb in pbounds]
        self._segment_bounds = segment_starts + [len(self._hpq)]

    def pull_samples(self):
        # TODO:ahhh... this is more complex than I was thinking
        #   how do we pop these low probability dudes?
        # idea 1: use

        # TODO: ideas?
        # properties of original paper?
        # > batch_size segments with same sum(probability)
        # > randomly samples within each segment
        # uhh... why are you using a priority queue for this?
        # just need the thing to be approximately sorted?
        # welll... not quite --> heap automatically shifts data
        #       into correct segment (and sets which elems to remove from last segment first)


        # TODO: I supposed we could implement our own
        #   pop method ~ handles the sift up procedure starting
        #   from some index ~ log(N) swap operations ~ this is the best we'll do
        # TEST: make sure it yield same result as bult in heappop for item 0

        # pulls batch_size samples
        # NOTE: this function pops samples from priority queue
        #   --> you have to reinsert them if you want them to stay
        # uses stratfied sampling system ~ pulls 1 sample from each segment
        # Returns: List[List]
        ret = []
        for i in range(len(self._segment_bounds) - 1):
            ind = self.rng.integers(self._segment_bounds[i],
                                    self._segment_bounds[i + 1])
            ret.append()


if __name__ == "__main__":
    # TODO: move to tests
    items = [PItem(1., ["a", "b"]),
             PItem(2., ["c", "d"]),
             PItem(-100., ["z"]),
             PItem(-50, ["opa"])]

    # TESTING heap functionality
    # insert into heap:
    heap = []
    for v in items:
        heapq.heappush(heap, v)
    print(heap)
    # heapsort check
    priorz = []
    for _ in range(4):
        priorz.append(heapq.heappop(heap).priority)
    print(priorz)
    print(heap)
    # insert again --> test popping internal node
    for v in items:
        heapq.heappush(heap, v)
    print(heap_pop_target(heap, 0))
    print(heap)
    print(heap_pop_target(heap, 1))
    print(heap)
    
