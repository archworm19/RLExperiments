"""Replay Buffers for Prioritized Experience Replay
    First introduced in Schaul et al, 2015
"""
import heapq
import numpy as np
import numpy.random as npr
from typing import List
from dataclasses import dataclass, field


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


def heap_pop_target(heap: List[PItem], target_idx: int) -> PItem:
    """pop target element of min-heap
    NOTE: this is pretty much the same procedure
        as pop --> heapq might support eventually

    Args:
        heap (List[PItem]): min-heap
            smallest values are prioritized
        target_idx (int): index to be popped
            (deleted and returned)

    Returns:
        PItem: popped item
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

    #
    # System for synchronizing segment boundaries with underlying heap
    #   > append: adds to heap, segment bounds remain constant
    #       desyncs ~ bounds point to subset of heap
    #   > purge: (called by append) ensures heap size <= buffer size
    #       maintains bounds ~ heap relationship
    #   > pull/sampling: pops <=N elements off the heap
    #       we've implemented logic to ensure bounds ~ heap relationship
    #       preserved == if sample popped from segment --> segment
    #       bounds adjusted accordingly
    #   > recalculating segment bounds: exactly syncs
    # Result of this sync: lowest <= segment_update_interval probability
    #       points won't be accessible

    def __init__(self, rng: npr.Generator, buffer_size: int,
                 batch_size: int,
                 segment_update_interval: int):
        # batch_size = number of samples to pull at a time
        # cumulative distro is broken up into
        #   [batch_size] segments of equal sum(probability)
        #   segments are tiled across underlying heapq array
        #       --> earlier segments will have fewer elements
        # segment_update_interval = number of append steps
        #       between segment size calculations
        self.rng = rng
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.segment_update_interval = segment_update_interval
        self._segment_bounds = None
        self._hpq = []
        self._segup_count = 0

    def _purge(self):
        num_del = max(0, len(self._hpq) - self.buffer_size)
        for _ in range(num_del):
            self._hpq.pop()

    def append(self, sample_prob: float, new_dat: List):
        # sample_prob = sampling probability
        #   will produce odd results if non-positive
        # NOTE: negate sampling prob -->
        #   highest prob will have lowest prority scores
        #   = pulled first
        # NOTE: if segments are not recalculated -->
        #   there will be 1 additional low-probability
        #   datapoint that cannot be sampled
        v = PItem(priority=-1. * sample_prob, item=new_dat)
        heapq.heappush(self._hpq, v)
        self._purge()
        self._segup_count += 1
        if self._segup_count >= self.segment_update_interval:
            self._calculate_segment_bounds()
            self._segup_count = 0  # reset counter

    def _calculate_segment_bounds(self):
        # safe and slow segment bound calculator
        # ASSUMES: items priorities are positive
        # NOTE: if heap size < batch_size --> doesn't calculate
        # NOTE: this algorithm is SLOOOW (it's O(N) loop in native python)
        # > calc sum = Z
        #       > from left to right
        #           > select items till sum(select items) >= (Z / segments)
        #           > update Z: Z -= sum(select items)
        #           > update segments: segments -= 1
        #
        # when there are a few super extreme values (Ex: sum(item) >= (Z/segments))
        #   this method guarantees that segments do NOT overlap
        #
        # when segment values are small relative to Z
        #   this method will find non-overlapping segments of approx= sum
        #
        # Proof? after k segment removals -->
        #   there is Z_{k-1} remaining sum and (N - k) remaining segments
        #   > ratio for Z_{k-1}: ratio = Z_{k-1} / (N - k)
        #   > Z_k = Z_{k-1} - (Z_{k-1} / (N - k)) = Z_{k-1} * (N - k - 1) / (N - k)
        #   > ratio for Z_k: ratio = Z_k / (N - k - 1) = Z_{k-1} / (N - k)
        #                                   = ratio for Z_{k-1}
        #   --> thus, item sum (which is equal to the ratio) for kth segment
        #           = item sum for (k - 1) segment --> induction
        if len(self._hpq) < self.batch_size:
            self._segment_bounds = None
        Z = np.sum([vi.priority * -1. for vi in self._hpq])
        num_seg = self.batch_size
        seg_bounds = [0]
        run_sum = 0.
        for i, v in enumerate(self._hpq):
            # TODO: early stopping in case we've stepped too far
            if len(seg_bounds) >= (self.batch_size + 1):
                break
            vp = v.priority * -1.0
            run_sum += vp
            if run_sum >= Z / num_seg:
                # new segment
                seg_bounds.append(i + 1)
                Z -= run_sum
                num_seg -= 1
                run_sum = 0.
        if len(seg_bounds) < self.batch_size + 1:
            seg_bounds.append(len(self._hpq))
        self._segment_bounds = seg_bounds

    def pull_samples(self):
        """Pull samples
            uses stratfied sampling system ~ pops 1 sample from each segment
            samples are removed from underlying heap

        Returns:
            List[Tuple[float, List]]: samples stored as (probability, item)
                pairs
            List[int]: length of each segment
                NOTE: requires num_sample fed in >= max(update_inteval, batch_size)
                    if requirement not met, returns None, None
        """
        if self._segment_bounds is None:
            return None, None

        # segment lengths are the true sampling probabilities
        # NOTE: should be calculated before popping occurs!
        seg_lengths = [s2 - s1 for s1, s2 in zip(self._segment_bounds[:-1],
                                                 self._segment_bounds[1:])
                       if (s2 - s1) > 0]

        ret = []
        for i in range(len(self._segment_bounds) - 1):
            # ignore empty segments:
            if self._segment_bounds[i+1] <= self._segment_bounds[i]:
                continue
            ind = self.rng.integers(self._segment_bounds[i],
                                    self._segment_bounds[i + 1])
            v = heap_pop_target(self._hpq, ind)
            ret.append((-1. * v.priority, v.item))
            # adjust segment sizes to maintain consistency
            for j in range(i+1, len(self._segment_bounds)):
                self._segment_bounds[j] = self._segment_bounds[j] - 1

        # safety check: if some segments are empty --> require segment bounds
        #   to be reset
        if np.any(np.array(self._segment_bounds[:-1]) == np.array(self._segment_bounds[1:])):
            self._calculate_segment_bounds()
        return ret, seg_lengths
