"""Test priority replay buffers"""
import numpy as np
import numpy.random as npr
import heapq
from unittest import TestCase
from replay_buffers.priority_buffs import PItem, heap_pop_target, MemoryBufferPQ


class TestPopTarget(TestCase):

    def _get_heap(self):
        items = [PItem(1., ["a", "b"]),
                 PItem(2., ["c", "d"]),
                 PItem(-100., ["z"]),
                 PItem(-50, ["opa"])]
        heap = []
        for v in items:
            heapq.heappush(heap, v)
        return heap

    def test_heap_sort(self):
        heap = self._get_heap()
        priorz = []
        for _ in range(len(heap)):
            priorz.append(heap_pop_target(heap, 0).priority)
        self.assertEqual(priorz, [-100.0, -50, 1.0, 2.0])

    def test_internal_pop(self):
        heap = self._get_heap()
        target = heap[1].priority
        self.assertEqual(heap_pop_target(heap, 1).priority, target)
        priorz = []
        for _ in range(len(heap)):
            priorz.append(heap_pop_target(heap, 0).priority)
        self.assertEqual(priorz, [-100.0, 1.0, 2.0])


class TestPQ(TestCase):

    def setUp(self) -> None:
        self.buffer_size = 1024  # some tests rely on law of large numbers
        self.batch_size = 4
        self.rng = npr.default_rng(42)
        self.PQ = MemoryBufferPQ(self.rng, self.buffer_size, self.batch_size,
                                 50)
        return super().setUp()

    def _load_data(self):
        # fill up the buffer with test data
        # priorities taken from uniform distribution
        probs = []
        for _ in range(self.buffer_size):
            probs.append(self.rng.random())
            self.PQ.append(probs[-1], [])
        return probs

    def test_append(self):
        # max element should be -1 * largest probability
        probs = self._load_data()
        self.assertAlmostEqual(np.amax(probs), -1 * self.PQ._hpq[0].priority, 4)

    def test_segment_bounds(self):
        self._load_data()
        self.PQ._calculate_segment_bounds()

        # make sure each segment contains the same probability
        hprobs = [-1. * v.priority for v in self.PQ._hpq]
        hprobs = hprobs / np.sum(hprobs)
        psums = []
        for i in range(len(self.PQ._segment_bounds) - 1):
            ind0 = self.PQ._segment_bounds[i]
            ind1 = self.PQ._segment_bounds[i + 1]
            psums.append(np.sum(hprobs[ind0:ind1]))

        for ps in psums:
            self.assertAlmostEqual(ps, 1. / self.batch_size, 2)

        # make sure the segment sizes grow monotonically
        seg_sizes = [s2 - s1 for s1, s2 in
                     zip(self.PQ._segment_bounds[:-1], self.PQ._segment_bounds[1:])]
        for i in range(1, len(seg_sizes)):
            self.assertTrue(seg_sizes[i] >= seg_sizes[i - 1])

    def test_sampling(self):
        self._load_data()
        for i in range(3):
            dat, seg_lengths = self.PQ.pull_samples()
            self.assertTrue(len(dat) == self.batch_size)
            self.assertTrue(len(seg_lengths) == self.batch_size)
            tot_length = sum(seg_lengths)
            # test the popping
            self.assertTrue(tot_length == self.buffer_size - i * self.batch_size)
            self.assertTrue((len(self.PQ._hpq) + self.batch_size) == tot_length)


class TestPQsmallbuff(TestCase):
    # PQ has some odd behavior when buffer is not full

    def setUp(self) -> None:
        self.buffer_size = 1024  # some tests rely on law of large numbers
        self.rng = npr.default_rng(42)

    def _load_data(self, PQ: MemoryBufferPQ, num_load: int):
        # fill up the buffer with test data
        # priorities taken from uniform distribution
        probs = []
        for _ in range(num_load):
            probs.append(self.rng.random())
            PQ.append(probs[-1], [])
        return probs

    def test_sync(self):
        update_interval = 50
        batch_size = 32
        PQ = MemoryBufferPQ(self.rng, self.buffer_size, batch_size,
                            update_interval)
        # load in 32 datapoints (< update interval)
        #   --> sampling is not yet possible
        self._load_data(PQ, 32)
        d, _ = PQ.pull_samples()
        self.assertIsNone(d)
        # after another 100 --> we should be good to go
        self._load_data(PQ, 100)
        d, _ = PQ.pull_samples()
        self.assertIsNotNone(d)


if __name__ == "__main__":
    T = TestPopTarget()
    T.test_heap_sort()
    T.test_internal_pop()
    T = TestPQ()
    T.setUp()
    T.test_append()
    T.test_segment_bounds()
    T.test_sampling()
    T = TestPQsmallbuff()
    T.setUp()
    T.test_sync()
