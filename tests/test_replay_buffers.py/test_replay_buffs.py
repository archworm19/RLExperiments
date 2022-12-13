"""Test basic replay buffers"""
import numpy.random as npr
from unittest import TestCase
from replay_buffers.replay_buffs import MemoryBuffer, MemoryBufferNstep


class TestMemoryBuffer(TestCase):

    def setUp(self) -> None:
        rng = npr.default_rng(42)
        kz = ["A", "B"]
        self.buffer_size = 3
        self.mbuff = MemoryBuffer(kz, rng, self.buffer_size)

    def test_add_and_purge(self):
        # for test consistency
        self.mbuff.rng = npr.default_rng(42)
        for i in range(self.buffer_size):
            d = {"A": i, "B": 2*i}
            self.mbuff.append(d)
        # test randomness:
        v1 = self.mbuff.pull_sample(3)
        v2 = self.mbuff.pull_sample(3)
        # note how they're different
        self.assertEqual(v1, {'A': [0, 2, 1], 'B': [0, 4, 2]})
        self.assertEqual(v2, {'A': [1, 1, 2], 'B': [2, 2, 4]})
        # add another:
        d100 = {"A": 100, "B": 100}
        self.mbuff.append(d100)
        v100 = self.mbuff.pull_sample(3)
        self.assertEqual(v100, {'A': [1, 100, 1], 'B': [2, 100, 2]})
        # also: 0 should be gone:
        for _ in range(10):
            vi = self.mbuff.pull_sample(3)
            self.assertTrue(0 not in vi["A"])


class TestMemBuffNStep(TestCase):

    def setUp(self) -> None:
        rng = npr.default_rng(42)
        kz = ["A", "B", "term"]
        self.buffer_size = 100
        self.N = 3
        self.mbuff = MemoryBufferNstep(self.N, kz, rng, self.buffer_size,
                                       "term")

    def _add_data(self, a_val: int, b_val: int, num: int):
        for _ in range(num - 1):
            self.mbuff.append({"A": a_val, "B": b_val, "term": False})
        self.mbuff.append({"A": a_val, "B": b_val, "term": True})


    def test_sample(self):
        # add 2 sets (after a terminal bit)
        # adds to 100 --> purge all old data
        self._add_data(-1, -1, 1)
        self._add_data(1, 2, 30)
        self._add_data(11, 21, 70)

        sample = self.mbuff.pull_sample(100)

        # shape testing:
        self.assertTrue(len(sample["A"]) == 100)
        self.assertTrue(len(sample["A"][0]) == self.N)

        # ensure no mixing of sets:
        for v in sample["A"]:
            self.assertTrue(sum(v) == 3 * 1
                            or sum(v) == 3 * 11)
        for v in sample["B"]:
            self.assertTrue(sum(v) == 3 * 2
                            or sum(v) == 3 * 21)

        # ensure termination is included
        term_end_counts, term_pre_counts = 0, 0
        for v in sample["term"]:
            term_end_counts += v[-1] * 1
            term_pre_counts += v[0] * 1
            term_pre_counts += v[1] * 1
        self.assertTrue(term_end_counts > 0)
        self.assertTrue(term_pre_counts < 1)

    def test_purge(self):
        # add 2 sets (after a terminal bit)
        # adds to 100 --> purge all old data
        self._add_data(-1, -1, 1)
        self._add_data(1, 2, 30)
        self._add_data(11, 21, 70)

        # add 30 --> purges set 1
        self._add_data(100, 200, 30)
        sample = self.mbuff.pull_sample(50)
        for v in sample["A"]:
            self.assertTrue(1 not in v)

        # add 70 --> purges set 2
        self._add_data(100, 200, 70)
        sample = self.mbuff.pull_sample(50)
        for v in sample["A"]:
            self.assertTrue(11 not in v)


if __name__ == "__main__":
    T = TestMemoryBuffer()
    T.setUp()
    T.test_add_and_purge()
    T = TestMemBuffNStep()
    T.setUp()
    T.test_sample()
    T.test_purge()
