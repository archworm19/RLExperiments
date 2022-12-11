"""Test basic replay buffers"""
import numpy.random as npr
from unittest import TestCase
from replay_buffers.replay_buffs import MemoryBuffer


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


if __name__ == "__main__":
    T = TestMemoryBuffer()
    T.setUp()
    T.test_add_and_purge()
