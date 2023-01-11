"""Test priority replay buffers"""
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


if __name__ == "__main__":
    T = TestPopTarget()
    T.test_heap_sort()
    T.test_internal_pop()
