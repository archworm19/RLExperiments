"""Test forward model prediction error"""
import numpy as np
import tensorflow as tf
from unittest import TestCase
from frameworks.intrinsic_motivation.forward_model_error import _combine_2dims


class TestUtils(TestCase):

    def test_combine2(self):
        dim0 = 5
        dim1 = 4
        N = 3
        x1 = np.ones((dim0, dim1, N))
        x2 = np.ones((dim0, dim1))
        for i in range(dim0):
            for j in range(dim1):
                x1[i,j] = i * dim1 + j
                x2[i,j] = i * dim1 + j
        [y1, y2] = _combine_2dims([tf.constant(x1), tf.constant(x2)])
        v = np.arange(int(dim0 * dim1))
        self.assertTrue(np.all(y2.numpy() == v))
        self.assertTrue(np.all(y1.numpy() == np.tile(v[:,None], (1, N))))
        self.assertTrue(np.all(np.reshape(y1.numpy(), (dim0, dim1, N)) == x1))


if __name__ == "__main__":
    T = TestUtils()
    T.test_combine2()
