import unittest

import numpy as np

from pyclam.manifold import Manifold, Distance


class TestDistance(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = np.asarray(([0, 0], [1, 1], [2, 2], [3, 3], [4, 4]), dtype=np.float64)
        cls.metric = 'cityblock'

        cls.manifold: Manifold = Manifold(cls.data, cls.metric)
        cls.distance: Distance = cls.manifold.distance
        return

    def test_errors(self):
        self.assertRaises(IndexError, self.distance, 1, 5)
        self.assertRaises(AssertionError, self.distance, 1, [])

    def test_call(self):
        i, j = [0, 1], [0, 1, 2, 3]

        expected = np.asarray([[0, 2, 4, 6], [2, 0, 2, 4]])
        self.assertAlmostEqual((expected - self.distance(i, j)).sum(), 0., places=10)

        i, j = [0, 1], np.asarray([5, 5], dtype=np.float64)
        expected = np.asarray([10, 8], dtype=np.float64)
        self.assertAlmostEqual((expected - self.distance(i, j)).sum(), 0., places=10)
        self.assertAlmostEqual((expected - self.distance(j, i)).sum(), 0., places=10)

        print(self.distance.history)
