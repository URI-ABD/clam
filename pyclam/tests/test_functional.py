import unittest

import numpy as np

from chess import Manifold
from chess.criterion import MinPoints
from tests.utils import linear_search


class TestManifoldFunctional(unittest.TestCase):
    def test_random_no_limits(self):
        # We begin by getting some data and building with no constraints.
        data = np.random.randn(100, 3)
        m = Manifold(data, 'euclidean')
        m.build()
        # With no constraints, clusters should be singletons.
        self.assertEqual(data.shape[0], len(m.graphs[-1]))
        self.assertEqual(1, len(m.find_clusters(data[0], 0., -1)))
        self.assertEqual(1, len(m.find_points(data[0], 0.)))
        return

    def test_random_large(self):
        data = np.random.randn(1000, 3)
        m = Manifold(data, 'euclidean')
        m.build(MinPoints(10))
        for _ in range(10):
            point = int(np.random.choice(3))
            linear_results = linear_search(data[point], 0.5, data, m.metric)
            self.assertEqual(len(linear_results), len(m.find_points(data[point], 0.5)))
        return

    def test_all_same(self):
        # A bit simpler, every point is the same.
        data = np.ones((1000, 3))
        m = Manifold(data, 'euclidean')
        m.build()
        # There should only ever be one cluster here.
        self.assertEqual(1, len(m.graphs))
        m.build_tree()
        # Even after explicit deepen calls.
        self.assertEqual(1, len(m.graphs))
        self.assertEqual(1, len(m.find_clusters(data[0], 0.0, -1)))
        # And, we should get all 1000 points back for any of the data.
        self.assertEqual(1000, len(m.find_points(data[0], 0.0)))
        return

    def test_two_points_with_dups(self):
        # Here we have two distinct clusters.
        data = np.concatenate([np.ones((500, 2)) * -2, np.ones((500, 2)) * 2])
        m = Manifold(data, 'euclidean')
        # We expect building to stop with two clusters.
        m.build()
        self.assertEqual(2, len(m.graphs[-1]), f'Expected 2 clusters, got {len(m.graphs[-1])}')
        return
