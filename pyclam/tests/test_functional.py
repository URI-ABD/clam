import unittest

import numpy as np

from pyclam import Manifold, criterion
from pyclam.tests.utils import linear_search


class TestManifoldFunctional(unittest.TestCase):
    def test_random_no_limits(self):
        # We begin by getting some data and building with no constraints.
        data = np.random.randn(100, 3)
        manifold = Manifold(data, 'euclidean').build()
        # With no constraints, clusters should be singletons.
        self.assertEqual(data.shape[0], manifold.graph.population)
        self.assertEqual(1, len(manifold.find_clusters(data[0], 0., -1)))
        self.assertEqual(1, len(manifold.find_points(data[0], 0.)))
        self.assertEqual(data.shape[0], manifold.layers[-1].cardinality)
        return

    def test_random_large(self):
        data = np.random.randn(1000, 3)
        manifold = Manifold(data, 'euclidean').build(
            criterion.MaxDepth(10),
            criterion.LFDRange(60, 50),
        )
        for _ in range(10):
            point = int(np.random.choice(3))
            linear_results = linear_search(data[point], 0.5, data, manifold.metric)
            self.assertEqual(len(linear_results), len(manifold.find_points(data[point], 0.5)))
        return

    def test_all_same(self):
        # A bit simpler, every point is the same.
        data = np.ones((1000, 3))
        manifold = Manifold(data, 'euclidean').build()
        # There should only ever be one cluster here.
        self.assertEqual(1, len(manifold.layers))
        manifold.build_tree()
        # Even after explicit deepen calls.
        self.assertEqual(1, len(manifold.layers))
        self.assertEqual(1, len(manifold.find_clusters(np.asarray([1, 1, 1]), 0.0, -1)))
        # And, we should get all 1000 points back for any of the data.
        self.assertEqual(1000, len(manifold.find_points(data[0], 0.0)))
        return

    def test_two_points_with_dups(self):
        # Here we have two distinct clusters.
        data = np.concatenate([np.ones((500, 2)) * -2, np.ones((500, 2)) * 2])
        manifold = Manifold(data, 'euclidean').build()
        # We expect building to stop with two clusters.
        self.assertEqual(2, manifold.graph.cardinality, f'Expected 2 clusters, got {manifold.graph.cardinality}')
        return
