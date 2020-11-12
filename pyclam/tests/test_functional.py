import unittest

import numpy as np

from pyclam import Manifold


# TODO: Should we scrap this file?
class TestManifoldFunctional(unittest.TestCase):
    def test_random_no_limits(self):
        # We begin by getting some data and building with no constraints.
        data = np.random.randn(100, 3)
        manifold = Manifold(data, 'euclidean').build()

        for layer in manifold.layers:  # each layer must have the same number of points
            self.assertEqual(data.shape[0], layer.population, f'expected {data.shape[0]} points in the graph. '
                                                              f'Got {layer.cardinality} instead.')

        for leaf in manifold.layers[-1]:  # With no constraints, clusters should be singletons.
            self.assertEqual(1, leaf.cardinality, f'expected each leaf to be a singleton. '
                                                  f'Got leaf {str(leaf)} with {leaf.cardinality} points')
        return

    # TODO: Is this test even needed here?
    # def test_random_large(self):
    #     data = np.random.randn(1000, 3)
    #     manifold = Manifold(data, 'euclidean').build(
    #         criterion.MaxDepth(10),
    #         criterion.LFDRange(60, 50),
    #     )
    #     for _ in range(10):
    #         point = int(np.random.choice(3))
    #         linear_results = linear_search(data[point], 0.5, data, manifold.metric)
    #         # self.assertEqual(len(linear_results), len(manifold.find_points(data[point], 0.5)))
    #     return

    def test_all_same(self):
        # A bit simpler, every point is the same.
        data = np.ones((1000, 3))
        manifold = Manifold(data, 'euclidean').build()
        # There should only ever be one cluster here.
        self.assertEqual(1, len(manifold.layers))
        manifold.build_tree()
        # Even after explicit deepen calls.
        self.assertEqual(1, len(manifold.layers))
        return

    def test_two_points_with_dups(self):
        # Here we have two distinct clusters.
        data = np.concatenate([np.ones((500, 2)) * -2, np.ones((500, 2)) * 2])
        manifold = Manifold(data, 'euclidean').build()
        # We expect building to stop with two clusters.
        self.assertEqual(2, manifold.layers[1].cardinality, f'Expected 2 clusters in the layer-graph at depth 1. '
                                                            f'Got {manifold.layers[1].cardinality} instead.')
        self.assertEqual(2, manifold.graphs[0].cardinality, f'Expected 2 clusters in the optimal graph. '
                                                            f'Got {manifold.graphs[0].cardinality} instead.')
        return
