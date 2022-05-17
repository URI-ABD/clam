import random
import unittest
from typing import Dict, List, Tuple, Set

import numpy as np
from scipy.spatial.distance import cdist

from pyclam import datasets, Search, Cluster

np.random.seed(42), random.seed(42)


class TestSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data, _ = datasets.bullseye(n=1000, num_rings=3)
        cls.query, cls.metric = cls.data[0], 'euclidean'
        cls.distances: Dict[int, float] = {
            point: distance
            for point, distance in zip(
                range(cls.data.shape[0]),
                cdist(np.asarray([cls.query]), cls.data, cls.metric)[0]
            )
        }
        return

    def test_init(self):
        search = Search(self.data, 'euclidean')
        self.assertEqual(0, search.depth, f'tree depth should be 0')

        search = search.build(max_depth=5)
        self.assertGreaterEqual(5, search.depth, f'tree depth should be 5')

        search = search.build(max_depth=8)
        self.assertGreaterEqual(8, search.depth, f'tree depth should be 8')
        return

    def test_rnn(self):
        search = Search(self.data, self.metric).build(max_depth=10)

        self.assertEqual(1, len(search.rnn(self.query, 0)))
        self.assertLessEqual(1, len(search.rnn(self.query, 1)))

        for radius in [0.25, 0.5, 1.0, 2.0, 5.0]:
            naive_results: Dict[int, float] = {point: distance for point, distance in self.distances.items() if distance <= radius}
            rnn_results: List[int] = list(search.rnn(self.query, radius).keys())
            self.assertEqual(len(naive_results), len(rnn_results), f'expected the same number of results from naive and rnn searches.')
            self.assertSetEqual(set(naive_results.keys()), set(rnn_results), f'expected the same set of results from naive and rnn searches.')

            sorted_naive = list(sorted([(d, p) for p, d in naive_results.items()]))
            naive_points = self.data[[p for _, p in sorted_naive]]
            rnn_points = search.rnn_points(self.query, radius)
            self.assertEqual(naive_points.shape, rnn_points.shape, f'found mismatch between shapes of points returned')
            for left, right in zip(naive_points, rnn_points):
                self.assertEqual(list(left), list(right), f'found mismatch between arrays of points returned')
        return

    def test_knn(self):
        search = Search(self.data, self.metric).build(max_depth=10)
        points: List[Tuple[float, int]] = list(sorted([(distance, point) for point, distance in self.distances.items()]))
        points: List[int] = [p for _, p in points]

        ks: List[int] = list(range(1, 10))
        ks.extend(range(10, self.data.shape[0], 1000))
        for k in ks:
            naive_results: List[int] = points[:k]
            knn_results: List[int] = list(search.knn(self.query, k).keys())
            self.assertEqual(len(naive_results), len(knn_results), f'expected the same number of results from naive and knn searches.')
            self.assertSetEqual(set(naive_results), set(knn_results), f'expected the same set of results from naive and knn searches.')

            naive_points = self.data[naive_results]
            knn_points = search.knn_points(self.query, k)
            self.assertEqual(naive_points.shape, knn_points.shape, f'found mismatch between shapes of points returned')
            for left, right in zip(naive_points, knn_points):
                self.assertEqual(list(left), list(right), f'found mismatch between arrays of points returned')
        return

    def test_tree_search_history(self):
        search = Search(self.data, self.metric).build(max_depth=10)
        radius: float = search.root.radius / 10
        history, hits = search.tree_search_history(self.query, radius)

        for hit in hits:
            self.assertTrue(hit in history, f'The hit {str(hit)} was not found in history.')
        self.assertLessEqual(len(hits), len(history), f'history should have at least a many members as hits.')

        for cluster in history:
            if cluster not in hits:
                self.assertGreater(len(cluster.children), 0, f'A non-hit member of history must have had children.')

        depths: Set[int] = {cluster.depth for cluster in history}
        for d in range(len(depths)):
            self.assertIn(d, depths, f'history should contain clusters from every depth. Did not contain: {d}')
