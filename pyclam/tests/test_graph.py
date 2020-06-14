import unittest
from itertools import combinations

import numpy as np

from pyclam import datasets
from pyclam.criterion import MaxDepth
from pyclam.manifold import Manifold, Graph


class TestGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        np.random.seed(42)
        cls.data, _ = datasets.bullseye()
        cls.manifold = Manifold(cls.data, 'euclidean')
        cls.manifold.build(MaxDepth(10))
        return

    def test_init(self):
        Graph(*[c for c in self.manifold.graph])
        with self.assertRaises(AssertionError):
            Graph('1')
        with self.assertRaises(AssertionError):
            Graph('1', '2', '3')
        return

    def test_eq(self):
        self.assertEqual(self.manifold.layers[0], self.manifold.layers[0])
        for left, right in combinations(self.manifold.layers, 2):
            self.assertNotEqual(left, right)
        return

    def test_iter(self):
        clusters = list(self.manifold.layers[1])
        self.assertEqual(2, len(clusters))
        self.assertIn(clusters[0], self.manifold.select('').children)
        return

    def test_metric(self):
        self.assertEqual(self.manifold.metric, self.manifold.graph.metric)

    def test_population(self):
        self.assertEqual(len(self.manifold.argpoints), self.manifold.graph.population)
        return

    def test_str(self):
        self.assertEqual(self.manifold.graph.cardinality, len(str(self.manifold.graph).split(', ')))
        return

    def test_repr(self):
        self.assertIsInstance(repr(self.manifold.graph), str)
        return

    def test_contains(self):
        root = self.manifold.select('')
        self.assertNotIn(root, self.manifold.graph)
        return

    def test_edges(self):
        v = self.manifold.graph
        e = self.manifold.graph.edges
        self.assertGreaterEqual(len(e), 0)
        self.assertLessEqual(len(e), v.cardinality * (v.cardinality - 1) / 2)
        return

    def test_subgraphs(self):
        subgraphs = self.manifold.graph.subgraphs
        [self.assertIsInstance(g, Graph) for g in subgraphs]
        self.assertEqual(sum(g.cardinality for g in subgraphs), self.manifold.graph.cardinality)
        return

    def test_clear_cache(self):
        self.manifold.graph.clear_cache()
        _ = self.manifold.graph.edges
        self.assertIn('_edges', self.manifold.graph.__dict__)
        self.manifold.graph.clear_cache()
        self.assertNotIn('_edges', self.manifold.graph.__dict__)
        return

    def test_bft(self):
        g = self.manifold.graph.bft(next(iter(self.manifold.graph)))
        self.assertGreater(len(g), 0)
        self.assertLessEqual(len(g), self.manifold.graph.cardinality)
        return

    def test_dft(self):
        g = self.manifold.graph.dft(next(iter(self.manifold.graph)))
        self.assertGreater(len(g), 0)
        self.assertLessEqual(len(g), self.manifold.graph.cardinality)
        return

    def test_random_walks(self):
        # manifold = self.manifold.build(MaxDepth(10))
        g = self.manifold.graph
        results = g.random_walks(list(g.clusters), 250)
        print(sum(results.values()))
        self.assertGreater(len(results), 0)
        [self.assertGreaterEqual(v, 0) for k, v in results.items()]
        return
