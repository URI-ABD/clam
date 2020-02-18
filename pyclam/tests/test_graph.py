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
        Graph(*[c for c in self.manifold.graphs[-1]])
        with self.assertRaises(AssertionError):
            Graph('1')
        with self.assertRaises(AssertionError):
            Graph('1', '2', '3')
        return

    def test_eq(self):
        self.assertEqual(self.manifold.graphs[0], self.manifold.graphs[0])
        for left, right in combinations(self.manifold.graphs, 2):
            self.assertNotEqual(left, right)
        return

    def test_iter(self):
        clusters = list(self.manifold.graphs[1])
        self.assertEqual(2, len(clusters))
        self.assertIn(clusters[0], self.manifold.select('').children)
        return

    def test_metric(self):
        self.assertEqual(self.manifold.metric, self.manifold.graphs[0].metric)

    def test_len(self):
        clusters = list(self.manifold.graphs[-1])
        g = Graph(*clusters)
        self.assertEqual(len(clusters), g.cardinality)
        return

    def test_population(self):
        self.assertEqual(len(self.manifold.argpoints), self.manifold.graphs[-1].population)
        return

    def test_str(self):
        self.assertEqual(self.manifold.graphs[-1].cardinality, len(str(self.manifold.graphs[-1]).split(';')))
        return

    def test_repr(self):
        self.assertIsInstance(repr(self.manifold.graphs[-1]), str)
        return

    def test_contains(self):
        root = self.manifold.select('')
        self.assertIn(root, self.manifold.graphs[0])
        self.assertNotIn(root, self.manifold.graphs[1])

        clusters = self.manifold.find_clusters(root.medoid, root.radius, depth=5)
        for c in clusters:
            self.assertIn(c, self.manifold.graphs[5])
        return

    def test_build_edges(self):
        self.manifold.graphs[-1].build_edges()
        self.manifold.build_graph(5)
        with self.assertRaises(ValueError):
            self.manifold.build_graph(25)
        return

    def test_manifold(self):
        self.assertEqual(self.manifold, self.manifold.graphs[0].manifold)
        return

    def test_depth(self):
        for i, g in enumerate(self.manifold.graphs):
            self.assertEqual(i, g.depth)
        return

    def test_edges(self):
        v = self.manifold.graphs[-1]
        e = self.manifold.graphs[-1].edges
        self.assertGreaterEqual(len(e), 0)
        self.assertLessEqual(len(e), v.cardinality * (v.cardinality - 1) / 2)
        return

    def test_subgraphs(self):
        subgraphs = self.manifold.graphs[-1].subgraphs
        [self.assertIsInstance(g, Graph) for g in subgraphs]
        self.assertEqual(sum(g.cardinality for g in subgraphs), self.manifold.graphs[-1].cardinality)
        return

    def test_subgraph(self):
        root = self.manifold.select('')
        self.assertTrue(self.manifold.graphs[0].subgraph(root))
        with self.assertRaises(ValueError):
            self.manifold.graphs[1].subgraph(root)
        return

    def test_clear_cache(self):
        self.manifold.graphs[-1].clear_cache()
        _ = self.manifold.graphs[-1].edges
        self.assertIn('_edges', self.manifold.graphs[-1].__dict__)
        self.manifold.graphs[-1].clear_cache()
        self.assertNotIn('_edges', self.manifold.graphs[-1].__dict__)
        return

    def test_bft(self):
        g = self.manifold.graphs[-1].bft(next(iter(self.manifold.graphs[-1])))
        self.assertGreater(len(g), 0)
        self.assertLessEqual(len(g), self.manifold.graphs[-1].cardinality)

        # TODO: Should this work?
        g = self.manifold.graphs[-1].bft(next(iter(self.manifold.graphs[-2])))
        self.assertGreater(len(g), 0)
        self.assertLessEqual(len(g), self.manifold.graphs[-1].cardinality)
        return

    def test_dft(self):
        g = self.manifold.graphs[-1].dft(next(iter(self.manifold.graphs[-1])))
        self.assertGreater(len(g), 0)
        self.assertLessEqual(len(g), self.manifold.graphs[-1].cardinality)
        return

    def test_random_walks(self):
        # manifold = self.manifold.build(MaxDepth(10))
        g = self.manifold.graphs[-1]
        results = g.random_walks(list(g.clusters), 250)
        print(sum(results.values()))
        self.assertGreater(len(results), 0)
        [self.assertGreaterEqual(v, 0) for k, v in results.items()]
        self.manifold.build(MaxDepth(5))
        self.manifold.build_tree(MaxDepth(6))
        g = self.manifold.graphs[-1]
        results = g.random_walks(list(g.clusters), 100)
        print(sum(results.values()))
        self.assertGreater(len(results), 0)
        [self.assertGreaterEqual(v, 0) for k, v in results.items()]
        return
