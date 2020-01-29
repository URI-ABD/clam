import unittest
from itertools import combinations

import numpy as np

from chess.criterion import MaxDepth
from chess.manifold import Manifold, Graph


class TestGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = np.random.randn(1000, 10)
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
        for l, r in combinations(self.manifold.graphs, 2):
            self.assertNotEqual(l, r)
        return

    def test_iter(self):
        clusters = list(self.manifold.graphs[1])
        self.assertEqual(2, len(clusters))
        self.assertIn(clusters[0], self.manifold.select('').children)
        return

    def test_len(self):
        clusters = list(self.manifold.graphs[-1])
        g = Graph(*clusters)
        self.assertEqual(len(clusters), len(g))
        return

    def test_str(self):
        self.assertEqual(len(self.manifold.graphs[-1]), len(str(self.manifold.graphs[-1]).split(';')))
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
        self.assertLessEqual(len(e), len(v) * (len(v) - 1) / 2)
        return

    def test_subgraphs(self):
        sgs = self.manifold.graphs[-1].subgraphs
        [self.assertIsInstance(g, Graph) for g in sgs]
        self.assertEqual(sum(len(g) for g in sgs), len(self.manifold.graphs[-1]))
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
        self.assertLessEqual(len(g), len(self.manifold.graphs[-1]))

        # TODO: Should this work?
        g = self.manifold.graphs[-1].bft(next(iter(self.manifold.graphs[-2])))
        self.assertGreater(len(g), 0)
        self.assertLessEqual(len(g), len(self.manifold.graphs[-1]))
        return

    def test_dft(self):
        g = self.manifold.graphs[-1].dft(next(iter(self.manifold.graphs[-1])))
        self.assertGreater(len(g), 0)
        self.assertLessEqual(len(g), len(self.manifold.graphs[-1]))
        return

    def test_random_walk(self):
        g = self.manifold.graphs[-1]
        results = g.random_walk()
        self.assertGreater(len(results), 0)
        [self.assertGreaterEqual(v, 0) for k, v in results.items()]
        return
