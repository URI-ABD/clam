import random
import unittest
from tempfile import TemporaryFile

import numpy as np
from scipy.spatial.distance import cdist

from chess import datasets, criterion
from chess.manifold import Manifold, Cluster

np.random.seed(42)
random.seed(42)


class TestManifold(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data, cls.labels = datasets.random(n=1000, dimensions=3)
        cls.manifold = Manifold(cls.data, 'euclidean').build(criterion.MaxDepth(12))
        return

    def test_init(self):
        m = Manifold(self.data, 'euclidean')
        self.assertEqual(1, len(m.graphs))

        m = Manifold(self.data, 'euclidean', [1, 2, 3])
        self.assertListEqual([1, 2, 3], m.argpoints)

        fraction = 0.2
        m = Manifold(self.data, 'euclidean', fraction)
        self.assertEqual(int(len(self.data) * fraction), len(m.argpoints))

        with self.assertRaises(ValueError):
            # noinspection PyTypeChecker
            Manifold(self.data, 'euclidean', ['a', 'b', 'c'])

        with self.assertRaises(ValueError):
            # noinspection PyTypeChecker
            Manifold(self.data, 'euclidean', 'apples')
        return

    def test_eq(self):
        self.assertEqual(self.manifold, self.manifold)
        other = Manifold(self.data, 'euclidean', argpoints=0.2)
        self.assertNotEqual(self.manifold, other)
        other.build()
        self.assertNotEqual(self.manifold, other)
        self.assertEqual(other, other)
        return

    def test_getitem(self):
        self.assertEqual(self.manifold.graphs[0], self.manifold[0])
        with self.assertRaises(IndexError):
            _ = self.manifold[100]
        return

    def test_subgraph(self):
        g = self.manifold.graphs[-1]
        c = next(iter(g))
        self.assertIn(self.manifold.subgraph(c), g.subgraphs)
        self.assertIn(self.manifold.subgraph(c.name), g.subgraphs)
        return

    def test_graph(self):
        g = self.manifold.graphs[-1]
        c = next(iter(g))
        self.assertEqual(g, self.manifold.graph(c))
        self.assertEqual(g, self.manifold.graph(c.name))
        return

    def test_iter(self):
        self.assertListEqual(self.manifold.graphs, list(iter(self.manifold)))
        return

    def test_str(self):
        self.assertIsInstance(str(self.manifold), str)
        return

    def test_repr(self):
        self.assertIsInstance(repr(self.manifold), str)
        return

    def test_find_points(self):
        self.assertEqual(1, len(self.manifold.find_points(self.data[0], radius=0.0)))
        self.assertLessEqual(1, len(self.manifold.find_points(self.data[0], radius=1.0)))

        point = self.data[0]
        distances = [(p, d) for p, d in zip(range(self.data.shape[0]), cdist(np.asarray([point]), self.data, self.manifold.metric)[0])]

        for radius in [0.25, 0.5, 1.0, 2.0, 5.0]:
            naive_results = {p: d for p, d in distances if d <= radius}
            results = self.manifold.find_points(point, radius)
            self.assertDictEqual(naive_results, results)

        return

    def test_find_clusters(self):
        self.manifold.build_tree()
        self.assertEqual(1, len(self.manifold.find_clusters(self.data[0], radius=0.0, depth=-1)))
        return

    def test_build(self):
        m = Manifold(self.data, 'euclidean').build(criterion.MaxDepth(1))
        self.assertEqual(2, len(m.graphs))
        m.build(criterion.MaxDepth(2))
        self.assertEqual(3, len(m.graphs))
        m.build()
        self.assertEqual(len(self.data), len(m.graphs[-1]))
        return

    def test_build_tree(self):
        m = Manifold(self.data, 'euclidean')
        self.assertEqual(1, len(m.graphs))

        m.build_tree(criterion.AddLevels(2))
        self.assertEqual(3, len(m.graphs))

        # MaxDepth shouldn't do anything in build_tree if we're beyond that depth already.
        m.build_tree(criterion.MaxDepth(1))
        self.assertEqual(3, len(m.graphs))

        m.build_tree()
        self.assertEqual(len(self.data), len(m.graphs[-1]))
        return

    def test_select(self):
        cluster = None
        for cluster in self.manifold.graphs[-1]:
            self.assertIsInstance(self.manifold.select(cluster.name), Cluster)
        else:
            with self.assertRaises(ValueError):
                self.manifold.select(cluster.name + '1')
        return

    def test_dump(self):
        with TemporaryFile() as fp:
            self.manifold.dump(fp)
        return

    def test_load(self):
        original = Manifold(self.data, 'euclidean').build(criterion.MinPoints(10))
        with TemporaryFile() as fp:
            original.dump(fp)
            fp.seek(0)
            loaded = Manifold.load(fp, self.data)
        self.assertEqual(original, loaded)
        self.assertEqual(original[0], loaded[0])

        for graph in loaded.graphs:
            for cluster in graph.clusters:
                self.assertIn('_radius', cluster.__dict__)
                self.assertIn('_argradius', cluster.__dict__)
                self.assertIn('_argsamples', cluster.__dict__)
                self.assertIn('_argmedoid', cluster.__dict__)
                self.assertIn('_local_fractal_dimension', cluster.__dict__)
        return

    def test_partition_backends(self):
        data = datasets.random(n=100, dimensions=5)[0]
        m_single = Manifold(data, 'euclidean')._partition_single([criterion.MaxDepth(5)])
        m_thread = Manifold(data, 'euclidean')._partition_threaded([criterion.MaxDepth(5)])
        self.assertEqual(m_single, m_thread)
        return

    def test_find_knn(self):
        data = datasets.bullseye()[0]
        point = data[0]
        points = sorted([(d, p) for p, d in zip(range(data.shape[0]), cdist(np.asarray([point]), data, 'euclidean')[0])])

        m = Manifold(data, 'euclidean')
        m.build_tree(criterion.MinPoints(10), criterion.MaxDepth(10))

        ks = list(range(100))
        ks.extend(range(100, data.shape[0], 100))
        for k in ks:
            naive_results = {p: d for d, p in points[:k]}
            results = m.find_knn(point, k)
            self.assertEqual(k, len(results.keys()))
            self.assertSetEqual(set(naive_results.keys()), set(results.keys()))
