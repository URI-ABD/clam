import random
import unittest
from tempfile import TemporaryFile

import numpy as np
from scipy.spatial.distance import cdist

from pyclam import datasets, criterion
from pyclam.manifold import Manifold, Cluster

np.random.seed(42)
random.seed(42)


class TestManifold(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data, cls.labels = datasets.random(n=1000, dimensions=3)
        cls.manifold = Manifold(cls.data, 'euclidean').build(criterion.MaxDepth(8))
        return

    def test_init(self):
        m = Manifold(self.data, 'euclidean')
        self.assertEqual(1, len(m.layers))

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
        other = Manifold(self.data, 'euclidean', argpoints=0.2).build(criterion.MaxDepth(10))
        self.assertNotEqual(self.manifold, other)
        self.assertEqual(other, other)

        other = Manifold(self.data, 'cosine')
        self.assertNotEqual(self.manifold, other)
        return

    def test_iter(self):
        self.assertListEqual(self.manifold.layers, list(iter(self.manifold)))
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
        distances = [(p, d) for p, d in zip(
            range(self.data.shape[0]),
            cdist(np.asarray([point]), self.data, self.manifold.metric)[0],
        )]

        for radius in [0.25, 0.5, 1.0, 2.0, 5.0]:
            naive_results = {(p, d) for p, d in distances if d <= radius}
            results = self.manifold.find_points(point, radius)
            self.assertSetEqual(naive_results, set(results))

        return

    def test_find_clusters(self):
        self.manifold.build_tree()
        self.assertEqual(1, len(self.manifold.find_clusters(self.data[0], radius=0.0, depth=-1)))
        return

    def test_build(self):
        m = Manifold(self.data, 'euclidean').build(criterion.MaxDepth(1))
        self.assertEqual(2, len(m.layers))
        m.build(criterion.MaxDepth(2))
        self.assertEqual(3, len(m.layers))
        self.assertEqual(len(self.data), m.graph.population)
        return

    def test_build_tree(self):
        m = Manifold(self.data, 'euclidean')
        self.assertEqual(1, len(m.layers))

        m.build_tree(criterion.AddLevels(2))
        self.assertEqual(3, len(m.layers))

        # MaxDepth shouldn't do anything in build_tree if we're beyond that depth already.
        m.build_tree(criterion.MaxDepth(1))
        self.assertEqual(3, len(m.layers))

        m.build_tree()
        self.assertEqual(len(self.data), m.layers[-1].cardinality)
        return

    def test_ancestry(self):
        name = '0101'
        lineage = self.manifold.ancestry(name)
        [self.assertEqual(name[:len(l.name)], l.name) for i, l in enumerate(lineage)]
        lineage = self.manifold.ancestry(lineage[-1])
        [self.assertEqual(name[:len(l.name)], l.name) for i, l in enumerate(lineage)]
        return

    def test_select(self):
        cluster = None
        for cluster in self.manifold.layers[-1]:
            self.assertIsInstance(self.manifold.select(cluster.name), Cluster)
        else:
            with self.assertRaises(ValueError):
                self.manifold.select(cluster.name + '01')
            with self.assertRaises(ValueError):
                self.manifold.select(cluster.name + '01110110')
        return

    def test_optimal_in_graph(self):
        data, labels = datasets.bullseye()
        manifold = Manifold(data, 'euclidean').build(criterion.MaxDepth(12))
        for cluster in manifold.graph:
            self.assertTrue(cluster.optimal)

    def test_neighbors(self):
        for dataset in [datasets.bullseye, ]:  # datasets.spiral_2d, datasets.tori, datasets.skewer, datasets.line]:
            data, labels = dataset()
            manifold = Manifold(data, 'euclidean')
            manifold.build(criterion.MaxDepth(12))

            for cluster in manifold.graph.clusters:
                potential_neighbors = [c for c in manifold.graph.clusters if c.name != cluster.name]
                argcenters = [c.argmedoid for c in potential_neighbors]
                distances = list(cluster.distance_from(argcenters))
                radii = [cluster.radius + c.radius for c in potential_neighbors]
                true_neighbors = {c: d for c, d, r in zip(potential_neighbors, distances, radii) if d <= r}
                neighbors = {edge.neighbor: edge.distance for edge in manifold.graph.edges[cluster]}

                extras = set(neighbors.keys()) - set(true_neighbors.keys())
                self.assertEqual(0, len(extras), msg=f'got extra neighbors: optimal, true {len(true_neighbors)}, actual {len(neighbors)}\n'
                                                     + "\n".join([f"{c.name}, {cluster.radius + c.radius:.6f}" for c in extras]))

                missed = set(true_neighbors.keys()) - set(neighbors.keys())
                self.assertEqual(0, len(missed), msg=f'missed some neighbors: optimal, true {len(true_neighbors)}, actual {len(neighbors)}\n'
                                                     + '\n'.join([f'{c.name}, {cluster.radius + c.radius:.6f}' for c in missed]))
            return

    def test_dump(self):
        with TemporaryFile() as fp:
            self.manifold.dump(fp)
        return

    def test_load(self):
        original = self.manifold
        with TemporaryFile() as fp:
            original.dump(fp)
            fp.seek(0)
            loaded = Manifold.load(fp, self.data)
        self.assertEqual(original, loaded)
        self.assertEqual(set(original.layers[-1]), set(loaded.layers[-1]))
        self.assertEqual(original.graph, loaded.graph)

        for layer in loaded.layers:
            for cluster in layer:
                self.assertIn('radius', cluster.cache)
                self.assertIn('argradius', cluster.cache)
                self.assertIn('argsamples', cluster.cache)
                self.assertIn('argmedoid', cluster.cache)
                self.assertIn('optimal', cluster.cache)
                self.assertIn('local_fractal_dimension', cluster.cache)
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
        points = sorted([(d, p) for p, d in zip(range(data.shape[0]),
                                                cdist(np.asarray([point]), data, 'euclidean')[0])])

        m = Manifold(data, 'euclidean')
        m.build_tree(criterion.MinPoints(10), criterion.MaxDepth(10))

        ks = list(range(10))
        ks.extend(range(10, data.shape[0], 1000))
        for k in ks:
            naive_results = {p for d, p in points[:k]}
            results = m.find_knn(point, k)
            self.assertEqual(k, len(results))
            self.assertSetEqual(naive_results, {p for p, _ in results})
