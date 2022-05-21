import random
import tempfile
import unittest

import numpy

from pyclam import Cluster
from pyclam import criterion
from pyclam import Manifold
from pyclam.utils import synthetic_datasets


class TestManifold(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        numpy.random.seed(42), random.seed(42)

        cls.data, cls.labels = synthetic_datasets.random(n=1000, dimensions=3)
        cls.manifold = Manifold(cls.data, 'euclidean')
        cls.manifold.build(
            criterion.MaxDepth(8),
            criterion.LFDRange(60, 50),
        )
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
        other = Manifold(self.data, 'euclidean', argpoints=0.2).build(
            criterion.MaxDepth(10),
            criterion.LFDRange(60, 50),
        )
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

    def test_build(self):
        m = Manifold(self.data, 'euclidean').build(criterion.MaxDepth(1))
        self.assertEqual(2, len(m.layers))
        m.build(criterion.MaxDepth(2))
        self.assertEqual(3, len(m.layers))
        self.assertEqual(len(self.data), m.graphs[0].population)
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

    def test_neighbors(self):
        for dataset in [synthetic_datasets.bullseye, ]:  # datasets.spiral_2d, datasets.tori, datasets.skewer, datasets.line]:
            data, labels = dataset()
            manifold = Manifold(data, 'euclidean').build(
                criterion.MaxDepth(12),
                criterion.Layer(8),
            )

            for cluster in manifold.graphs[0].clusters:
                potential_neighbors: list[Cluster] = [c for c in manifold.graphs[0].clusters if c.name != cluster.name]
                argcenters: list[int] = [c.argmedoid for c in potential_neighbors]
                distances: list[float] = list(cluster.distance_from(argcenters))
                radii: list[float] = [cluster.radius + c.radius for c in potential_neighbors]
                true_neighbors = {c: d for c, d, r in zip(potential_neighbors, distances, radii) if d <= r}
                neighbors = {edge.neighbor(cluster): edge.distance for edge in manifold.graphs[0].edges_from(cluster)}

                extras = set(neighbors.keys()) - set(true_neighbors.keys())
                self.assertEqual(0, len(extras), msg=f'got extra neighbors: optimal, true {len(true_neighbors)}, actual {len(neighbors)}\n'
                                                     + "\n".join([f"{c.name}, {cluster.radius + c.radius:.6f}" for c in extras]))

                missed = set(true_neighbors.keys()) - set(neighbors.keys())
                self.assertEqual(0, len(missed), msg=f'missed some neighbors: optimal, true {len(true_neighbors)}, actual {len(neighbors)}\n'
                                                     + '\n'.join([f'{c.name}, {cluster.radius + c.radius:.6f}' for c in missed]))
        return

    def test_dump(self):
        with tempfile.TemporaryFile() as fp:
            self.manifold.dump(fp)
        return

    def test_load(self):
        original = self.manifold
        with tempfile.TemporaryFile() as fp:
            original.dump(fp)
            fp.seek(0)
            loaded = Manifold.load(fp, self.data)
        self.assertEqual(original, loaded)
        self.assertEqual(set(original.layers[-1]), set(loaded.layers[-1]))
        self.assertEqual(original.graphs[0], loaded.graphs[0])

        for layer in loaded.layers:
            for cluster in layer:
                self.assertIn('radius', cluster.cache)
                self.assertIn('argradius', cluster.cache)
                self.assertIn('argsamples', cluster.cache)
                self.assertIn('argmedoid', cluster.cache)
                self.assertIn('local_fractal_dimension', cluster.cache)
        return

    def test_partition_backends(self):
        data = synthetic_datasets.random(n=100, dimensions=5)[0]
        m_single = Manifold(data, 'euclidean')._partition_single([criterion.MaxDepth(5)])
        m_thread = Manifold(data, 'euclidean')._partition_threaded([criterion.MaxDepth(5)])
        self.assertEqual(m_single, m_thread)
        return
