import unittest

import numpy as np

from pyclam import criterion, datasets, Manifold, Cluster
from pyclam.manifold import BATCH_SIZE


class TestCluster(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = np.random.randn(1_000, 100)
        cls.manifold = Manifold(cls.data, 'euclidean')
        return

    def setUp(self) -> None:
        self.cluster = Cluster(self.manifold, self.manifold.argpoints, '')
        self.children = list(self.cluster.partition())
        return

    def test_init(self):
        Cluster(self.manifold, self.manifold.argpoints, '')
        with self.assertRaises(ValueError):
            Cluster(self.manifold, [], '')
        return

    def test_eq(self):
        self.assertEqual(self.cluster, self.cluster)
        self.assertNotEqual(self.cluster, self.children[0])
        self.assertNotEqual(self.children[0], self.children[1])
        return

    def test_bool(self):
        self.assertTrue(self.cluster)

    def test_hash(self):
        self.assertIsInstance(hash(self.cluster), int)
        return

    def test_str(self):
        self.assertEqual('root', str(self.cluster))
        self.assertSetEqual({'0', '01'}, {str(c) for c in self.children})
        return

    def test_repr(self):
        self.assertIsInstance(repr(self.cluster), str)
        return

    def test_iter(self):
        self.assertEqual(
            (int(np.ceil(self.data.shape[0] / BATCH_SIZE)), min(BATCH_SIZE, self.data.shape[0]), self.data.shape[1]),
            np.array(list(self.cluster.points)).shape
        )
        return

    def test_contains(self):
        self.assertIn(self.data[0], self.cluster)
        return

    def test_metric(self):
        self.assertEqual(self.manifold.metric, self.cluster.metric)
        return

    def test_depth(self):
        self.assertEqual(0, self.cluster.depth)
        self.assertEqual(1, self.children[0].depth)
        return

    def test_points(self):
        self.assertTrue(np.array_equal(
            self.manifold.data,
            np.array(list(self.cluster.points)).reshape(self.data.shape)
        ))
        return

    def test_argpoints(self):
        self.assertSetEqual(
            set(self.manifold.argpoints),
            set(self.cluster.argpoints)
        )
        return

    def test_samples(self):
        self.assertEqual((self.cluster.nsamples, self.data.shape[-1]), self.cluster.samples.shape)
        return

    def test_argsamples(self):
        data = np.zeros((100, 100))
        for i in range(10):
            data = np.concatenate([data, np.ones((1, 100)) * i], axis=0)
            manifold = Manifold(data, 'euclidean')
            cluster = Cluster(manifold, manifold.argpoints, '')
            self.assertLessEqual(i + 1, len(cluster.argsamples))
        return

    def test_nsamples(self):
        self.assertEqual(
            int(np.sqrt(len(self.data))),
            self.cluster.nsamples
        )
        return

    def test_centroid(self):
        self.assertEqual((self.data.shape[-1],), self.cluster.centroid.shape)
        self.assertFalse(np.any(self.cluster.centroid == self.data))
        return

    def test_medoid(self):
        self.assertEqual((self.data.shape[-1],), self.cluster.medoid.shape)
        self.assertTrue(np.any(self.cluster.medoid == self.data))
        return

    def test_argmedoid(self):
        self.assertIn(self.cluster.argmedoid, self.cluster.argpoints)
        return

    def test_radius(self):
        self.assertGreaterEqual(self.cluster.radius, 0.0)
        return

    def test_argradius(self):
        self.assertIn(self.cluster.argradius, self.cluster.argpoints)
        return

    def test_local_fractal_dimension(self):
        self.assertGreaterEqual(self.cluster.local_fractal_dimension, 0)
        return

    def test_clear_cache(self):
        self.cluster.clear_cache()
        self.assertNotIn('argsamples', self.cluster.cache)
        return

    def test_tree_search(self):
        np.random.seed(42)
        data, labels = datasets.line()
        manifold = Manifold(data, 'euclidean')
        manifold.build_tree(criterion.MinPoints(10), criterion.MaxDepth(5))
        # Finding points that are in data.
        for depth, layer in enumerate(manifold.layers):
            for cluster in layer.clusters:
                linear = set([c for c in layer if c.overlaps(cluster.medoid, cluster.radius)])
                tree = set(next(iter(manifold.layers[0])).tree_search(cluster.medoid, cluster.radius, cluster.depth).keys())
                self.assertSetEqual(set(), tree - linear)
                for d in range(depth, 0, -1):
                    parents = set([manifold.select(cluster.name[:-1]) for cluster in linear])
                    for parent in parents:
                        results = parent.tree_search(cluster.medoid, cluster.radius, parent.depth)
                        self.assertIn(parent, results, msg=f'\n{parent.name} not in {[c.name for c in results]}. '
                                                           f'got {len(results)} hits.')
        # Attempting to find points that *may* be in the data
        results = manifold.root.tree_search(point=np.asarray([0, 1]), radius=0., depth=-1)
        self.assertEqual(0, len(results))
        with self.assertRaises(ValueError):
            _ = manifold.root.tree_search(point=np.asarray([0, 1]), radius=0., depth=-5)
        return

    def test_partition(self):
        manifold = Manifold(datasets.xor()[0], 'euclidean')
        cluster = manifold.select('')
        children = list(cluster.partition())
        self.assertGreater(len(children), 1)
        return

    def test_distance(self):
        self.assertGreater(self.children[0].distance_from([self.children[1].argmedoid]), 0)
        return

    def test_overlaps(self):
        point = np.ones((100,))
        self.assertTrue(self.cluster.overlaps(point, 1.))
        return

    def test_to_json(self):
        data = self.cluster.json()
        self.assertFalse(data['argpoints'])
        self.assertTrue(data['children'])
        data = self.children[0].json()
        self.assertTrue(data['argpoints'])
        self.assertFalse(data['children'])
        return

    def test_from_json(self):
        c = Cluster.from_json(self.manifold, self.cluster.json())
        self.assertEqual(self.cluster, c)
        return
