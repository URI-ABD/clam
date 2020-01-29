import unittest

import numpy as np

from chess import criterion, datasets
from chess.manifold import Manifold, Cluster, BATCH_SIZE


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

    def test_hash(self):
        self.assertIsInstance(hash(self.cluster), int)
        return

    def test_str(self):
        self.assertEqual('root', str(self.cluster))
        self.assertSetEqual({'1', '2'}, set([str(c) for c in self.children]))
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
        self.assertNotIn('_argsamples', self.cluster.__dict__)
        return

    def test_tree_search(self):
        np.random.seed(42)
        data, labels = datasets.line()
        m = Manifold(data, 'euclidean')
        m.build_tree(criterion.MinPoints(10), criterion.MaxDepth(5))
        # Finding points that are in data.
        for depth, graph in enumerate(m.graphs):
            for cluster in graph:
                linear = set([c for c in graph if c.overlaps(cluster.medoid, cluster.radius)])
                tree = set(next(iter(m.graphs[0])).tree_search(cluster.medoid, cluster.radius, cluster.depth).keys())
                self.assertSetEqual(set(), tree - linear)
                for d in range(depth, 0, -1):
                    parents = set([m.select(cluster.name[:-1]) for cluster in linear])
                    for parent in parents:
                        self.assertIn(parent, parent.tree_search(cluster.medoid, cluster.radius, parent.depth))
        # Attempting to find points that *may* be in the data
        c: Cluster = next(iter(m.graphs[0]))
        results = c.tree_search(np.asarray([0, 1]), 0., -1)
        self.assertEqual(0, len(results))
        with self.assertRaises(ValueError):
            _ = c.tree_search(np.asarray([0, 1]), 0., -5)
        return

    def test_partition(self):
        children = list(self.cluster.partition())
        self.assertGreater(len(children), 1)
        return

    def test_neighbors(self):
        for dataset in [datasets.spiral_2d, datasets.tori, datasets.skewer, datasets.random, datasets.line, datasets.xor]:
            data, labels = dataset()
            manifold = Manifold(data, 'euclidean')
            manifold.build(criterion.MaxDepth(8))
            for depth, graph in enumerate(manifold.graphs):
                for cluster in graph:
                    potential_neighbors = [c for c in graph if c.name != cluster.name]
                    if len(potential_neighbors) == 0:
                        continue
                    elif len(potential_neighbors) == 1:
                        centers = np.expand_dims(potential_neighbors[0].medoid, axis=0)
                    else:
                        centers = np.stack([c.medoid for c in potential_neighbors])
                    distances = list(cluster.distance(centers))
                    radii = [cluster.radius + c.radius for c in potential_neighbors]
                    potential_neighbors = {c for c, d, r in zip(potential_neighbors, distances, radii) if d <= r}
                    if (potential_neighbors - set(cluster.neighbors.keys())) or (set(cluster.neighbors.keys()) - potential_neighbors):
                        print(depth, cluster.name, 'truth:', [n.name for n in potential_neighbors])
                        print(depth, cluster.name, 'missed:', [n.name for n in (potential_neighbors - set(cluster.neighbors.keys()))])
                        print(depth, cluster.name, 'extra:', [n.name for n in (set(cluster.neighbors.keys()) - potential_neighbors)])
                    self.assertFalse(cluster.neighbors.keys() - potential_neighbors)
                    self.assertFalse(potential_neighbors - cluster.neighbors.keys())
        return

    def test_distance(self):
        self.assertGreater(self.children[0].distance(np.expand_dims(self.children[1].medoid, 0)), 0)
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
