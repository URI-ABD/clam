import math
import unittest

import numpy

import pyclam
from pyclam import cluster_criteria
from pyclam import dataset
from pyclam import metric
from pyclam import space


class TestCluster(unittest.TestCase):

    def setUp(self):
        self.data = dataset.TabularDataset(numpy.random.randn(1_000, 100), name=f'{__name__}.data')
        self.distance_metric = metric.ScipyMetric('euclidean')
        self.metric_space = space.TabularSpace(self.data, self.distance_metric, False)
        self.root = pyclam.Cluster.new_root(self.metric_space).build().partition([cluster_criteria.MaxDepth(5)])
        return

    def test_init(self):
        pyclam.Cluster(
            self.metric_space,
            indices=list(range(self.data.cardinality)),
            name=f'{__name__}.init_cluster',
            parent=None,
        )
        with self.assertRaises(ValueError):
            pyclam.Cluster(
                self.metric_space,
                indices=list(),
                name=f'{__name__}.faulty_cluster',
                parent=None,
            )
        return

    def test_eq(self):
        self.assertEqual(self.root, self.root)
        self.assertNotEqual(self.root, self.root.left_child)
        self.assertNotEqual(self.root.left_child, self.root.right_child)
        return

    def test_hash(self):
        self.assertIsInstance(hash(self.root), int)
        return

    def test_str(self):
        self.assertEqual('1', str(self.root))
        self.assertEqual('10', str(self.root.left_child))
        self.assertEqual('11', str(self.root.right_child))
        return

    def test_repr(self):
        self.assertEqual(repr(self.root), f'{self.metric_space} :: Cluster 1')
        self.assertEqual(repr(self.root.left_child), f'{self.metric_space} :: Cluster 10')
        self.assertEqual(repr(self.root.right_child), f'{self.metric_space} :: Cluster 11')
        return

    def test_depth(self):
        self.assertEqual(0, self.root.depth)
        self.assertEqual(1, self.root.left_child.depth)
        return

    def test_points(self):
        self.assertSetEqual(
            set(range(self.data.cardinality)),
            set(self.root.indices),
        )
        self.assertSetEqual(
            set(range(self.data.cardinality)),
            set(self.root.left_child.indices + self.root.right_child.indices),
        )
        return

    def test_num_samples(self):
        self.assertEqual(len(self.root.arg_samples), int(math.sqrt(self.data.cardinality)))
        return

    def test_arg_center(self):
        self.assertTrue(0 <= self.root.arg_center < self.data.cardinality)
        return

    def test_center(self):
        self.assertTrue(numpy.all(self.data[self.root.arg_center] == self.root.center))
        return

    def test_arg_radius(self):
        self.assertTrue(0 <= self.root.arg_radius < self.data.cardinality)
        return

    def test_radius(self):
        self.assertGreaterEqual(self.root.radius, 0.)
        return

    def test_local_fractal_dimension(self):
        self.assertGreaterEqual(self.root.lfd, 0.)
        return

    def test_partition(self):
        self.assertFalse(self.root.is_leaf)
        self.assertEqual(self.root.max_leaf_depth, 5)
        return

    def test_iterative_partition(self):
        self.root = self.root.iterative_partition(criteria=[cluster_criteria.MaxDepth(5)])
        self.test_partition()
        return

    def test_ancestry(self):
        self.assertEqual(len(self.root.ancestry), 0)

        true_ancestry = [
            self.root,
            self.root.left_child,
            self.root.left_child.left_child,
            self.root.left_child.left_child.left_child,
            self.root.left_child.left_child.left_child.left_child,
        ]
        ancestry = self.root.left_child.left_child.left_child.left_child.left_child.ancestry
        self.assertEqual(true_ancestry, ancestry)

        return
