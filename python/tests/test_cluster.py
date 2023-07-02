import math
import unittest

import abd_clam
import numpy
from abd_clam import cluster_criteria
from abd_clam import dataset
from abd_clam import metric
from abd_clam import space


class TestCluster(unittest.TestCase):
    def setUp(self):
        self.data = dataset.TabularDataset(
            numpy.random.randn(1_000, 100),
            name=f"{__name__}.data",
        )
        self.distance_metric = metric.ScipyMetric("euclidean")
        self.metric_space = space.TabularSpace(self.data, self.distance_metric, False)
        self.root = (
            abd_clam.Cluster.new_root(self.metric_space)
            .build()
            .partition([cluster_criteria.MaxDepth(5)])
        )

    def test_init(self):
        abd_clam.Cluster(
            self.metric_space,
            indices=list(range(self.data.cardinality)),
            name=f"{__name__}.init_cluster",
            parent=None,
        )
        with self.assertRaises(ValueError):
            abd_clam.Cluster(
                self.metric_space,
                indices=[],
                name=f"{__name__}.faulty_cluster",
                parent=None,
            )

    def test_eq(self):
        self.assertEqual(self.root, self.root)
        self.assertNotEqual(self.root, self.root.left_child)
        self.assertNotEqual(self.root.left_child, self.root.right_child)

    def test_hash(self):
        self.assertIsInstance(hash(self.root), int)

    def test_str(self):
        self.assertEqual("1", str(self.root))
        self.assertEqual("10", str(self.root.left_child))
        self.assertEqual("11", str(self.root.right_child))

    def test_repr(self):
        self.assertEqual(repr(self.root), f"{self.metric_space} :: Cluster 1")
        self.assertEqual(
            repr(self.root.left_child),
            f"{self.metric_space} :: Cluster 10",
        )
        self.assertEqual(
            repr(self.root.right_child),
            f"{self.metric_space} :: Cluster 11",
        )

    def test_depth(self):
        self.assertEqual(0, self.root.depth)
        self.assertEqual(1, self.root.left_child.depth)

    def test_points(self):
        self.assertSetEqual(
            set(range(self.data.cardinality)),
            set(self.root.indices),
        )
        self.assertSetEqual(
            set(range(self.data.cardinality)),
            set(self.root.left_child.indices + self.root.right_child.indices),
        )

    def test_num_samples(self):
        self.assertEqual(
            len(self.root.arg_samples),
            int(math.sqrt(self.data.cardinality)),
        )

    def test_arg_center(self):
        self.assertTrue(0 <= self.root.arg_center < self.data.cardinality)

    def test_center(self):
        self.assertTrue(numpy.all(self.data[self.root.arg_center] == self.root.center))

    def test_arg_radius(self):
        self.assertTrue(0 <= self.root.arg_radius < self.data.cardinality)

    def test_radius(self):
        self.assertGreaterEqual(self.root.radius, 0.0)

    def test_local_fractal_dimension(self):
        self.assertGreaterEqual(self.root.lfd, 0.0)

    def test_partition(self):
        self.assertFalse(self.root.is_leaf)
        self.assertEqual(self.root.max_leaf_depth, 5)

    def test_iterative_partition(self):
        self.root = self.root.iterative_partition(
            criteria=[cluster_criteria.MaxDepth(5)],
        )
        self.test_partition()

    def test_ancestry(self):
        self.assertEqual(len(self.root.ancestry), 0)

        true_ancestry = [
            self.root,
            self.root.left_child,
            self.root.left_child.left_child,
            self.root.left_child.left_child.left_child,
            self.root.left_child.left_child.left_child.left_child,
        ]
        ancestry = (
            self.root.left_child.left_child.left_child.left_child.left_child.ancestry
        )
        self.assertEqual(true_ancestry, ancestry)
