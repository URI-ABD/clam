import math
import unittest

import numpy

import abd_clam
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
        assert self.root == self.root
        assert self.root != self.root.left_child
        assert self.root.left_child != self.root.right_child

    def test_hash(self):
        assert isinstance(hash(self.root), int)

    def test_str(self):
        assert str(self.root) == "1"
        assert str(self.root.left_child) == "10"
        assert str(self.root.right_child) == "11"

    def test_repr(self):
        assert repr(self.root) == f"{self.metric_space} :: Cluster 1"
        assert repr(self.root.left_child) == f"{self.metric_space} :: Cluster 10"
        assert repr(self.root.right_child) == f"{self.metric_space} :: Cluster 11"

    def test_depth(self):
        assert self.root.depth == 0
        assert self.root.left_child.depth == 1

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
        assert len(self.root.arg_samples) == int(math.sqrt(self.data.cardinality))

    def test_arg_center(self):
        assert 0 <= self.root.arg_center < self.data.cardinality

    def test_center(self):
        assert numpy.all(self.data[self.root.arg_center] == self.root.center)

    def test_arg_radius(self):
        assert 0 <= self.root.arg_radius < self.data.cardinality

    def test_radius(self):
        assert self.root.radius >= 0.0

    def test_local_fractal_dimension(self):
        assert self.root.lfd >= 0.0

    def test_partition(self):
        assert not self.root.is_leaf
        assert self.root.max_leaf_depth == 5

    def test_iterative_partition(self):
        self.root = self.root.iterative_partition(
            criteria=[cluster_criteria.MaxDepth(5)],
        )
        self.test_partition()

    def test_ancestry(self):
        assert len(self.root.ancestry) == 0

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
        assert true_ancestry == ancestry
