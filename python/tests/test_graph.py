import unittest

from abd_clam import Edge
from abd_clam import Graph
from abd_clam import cluster
from abd_clam import cluster_criteria
from abd_clam import dataset
from abd_clam import graph
from abd_clam import graph_criteria
from abd_clam import metric
from abd_clam import space
from abd_clam.utils import synthetic_data


class TestGraph(unittest.TestCase):
    def setUp(self) -> None:
        self.data = dataset.TabularDataset(
            synthetic_data.bullseye(n=1000, num_rings=2)[0],
            name=f"{__name__}.bullseye",
        )
        self.distance_metric = metric.ScipyMetric("euclidean")
        self.metric_space = space.TabularSpace(self.data, self.distance_metric, False)

        partition_criteria = [cluster_criteria.MaxDepth(15)]
        self.root = (
            cluster.Cluster.new_root(self.metric_space)
            .build()
            .partition(partition_criteria)
        )

        self.clusters2 = graph_criteria.Layer(2)(self.root)
        pdist2 = self.metric_space.distance_pairwise(
            [c.arg_center for c in self.clusters2],
        )
        self.edges2: set[Edge] = {
            Edge(left, right, pdist2[i, j])
            for i, left in enumerate(self.clusters2)
            for j, right in enumerate(self.clusters2)
            if (i != j) and (pdist2[i, j] <= left.radius + right.radius)
        }
        self.graph2 = Graph(self.clusters2, self.edges2).build()
        self.graph2b = Graph(self.clusters2).build()

        self.clusters5 = graph_criteria.Layer(5)(self.root)
        self.graph5 = Graph(self.clusters5).build()

    def test_init(self):
        self.assertTrue(isinstance(self.graph2, graph.Graph))
        self.assertEqual(self.graph2.population, self.data.cardinality)
        self.assertEqual(self.graph2.min_depth, min(c.depth for c in self.clusters2))
        self.assertEqual(self.graph2.max_depth, max(c.depth for c in self.clusters2))
        self.assertFalse(any(e.to_self for e in self.edges2))

        self.assertSetEqual(self.graph2.edges, self.graph2b.edges)

    def test_contains(self):
        self.assertNotIn(self.root, self.graph2.clusters)

    def test_components(self):
        components = self.graph2.components
        [self.assertIsInstance(component, Graph) for component in components]
        self.assertEqual(
            self.graph2.vertex_cardinality,
            sum(component.vertex_cardinality for component in components),
        )

    def test_jaccard(self):
        self.assertEqual(self.graph2.jaccard(self.graph2), 1.0)
        self.assertEqual(self.graph2.jaccard(self.graph2b), 1.0)
        self.assertEqual(self.graph2.jaccard(self.graph5), 1.0)
