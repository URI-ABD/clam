import unittest

from pyclam import cluster
from pyclam import cluster_criteria
from pyclam import dataset
from pyclam import Edge
from pyclam import Graph
from pyclam import graph
from pyclam import graph_criteria
from pyclam import metric
from pyclam import space
from . import synthetic_datasets


class TestGraph(unittest.TestCase):
    def setUp(self) -> None:
        self.data = dataset.TabularDataset(synthetic_datasets.bullseye(n=1000, num_rings=2)[0], name=f'{__name__}.bullseye')
        self.distance_metric = metric.ScipyMetric('euclidean')
        self.metric_space = space.TabularSpace(self.data, self.distance_metric, False)

        partition_criteria = [cluster_criteria.MaxDepth(15)]
        self.root = cluster.Cluster.new_root(self.metric_space).build().partition(partition_criteria)

        self.clusters2 = graph_criteria.Layer(2)(self.root)
        pdist2 = self.metric_space.distance_pairwise([c.arg_center for c in self.clusters2])
        self.edges2: set[Edge] = {
            Edge(l, r, pdist2[i, j])
            for i, l in enumerate(self.clusters2)
            for j, r in enumerate(self.clusters2)
            if (i != j) and (pdist2[i, j] <= l.radius + r.radius)
        }
        self.graph2 = Graph(self.clusters2, self.edges2).build()
        self.graph2b = Graph(self.clusters2).build()

        self.clusters5 = graph_criteria.Layer(5)(self.root)
        self.graph5 = Graph(self.clusters5).build()

        return

    def test_init(self):
        self.assertTrue(isinstance(self.graph2, graph.Graph))
        self.assertEqual(self.graph2.population, self.data.cardinality)
        self.assertEqual(self.graph2.min_depth, min(c.depth for c in self.clusters2))
        self.assertEqual(self.graph2.max_depth, max(c.depth for c in self.clusters2))
        self.assertFalse(any(e.to_self for e in self.edges2))

        self.assertSetEqual(self.graph2.edges, self.graph2b.edges)
        return

    def test_contains(self):
        self.assertNotIn(self.root, self.graph2.clusters)
        return

    def test_components(self):
        components = self.graph2.components
        [self.assertIsInstance(component, Graph) for component in components]
        self.assertEqual(self.graph2.vertex_cardinality, sum(component.vertex_cardinality for component in components))
        return

    def test_jaccard(self):
        self.assertEqual(self.graph2.jaccard(self.graph2), 1.)
        self.assertEqual(self.graph2.jaccard(self.graph2b), 1.)
        self.assertEqual(self.graph2.jaccard(self.graph5), 1.)
        return

    @unittest.skip
    def test_replace(self):
        # manifold = Manifold(self.data, distance_metric.ScipyMetric('euclidean')).build(
        #     criterion.MaxDepth(12),
        #     criterion.PropertyThreshold('lfd', 50, 'above'),
        # )
        # graph = manifold.layers[-1].build_edges()
        #
        # for i in range(10):
        #     clusters: dict[int, Cluster] = {c: cluster for c, cluster in zip(range(graph.cardinality), graph.clusters)}
        #     if len(clusters) < 10:
        #         break
        #     sample_size = len(clusters) // 10
        #     samples: list[int] = list(map(int, numpy.random.choice(graph.cardinality, size=sample_size, replace=False)))
        #     removals: set[Cluster] = {clusters[c] for c in samples if clusters[c].children}
        #     additions: set[Cluster] = set()
        #     [additions.update(cluster.children) for cluster in removals]
        #
        #     graph.replace_clusters(
        #         removals=removals,
        #         additions=additions,
        #     )
        #
        #     clusters: set[Cluster] = set(graph.clusters)
        #
        #     self.assertEqual(0, len(removals.intersection(clusters)), f'\n1. Some removals clusters were still in the graph. iter {i}')
        #     self.assertTrue(additions.issubset(clusters), f'\n2. Some additions clusters were not in the graph. iter {i}')
        #
        #     removal_edges: set[Edge] = {edge for cluster in removals for edge in graph.edges if cluster in edge}
        #     self.assertEqual(0, len(removal_edges), f'\n3. Some removals clusters were still found among edges. iter {i}')
        #
        #     self.assertEqual(0, len(graph.cache), f'\n4. Graph cache had some elements. {[k for k in graph.cache.keys()]}. iter {i}')
        return

    @unittest.skip
    def test_dot_file(self):
        # manifold: Manifold = Manifold(self.data, distance_metric.ScipyMetric('euclidean')).build(criterion.MinRadius(0.2), criterion.Layer(7))
        # graph: Graph = manifold.graphs[0]
        # old_clusters: set[Cluster] = {cluster for cluster in graph.clusters}
        # old_edges: set[Edge] = {edge for edge in graph.edges}
        #
        # dot_string = manifold.graphs[0].as_dot_string('bullseye_d7')
        #
        # graph = graph.from_dot_string(dot_string)
        # new_clusters: set[Cluster] = {cluster for cluster in graph.clusters}
        # new_edges: set[Edge] = {edge for edge in graph.edges}
        #
        # self.assertEqual(old_clusters, new_clusters, f'Found mismatch between old and new clusters.')
        # self.assertEqual(old_edges, new_edges, f'Found mismatch between old and new edges.')
        return
