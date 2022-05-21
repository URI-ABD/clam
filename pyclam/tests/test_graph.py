import unittest
import itertools

import numpy

from pyclam import Cluster
from pyclam import criterion
from pyclam import Edge
from pyclam import Graph
from pyclam import Manifold
from pyclam.utils import synthetic_datasets


class TestGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        numpy.random.seed(42)
        cls.data, _ = synthetic_datasets.bullseye(n=1000, num_rings=2)
        cls.manifold: Manifold = Manifold(cls.data, 'euclidean')
        cls.manifold.build(
            criterion.MaxDepth(10),
            criterion.Layer(5),
        )
        cls.graph: Graph = cls.manifold.graphs[0]
        return

    def test_init(self):
        Graph(*[cluster for cluster in self.graph.clusters])
        with self.assertRaises(AssertionError):
            Graph('1')
        with self.assertRaises(AssertionError):
            Graph('1', '2', '3')
        return

    def test_eq(self):
        self.assertEqual(self.manifold.layers[0], self.manifold.layers[0])
        for left, right in itertools.combinations(self.manifold.layers, 2):
            self.assertNotEqual(left, right)
        return

    def test_iter(self):
        clusters = list(self.manifold.layers[1].clusters)
        self.assertEqual(2, len(clusters))
        self.assertIn(clusters[0], self.manifold.select('').children)
        return

    def test_metric(self):
        self.assertEqual(self.manifold.metric, self.graph.metric)

    def test_population(self):
        self.assertEqual(len(self.manifold.argpoints), self.graph.population)
        return

    def test_str(self):
        self.assertEqual(self.graph.cardinality, len(str(self.graph).split(', ')))
        return

    def test_repr(self):
        self.assertIsInstance(repr(self.graph), str)
        return

    def test_contains(self):
        root = self.manifold.select('')
        self.assertNotIn(root, self.graph)
        return

    def test_components(self):
        components: set[Graph] = self.graph.components
        [self.assertIsInstance(component, Graph) for component in components]
        self.assertEqual(self.graph.cardinality, sum(component.cardinality for component in components))
        return

    def test_clear_cache(self):
        self.graph.clear_cache()
        _ = self.graph.depth
        self.assertIn('depth', self.graph.cache)
        self.graph.clear_cache()
        self.assertNotIn('depth', self.graph.cache)
        self.assertEqual(0, len(self.graph.cache))
        return

    def test_bft(self):
        visited = self.graph.bft(next(iter(self.graph.clusters)))
        self.assertGreater(len(visited), 0)
        self.assertLessEqual(len(visited), self.graph.cardinality)
        return

    def test_dft(self):
        visited = self.graph.dft(next(iter(self.graph.clusters)))
        self.assertGreater(len(visited), 0)
        self.assertLessEqual(len(visited), self.graph.cardinality)
        return

    def test_replace(self):
        manifold = Manifold(self.data, 'euclidean').build(
            criterion.MaxDepth(12),
            criterion.LFDRange(80, 20),
        )
        graph = manifold.layers[-1].build_edges()

        for i in range(10):
            clusters: dict[int, Cluster] = {c: cluster for c, cluster in zip(range(graph.cardinality), graph.clusters)}
            if len(clusters) < 10:
                break
            sample_size = len(clusters) // 10
            samples: list[int] = list(map(int, numpy.random.choice(graph.cardinality, size=sample_size, replace=False)))
            removals: set[Cluster] = {clusters[c] for c in samples if clusters[c].children}
            additions: set[Cluster] = set()
            [additions.update(cluster.children) for cluster in removals]

            graph.replace_clusters(
                removals=removals,
                additions=additions,
            )

            clusters: set[Cluster] = set(graph.clusters)

            self.assertEqual(0, len(removals.intersection(clusters)), f'\n1. Some removals clusters were still in the graph. iter {i}')
            self.assertTrue(additions.issubset(clusters), f'\n2. Some additions clusters were not in the graph. iter {i}')

            removal_edges: set[Edge] = {edge for cluster in removals for edge in graph.edges if cluster in edge}
            self.assertEqual(0, len(removal_edges), f'\n3. Some removals clusters were still found among edges. iter {i}')

            self.assertEqual(0, len(graph.cache), f'\n4. Graph cache had some elements. {[k for k in graph.cache.keys()]}. iter {i}')
        return

    def test_dot_file(self):
        manifold: Manifold = Manifold(self.data, 'euclidean').build(criterion.MinRadius(0.2), criterion.Layer(7))
        graph: Graph = manifold.graphs[0]
        old_clusters: set[Cluster] = {cluster for cluster in graph.clusters}
        old_edges: set[Edge] = {edge for edge in graph.edges}

        dot_string = manifold.graphs[0].as_dot_string('bullseye_d7')

        graph = graph.from_dot_string(dot_string)
        new_clusters: set[Cluster] = {cluster for cluster in graph.clusters}
        new_edges: set[Edge] = {edge for edge in graph.edges}

        self.assertEqual(old_clusters, new_clusters, f'Found mismatch between old and new clusters.')
        self.assertEqual(old_edges, new_edges, f'Found mismatch between old and new edges.')
        return

    def test_jaccard(self):
        manifold: Manifold = Manifold(self.data, 'euclidean').build(criterion.MaxDepth(5))

        for i, left in enumerate(manifold.layers):
            self.assertEqual(1, left.jaccard(left), 'identical graphs should have a jaccard index of 1.')
            for j, right in enumerate(manifold.layers):
                if i != j:
                    self.assertEqual(1, left.jaccard(right), f'different layers should have a jaccard index of 1.')

        while len(manifold.layers[-1].components) < 2:
            manifold.build(criterion.MaxDepth(manifold.depth + 1))

        for i, left in enumerate(manifold.layers[-1].components):
            self.assertEqual(1, left.jaccard(left), 'identical components should have a jaccard index of 1.')
            for j, right in enumerate(manifold.layers[-1].components):
                if i != j:
                    self.assertEqual(0, left.jaccard(right), f'different components should have a jaccard index of 0.')

    def test_pruned(self):
        manifold: Manifold = Manifold(self.data, 'euclidean').build(criterion.MaxDepth(10), criterion.Layer(8))
        graph = manifold.graphs[0]
        pruned_graph, subsumed_clusters = graph.pruned_graph

        self.assertLessEqual(pruned_graph.cardinality, graph.cardinality)
        self.assertSetEqual(set(pruned_graph.clusters), set(subsumed_clusters.keys()))
        for cluster, subsumed in subsumed_clusters.items():
            self.assertEqual(0, len(subsumed.intersection(set(pruned_graph.clusters))))
