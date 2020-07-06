import unittest
from itertools import combinations
from typing import Set, Dict, List

import numpy as np

from pyclam import datasets, criterion
from pyclam.manifold import Manifold, Graph, Cluster
from pyclam.types import Edge


class TestGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        np.random.seed(42)
        cls.data, _ = datasets.bullseye(n=1000, num_rings=2)
        cls.manifold = Manifold(cls.data, 'euclidean')
        cls.manifold.build(
            criterion.MaxDepth(10),
            criterion.LFDRange(60, 50),
        )
        return

    def test_init(self):
        Graph(*[c for c in self.manifold.graph])
        with self.assertRaises(AssertionError):
            Graph('1')
        with self.assertRaises(AssertionError):
            Graph('1', '2', '3')
        return

    def test_eq(self):
        self.assertEqual(self.manifold.layers[0], self.manifold.layers[0])
        for left, right in combinations(self.manifold.layers, 2):
            self.assertNotEqual(left, right)
        return

    def test_iter(self):
        clusters = list(self.manifold.layers[1])
        self.assertEqual(2, len(clusters))
        self.assertIn(clusters[0], self.manifold.select('').children)
        return

    def test_metric(self):
        self.assertEqual(self.manifold.metric, self.manifold.graph.metric)

    def test_population(self):
        self.assertEqual(len(self.manifold.argpoints), self.manifold.graph.population)
        return

    def test_str(self):
        self.assertEqual(self.manifold.graph.cardinality, len(str(self.manifold.graph).split(',')))
        return

    def test_repr(self):
        self.assertIsInstance(repr(self.manifold.graph), str)
        return

    def test_contains(self):
        root = self.manifold.select('')
        self.assertNotIn(root, self.manifold.graph)
        return

    def test_cached_edges(self):
        self.assertGreaterEqual(len(self.manifold.graph.cached_edges), 0)
        self.assertLessEqual(len(self.manifold.graph.cached_edges), self.manifold.graph.cardinality * (self.manifold.graph.cardinality - 1) / 2)
        self.assertEqual(
            sum(len(self.manifold.graph.edges[cluster]) for cluster in self.manifold.graph.clusters),
            len(self.manifold.graph.cached_edges),
        )
        return

    def test_subgraphs(self):
        [self.assertIsInstance(subgraph, Graph) for subgraph in self.manifold.graph.subgraphs]
        self.assertEqual(self.manifold.graph.cardinality, sum(subgraph.cardinality for subgraph in self.manifold.graph.subgraphs))
        return

    def test_clear_cache(self):
        self.manifold.graph.clear_cache()
        _ = self.manifold.graph.cached_edges
        self.assertIn('edges', self.manifold.graph.cache)
        self.manifold.graph.clear_cache()
        self.assertNotIn('edges', self.manifold.graph.cache)
        return

    def test_bft(self):
        visited = self.manifold.graph.bft(next(iter(self.manifold.graph.walkable_clusters)))
        self.assertGreater(len(visited), 0)
        self.assertLessEqual(len(visited), self.manifold.graph.cardinality)
        return

    def test_dft(self):
        visited = self.manifold.graph.dft(next(iter(self.manifold.graph.walkable_edges)))
        self.assertGreater(len(visited), 0)
        self.assertLessEqual(len(visited), self.manifold.graph.cardinality)
        return

    def test_random_walks(self):
        results = self.manifold.graph.random_walks(
            starts=list(self.manifold.graph.walkable_clusters),
            steps=100,
        )
        self.assertGreater(len(results), 0)
        [self.assertGreater(v, 0) for k, v in results.items()]
        return

    @unittest.skip
    def test_replace(self):
        self.manifold.build(
            criterion.MaxDepth(12),
            criterion.LFDRange(80, 20),
        )
        self.manifold.layers[-1].build_edges()

        for i in range(100):
            clusters: Dict[int, Cluster] = {c: cluster for c, cluster in zip(range(self.manifold.graph.cardinality), self.manifold.graph.clusters)}
            if len(clusters) < 10:
                break
            sample_size = len(clusters) // 10
            samples: List[int] = list(map(int, np.random.choice(self.manifold.graph.cardinality, size=sample_size, replace=False)))
            removals: Set[Cluster] = {clusters[c] for c in samples if clusters[c].children}
            additions: Set[Cluster] = set()
            [additions.update(cluster.children) for cluster in removals]

            self.manifold.graph.replace_clusters(
                removals=removals,
                additions=additions,
                recompute_probabilities=True,
            )

            clusters: Set[Cluster] = set(self.manifold.graph.clusters)

            subsumed_clusters: Set[Cluster] = self.manifold.graph.cache['subsumed_clusters']
            subsumed_edges: Dict[Cluster, Set[Edge]] = self.manifold.graph.cache['subsumed_edges']

            walkable_clusters: Set[Cluster] = self.manifold.graph.cache['walkable_clusters']
            walkable_edges: Dict[Cluster, Set[Edge]] = self.manifold.graph.cache['walkable_edges']

            self.assertTrue(subsumed_clusters.issubset(clusters), f"\n1. subsumed clusters were not subset of clusters. iter: {i}")
            self.assertTrue(walkable_clusters.issubset(clusters), f"\n2. walkable clusters were not subset of clusters. iter: {i}")
            self.assertTrue(walkable_clusters.isdisjoint(subsumed_clusters), f"\n3. walkable clusters and subsumed clusters were not disjoint sets. iter: {i}")
            self.assertSetEqual(clusters, subsumed_clusters.union(walkable_clusters),
                                f"\n4. union of subsumed and walkable clusters was not the same as all clusters. iter: {i}")
            self.assertSetEqual(clusters, set(subsumed_edges.keys()), f"\n5. keys in subsumed edges were not the same as clusters. iter: {i}")
            self.assertSetEqual(walkable_clusters, set(walkable_edges.keys()), f"\n6. keys in walkable edges were not the same as walkable clusters. iter: {i}")
        return
