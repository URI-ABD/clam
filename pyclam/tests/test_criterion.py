import unittest
from inspect import stack
from itertools import cycle

from matplotlib import pyplot as plt

import chess.datasets as d
import chess.manifold as m
from chess.criterion import MinPoints, MinRadius, MaxDepth, LeavesSubgraph, MinNeighborhood, NewSubgraph, MinCardinality, MedoidNearCentroid, \
    UniformDistribution


class TestCriterion(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = d.bullseye(n=500)[0]
        return

    def setUp(self) -> None:
        self.manifold = m.Manifold(self.data, 'euclidean')
        return

    def test_min_radius(self):
        min_radius = 0.1
        self.manifold.build(MinRadius(min_radius), MaxDepth(8))
        self.assertTrue(all((c.radius >= min_radius for g in self.manifold for c in g)))
        [self.assertLessEqual(len(c.children), 1) for g in self.manifold for c in g if c.radius <= min_radius]
        return

    def test_leaves_subgraph(self):
        self.assertEqual(1, len(self.manifold.graphs[-1].subgraphs))
        self.manifold.build(LeavesSubgraph(self.manifold), MaxDepth(8))
        self.assertGreater(len(self.manifold.graphs[-1].subgraphs), 1)
        return

    def test_min_cardinality(self):
        data = d.random()[0]
        self.manifold = m.Manifold(data, 'euclidean')
        self.assertEqual(1, len(self.manifold.graphs[-1].subgraphs))
        self.manifold.build(MinCardinality(1))
        self.assertGreater(len(self.manifold.graphs[-1].subgraphs), 1)
        self.assertTrue(all((len(c.neighbors) == 0) for c in self.manifold.graphs[-1]))
        return

    def test_min_neighborhood(self):
        self.manifold.build(MinNeighborhood(5, 1), MaxDepth(8))
        return

    def test_new_subgraph(self):
        self.manifold.build(NewSubgraph(self.manifold))
        return

    def test_combinations(self):
        min_radius, min_points, max_depth = 0.15, 10, 8
        self.manifold.build(MinRadius(min_radius), MinPoints(min_points), MaxDepth(max_depth))
        self.assertTrue(all((c.radius >= min_radius for g in self.manifold for c in g)))
        [self.assertLessEqual(len(c.children), 1) for g in self.manifold.graphs for c in g
         if c.radius <= min_radius or len(c.argpoints) <= min_points or c.depth >= max_depth]
        # self.plot()
        return

    def test_medoid_near_centroid(self):
        self.manifold.build(MedoidNearCentroid(), MaxDepth(8))
        # self.plot()
        return

    def test_uniform_distribution(self):
        self.manifold.build(UniformDistribution(), MaxDepth(8))
        # self.plot()
        return

    def plot(self):
        colors = cycle('bgrcmy')
        shapes = cycle('.ov^<>1234spP*+xXD|')
        for graph in self.manifold.graphs:
            fig, ax = plt.subplots()
            for cluster in graph:
                marker = next(colors) + next(shapes)

                # The points themselves.
                [plt.plot(batch[:, 0], batch[:, 1], marker, alpha=0.7) for batch in cluster.points]

                # Cluster attributes
                ax.add_artist(
                    plt.Circle(
                        tuple(cluster.medoid),
                        cluster.radius,
                        fill=False,
                        color='k',
                        zorder=10
                    )
                )
                plt.plot(*cluster.centroid, '.k')
                plt.plot(*cluster.medoid, '*k')
            plt.title(f'{graph.depth} - {stack()[1].function}')
            plt.show()
