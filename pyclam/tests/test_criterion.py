import unittest

from pyclam import Manifold, datasets, criterion, Graph


# noinspection SpellCheckingInspection
class TestCriterion(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = datasets.bullseye(n=500)[0]
        return

    def setUp(self) -> None:
        self.manifold = Manifold(self.data, 'euclidean')
        return

    def test_min_radius(self):
        min_radius = 0.1
        self.manifold.build(
            criterion.MinRadius(min_radius),
        )
        self.assertTrue(all((
            cluster.radius > min_radius
            for layer in self.manifold.layers
            for cluster in layer
            if cluster.children
        )))
        self.assertTrue(all((
            cluster.radius <= min_radius
            for layer in self.manifold.layers
            for cluster in layer
            if not cluster.children
        )))
        return

    def test_combinations(self):
        min_points, max_depth = 10, 8
        self.manifold.build(criterion.MinPoints(min_points), criterion.MaxDepth(max_depth))
        [self.assertLessEqual(len(c.children), 1) for g in self.manifold.layers for c in g
         if len(c.argpoints) <= min_points or c.depth >= max_depth]
        # self.plot()
        return

    def test_medoid_near_centroid(self):
        self.manifold.build(criterion.MedoidNearCentroid(), criterion.MaxDepth(8))
        # self.plot()
        return

    def test_uniform_distribution(self):
        self.manifold.build(criterion.UniformDistribution(), criterion.MaxDepth(8))
        # self.plot()
        return

    def test_lfd_range(self):
        self.manifold.build(criterion.MaxDepth(12), criterion.LFDRange(60, 50))

        for leaf in self.manifold.layers[-1].clusters:
            ancestry = self.manifold.ancestry(leaf)
            included = sum((1 if ancestor in self.manifold.graph.clusters else 0 for ancestor in ancestry))
            self.assertEqual(1, included, f"expected exactly one ancestor to be in graph. Found {included}")
        return

    def _graph_invariant(self, manifold: Manifold, graph: Graph):
        for leaf in manifold.layers[-1].clusters:
            ancestry = manifold.ancestry(leaf)
            included = sum((1 if ancestor in graph.clusters else 0 for ancestor in ancestry))
            self.assertEqual(1, included, f"expected exactly one ancestor to be in graph. Found {included}")
        return

    def test_linear_regression(self):
        constants = [-0.00000074, 0.33298534, 0.06788443, -0.00021080, -0.43125318, -0.12271546]

        self.manifold.build(
            criterion.MaxDepth(50),
            criterion.LinearRegressionConstants(constants),
        )

        self._graph_invariant(self.manifold, self.manifold.graph)
        return

    def test_minimize_subsumed(self):
        fraction: float = 0.2

        self.manifold.build(
            criterion.MaxDepth(12),
            criterion.LFDRange(80, 20),
            criterion.MinimizeSubsumed(fraction),
        )

        for leaf in self.manifold.layers[-1].clusters:
            ancestry = self.manifold.ancestry(leaf)
            included = sum((1 if ancestor in self.manifold.graph.clusters else 0 for ancestor in ancestry))
            self.assertEqual(1, included, f"expected exactly one ancestor to be in graph. Found {included}")
        return

    # def plot(self):
    #     from inspect import stack
    #     from itertools import cycle
    #
    #     from matplotlib import pyplot as plt
    #     colors = cycle('bgrcmy')
    #     shapes = cycle('.ov^<>1234spP*+xXD|')
    #     for graph in self.manifold.graphs:
    #         fig, ax = plt.subplots()
    #         for cluster in graph:
    #             marker = next(colors) + next(shapes)
    #
    #             # The points themselves.
    #             [plt.plot(batch[:, 0], batch[:, 1], marker, alpha=0.7) for batch in cluster.points]
    #
    #             # Cluster attributes
    #             ax.add_artist(
    #                 plt.Circle(
    #                     tuple(cluster.medoid),
    #                     cluster.radius,
    #                     fill=False,
    #                     color='k',
    #                     zorder=10
    #                 )
    #             )
    #             plt.plot(*cluster.centroid, '.k')
    #             plt.plot(*cluster.medoid, '*k')
    #         plt.title(f'{graph.depth} - {stack()[1].function}')
    #         plt.show()
