import unittest
from collections import Counter
from typing import Dict

import numpy as np

from pyclam import Graph
from pyclam import Manifold
from pyclam import criterion
from pyclam import datasets


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

        self.assertEqual(1, len(self.manifold.graphs), f'expected to have only one graph. Got {len(self.manifold.graphs)} instead.')
        self._graph_invariant(self.manifold, self.manifold.graphs[0])
        return

    def _graph_invariant(self, manifold: Manifold, graph: Graph):
        for leaf in manifold.layers[-1].clusters:
            ancestry = manifold.ancestry(leaf)
            included = sum((1 if ancestor in graph.clusters else 0 for ancestor in ancestry))
            self.assertEqual(1, included, f'expected exactly one ancestor to be in graph. Found {included} for {str(leaf)}')
        return

    def test_multiple_selection_clauses(self):
        def from_cc_gmean(ratios: np.array) -> float:
            return float(np.dot(
                a=ratios,
                b=np.asarray([1.26637064, 1.10890454, -0.10656351, -0.00044809, -0.39920286, 0.34369123]),
            ))

        def from_pc_gmean(ratios: np.array) -> float:
            return float(np.dot(
                a=ratios,
                b=np.asarray([0.09493048, 0.62547724, -0.16254063, -0.00043795, 0.13036630, -0.35289447]),
            ))

        def from_kn_gmean(ratios: np.array) -> float:
            return float(np.dot(
                a=ratios,
                b=np.asarray([-0.36966981, 0.22567179, -0.08289614, 0.00006676, 0.55057955, -0.86832389]),
            ))

        def from_sc_gmean(ratios: np.array) -> float:
            return float(np.dot(
                a=ratios,
                b=np.asarray([-0.23789545, 0.08811996, -0.02485702, 0.00003283, 0.29501756, -0.49954759]),
            ))

        self.manifold.build(
            criterion.MaxDepth(12),
            criterion.MetaMLSelect(from_cc_gmean),
            criterion.MetaMLSelect(from_pc_gmean),
            criterion.MetaMLSelect(from_kn_gmean),
            criterion.MetaMLSelect(from_sc_gmean),
        )

        self.assertEqual(4, len(self.manifold.graphs), f'expected to have only one graph. Got {len(self.manifold.graphs)} instead.')
        [self._graph_invariant(self.manifold, graph) for graph in self.manifold.graphs]
        return

    def test_property_selector(self):
        data, _ = datasets.bullseye()
        manifold: Manifold = Manifold(data, 'euclidean').build(
            criterion.MaxDepth(12),
            criterion.PropertyThreshold('cardinality', 50, 'below'),
            criterion.PropertyThreshold('radius', 50, 'below'),
            criterion.PropertyThreshold('lfd', 10, 'below'),
        )

        for graph in manifold.graphs:
            self.assertLessEqual(graph.cardinality, manifold.layers[-1].cardinality, f'selected only leaves')
        return

    def test_labels(self):
        data, labels = datasets.bullseye()

        anomalies = np.random.random(size=(data.shape[0] // 50, 2)) * (np.max(data, axis=0) - np.min(data, axis=0)) + np.min(data, axis=0)
        data = np.concatenate([data, anomalies])
        labels.extend([0 for _ in range(anomalies.shape[0])])
        labels_dict: Dict[int, int] = {i: l for i, l in enumerate(labels)}
        half_labels_dict = {i: l for i, l in labels_dict.items() if np.random.random() < 0.5}

        manifold: Manifold = Manifold(data, 'euclidean').build(
            criterion.MaxDepth(10),
            criterion.Labels(labels_dict),
            criterion.Labels(half_labels_dict),
        )
        for cluster in manifold.graphs[0]:
            _labels = [labels_dict[p] for p in cluster.argpoints]
            max_fraction = max(dict(Counter(_labels)).values()) / cluster.cardinality
            self.assertLessEqual(1 - max_fraction, 0.002, f'bat cluster selected: {str(cluster)}')

        for cluster in manifold.graphs[1]:
            _half_labels = [half_labels_dict[p] for p in cluster.argpoints if p in half_labels_dict]
            if len(_half_labels) > 0:
                max_fraction = max(dict(Counter(_half_labels)).values()) / cluster.cardinality
                self.assertLessEqual(1 - max_fraction, 0.002, f'bat cluster selected: {str(cluster)}')

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
