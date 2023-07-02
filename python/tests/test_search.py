import operator
import unittest

from abd_clam import dataset
from abd_clam import metric
from abd_clam import search
from abd_clam import space
from abd_clam.utils import synthetic_data


class TestSearch(unittest.TestCase):
    def setUp(self) -> None:
        self.data = dataset.TabularDataset(
            synthetic_data.bullseye(n=100, num_rings=3)[0],
            name=f"{__name__}.data",
        )
        self.indices = list(range(self.data.cardinality))
        self.distance_metric = metric.ScipyMetric("euclidean")
        self.metric_space = space.TabularSpace(self.data, self.distance_metric, False)
        self.max_depth = 10
        self.cakes = search.CAKES(self.metric_space).build(max_depth=self.max_depth)

        self.search_radii = [(self.cakes.root.radius / i) for i in range(200, 100, -20)]
        self.query = self.data[0]
        self.distances = self.metric_space.distance_pairwise(self.indices)

    def test_init(self) -> None:
        self.assertTrue(isinstance(self.cakes, search.CAKES))
        _ = self.cakes.metric_space
        _ = self.cakes.root
        self.assertLessEqual(self.cakes.depth, self.max_depth)

    def test_rnn_search(self) -> None:
        self.assertEqual(1, len(self.cakes.rnn_search(self.query, 0.0)))
        self.assertLessEqual(1, len(self.cakes.rnn_search(self.query, 1.0)))

        for i in self.indices:
            for radius in self.search_radii:
                naive_results: dict[int, float] = {
                    j: d
                    for j, d in zip(self.indices, self.distances[i, :])
                    if d <= radius
                }
                rnn_results: dict[int, float] = self.cakes.rnn_search(
                    self.data[i],
                    radius,
                )

                self.assertEqual(
                    len(naive_results),
                    len(rnn_results),
                    "expected the same number of results from naive and rnn searches.",
                )
                self.assertSetEqual(
                    set(naive_results.keys()),
                    set(rnn_results.keys()),
                    "expected the same set of results from naive and rnn searches.",
                )

    def test_knn_search(self) -> None:
        distances = dict(zip(self.indices, self.distances[0, :]))
        points: list[int] = [
            p for p, _ in sorted(distances.items(), key=operator.itemgetter(1))
        ]

        ks = list(range(1, 10))
        ks.extend(range(10, self.data.cardinality, 1000))
        for k in ks:
            naive_results = points[:k]
            knn_results: list[int] = list(self.cakes.knn_search(self.query, k).keys())
            self.assertEqual(
                len(naive_results),
                len(knn_results),
                "expected the same number of results from naive and knn searches.",
            )
            self.assertSetEqual(
                set(naive_results),
                set(knn_results),
                "expected the same set of results from naive and knn searches.",
            )

    def test_tree_search_history(self) -> None:
        radius: float = self.cakes.root.radius / 10
        history, hits = self.cakes.tree_search_history(self.query, radius)

        for c in hits:
            self.assertTrue(c in history, f"The hit {c!s} was not found in history.")
        self.assertLessEqual(
            len(hits),
            len(history),
            "history should have at least a many members as hits.",
        )

        for c in history:
            if c not in hits:
                self.assertFalse(
                    c.is_leaf,
                    "A non-hit member of history must have had children.",
                )

        depths = {cluster.depth for cluster in history}
        depth_range = set(range(len(depths)))
        missing_depths = depths.union(depth_range) - depths.intersection(depth_range)
        self.assertEqual(
            0,
            len(missing_depths),
            f"history should contain clusters from every depth. "
            f"Did not contain: {missing_depths}",
        )
