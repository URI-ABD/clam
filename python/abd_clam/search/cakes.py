"""CAKES: CLAM Augmented K-nearest neighbors Entropy-scaling Search."""

import math
import typing

from ..core import cluster
from ..core import cluster_criteria
from ..core import dataset
from ..core import metric
from ..core import space
from ..utils import constants
from ..utils import helpers

logger = helpers.make_logger(__name__)

ClusterHits = dict[cluster.Cluster, float]
IndexedHits = dict[int, float]


class CAKES:
    """CLAM Augmented K-nearest neighbors Entropy-scaling Search."""

    def __init__(self, metric_space: space.Space) -> None:
        """Initialize a CAKES object."""
        self.__metric_space = metric_space
        self.__root = cluster.Cluster.new_root(metric_space)

        self.__depth = self.__root.max_leaf_depth

    @classmethod
    def from_root(cls, root: cluster.Cluster) -> "CAKES":
        """Creates a CAKES object from a given root cluster."""
        cakes = super().__new__(cls)
        cakes.__metric_space = root.metric_space
        cakes.__root = root
        cakes.__depth = root.max_leaf_depth
        return cakes

    @property
    def depth(self) -> int:
        """Return the depth of the search tree."""
        return self.__depth

    @property
    def root(self) -> cluster.Cluster:
        """Return the root cluster."""
        return self.__root

    @property
    def metric_space(self) -> space.Space:
        """Return the metric space."""
        return self.__metric_space

    @property
    def data(self) -> dataset.Dataset:
        """Return the dataset."""
        return self.__metric_space.data

    @property
    def distance_metric(self) -> metric.Metric:
        """Return the distance metric."""
        return self.__metric_space.distance_metric

    def build(
        self,
        max_depth: typing.Optional[int] = None,
        additional_criteria: typing.Optional[
            list[cluster_criteria.ClusterCriterion]
        ] = None,
    ) -> "CAKES":
        """Builds the search tree upto singleton leaves, or an optional maximum depth.

        Args:
            max_depth: Optional. maximum depth of search tree.
            additional_criteria: Optional. Any additional criteria to use for
             building the Cluster Tree.

        Returns:
            the modified CAKES object.
        """
        logger.info(
            "Building search tree to " + "leaves"
            if max_depth is None
            else f"max_depth {max_depth} ...",
        )

        depth_criterion = (
            cluster_criteria.NotSingleton()
            if max_depth is None
            else cluster_criteria.MaxDepth(max_depth)
        )
        criteria = [depth_criterion] + (additional_criteria or [])
        self.__root = self.root.build().iterative_partition(criteria)

        self.__depth = self.__root.max_leaf_depth

        return self

    def rnn_search(
        self,
        query_instance: typing.Any,  # noqa: ANN401
        search_radius: float,
    ) -> IndexedHits:
        """Performs rho-nearest neighbors search around query with given radius.

        Args:
            query_instance: instance around which to search.
            search_radius: distance from query from within which to return
             all hits.

        Returns:
            dictionary of index-of-neighbor -> distance-to-neighbor.
        """
        candidate_clusters: list[cluster.Cluster] = self.tree_search(
            query_instance,
            search_radius,
        )

        if len(candidate_clusters) == 0:
            return {}

        return self.leaf_search(query_instance, search_radius, candidate_clusters)

    def knn_search(
        self,
        query_instance: typing.Any,  # noqa: ANN401
        k: int,
    ) -> IndexedHits:
        """Performs k-nearest neighbors search around query.

        Args:
            query_instance: instance around which to search.
            k: number of closest neighbors to look for.

        Returns:
            dictionary of index-of-neighbor -> distance-to-neighbor.
        """
        if k < 1:
            msg = f"k must be a positive integer. Got {k} instead."
            raise ValueError(msg)

        search_radius = (k / self.__root.cardinality) * self.__root.radius
        if search_radius <= 0.0:
            msg = f"Expected positive search_radius. Got {search_radius:.2e} instead."
            raise ValueError(msg)

        hits = self.rnn_search(query_instance, search_radius)
        while len(hits) == 0:  # make sure to have non-zero hits
            search_radius *= 2.0
            hits = self.rnn_search(query_instance, search_radius)

        while len(hits) < k:  # make sure to have more at least k hits
            # calculate lfd in ball around query
            if len(hits) == 1:
                lfd = 1.0
            else:
                half_count = len(
                    [
                        point
                        for point, distance in hits.items()
                        if distance <= search_radius / 2
                    ],
                )
                lfd = 1.0 if half_count == 0 else math.log2(len(hits) / half_count)

            # increase radius as guided by lfd
            if lfd < 1e-3:
                factor = 2
            else:
                factor = (k / len(hits)) ** (1.0 / (lfd + constants.EPSILON))
            if factor <= 1.0:
                msg = f"expected factor to be greater than 1. Got {factor:.2e} instead."
                raise ValueError(msg)

            search_radius *= factor

            # rerun rnn search
            hits = self.rnn_search(query_instance, search_radius)

        # sort hits in non-decreasing order of distance to query
        results = sorted([(distance, point) for point, distance in hits.items()])
        return {point: distance for distance, point in results[:k]}

    def tree_search(
        self,
        query_instance: typing.Any,  # noqa: ANN401
        search_radius: float,
    ) -> list[cluster.Cluster]:
        """Performs tree-search for the query, starting at the root.

        Consider the sphere centered at `query_instance` with the given
        `search_radius`. This method performs a breadth-first search over the
        tree and, at each iteration, discards the clusters whose volumes do not
        overlap with this sphere. Once the search terminates, the remaining
        clusters are returned.

        Args:
            query_instance: around which to search.
            search_radius: within which to look.

        Returns:
            dictionary of cluster -> distance-from-center-to-query
        """
        return self.tree_search_history(query_instance, search_radius)[1]

    def tree_search_history(
        self,
        query_instance: typing.Any,  # noqa: ANN401
        search_radius: float,
    ) -> tuple[ClusterHits, list[cluster.Cluster]]:
        """Performs tree-search for the query, starting at the root.

        Also returns the full history.
        """
        history: ClusterHits = {}
        hits: list[cluster.Cluster] = []
        candidates = [self.root]

        while len(candidates) > 0:
            logger.debug(f"Searching tree at depth {candidates[0].depth} ...")
            centers = [self.data[c.arg_center] for c in candidates]
            distances = list(
                map(float, self.distance_metric.one_to_many(query_instance, centers)),
            )
            close_enough: ClusterHits = {
                c: d
                for c, d in zip(candidates, distances)
                if d <= (c.radius + search_radius)
            }
            history.update(close_enough)
            terminal = {
                c
                for c, d in close_enough.items()
                if c.is_leaf or (c.radius + d) <= search_radius
            }
            hits.extend(terminal)
            candidates = [
                child
                for c in close_enough
                if c not in terminal
                for child in c.children  # type: ignore[union-attr]
            ]

        return history, hits

    def leaf_search(
        self,
        query_instance: typing.Any,  # noqa: ANN401
        search_radius: float,
        candidate_clusters: list[cluster.Cluster],
    ) -> IndexedHits:
        """Performs leaf search for query on given clusters.

        Args:
            query_instance: around which to look.
            search_radius: within which to look.
            candidate_clusters: candidate clusters which need to be exhaustively
             searched.

        Returns:
            dictionary of index-of-hit-instance -> distance-to-hit.
        """
        # get indices of all points in candidate clusters
        indices = [i for c in candidate_clusters for i in c.indices]
        logger.debug(f"Performing leaf-search over {len(indices)} instances.")

        # get distances from query to candidate points
        instances = [self.data[i] for i in indices]
        distances = list(
            map(float, self.distance_metric.one_to_many(query_instance, instances)),
        )

        # Filter hits with points within radius of query
        return {i: d for i, d in zip(indices, distances) if d <= search_radius}


__all__ = [
    "CAKES",
]
