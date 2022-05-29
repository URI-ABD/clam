import logging

import numpy

from ..core import Cluster
from ..core import criterion
from ..core import Manifold
from ..core import types
from ..utils import constants

logger = logging.getLogger(__name__)
logger.setLevel(constants.LOG_LEVEL)

ClusterResults = dict[Cluster, float]
Results = dict[int, float]


class CAKES:
    def __init__(self, data: types.Dataset, metric: types.Metric):
        self.data: types.Dataset = data
        self.metric: types.Metric = metric
        self.manifold: Manifold = Manifold(self.data, self.metric)
        self.distance = self.manifold.distance
        self.root: Cluster = self.manifold.root

    @staticmethod
    def from_manifold(manifold: Manifold) -> 'CAKES':
        search = CAKES(manifold.data, manifold.metric)
        search.manifold = manifold
        search.distance = manifold.distance
        search.root = manifold.root
        return search

    @property
    def depth(self) -> int:
        """ returns the depth of the search tree. """
        return self.manifold.depth

    def build(self, *, max_depth: int = None) -> 'CAKES':
        """ Builds the search tree upto leaves, or an optional maximum depth.

        This method can be called repeatedly, with higher depth values, to further increase the depth of the tree.

        :param max_depth: optional maximum depth of search tree
        :return: the modified Search object.
        """
        if max_depth is None:
            logger.info(f'Building CAKES tree to leaves.')
            self.manifold.build(criterion.Layer(-1))
        elif max_depth < 1:
            raise ValueError(f'Expected a positive integer for max_depth. Got {max_depth} instead.')
        elif max_depth > self.depth:
            logger.info(f'Building CAKES tree to depth {max_depth}.')
            self.manifold.build_tree(criterion.MaxDepth(max_depth), criterion.Layer(-1))
        return self

    def rnn(self, query: types.Datum, radius: float, *, max_depth: int = None) -> Results:
        """ Performs rho-nearest neighbors search around query with given radius.

        :param query: point around which to search.
        :param radius: distance from query within which te return all hits.
        :param max_depth: optional maximum search depth.
        :return: dictionary of index-of-hit -> distance-to-hit.
        """
        candidate_clusters: ClusterResults = self.tree_search(query, radius, start=None, max_depth=max_depth)
        return self.leaf_search(query, radius, candidate_clusters)

    def rnn_points(self, query: types.Datum, radius: float, *, max_depth: int = None) -> types.Dataset:
        """ Same as rnn, but returns the raw points rather than a dictionary of indices and distances. """
        results = [(d, p) for p, d in self.rnn(query, radius, max_depth=max_depth).items()]
        return self.data[[p for _, p in sorted(results)]]

    def knn(self, query: types.Datum, k: int, *, max_depth: int = None) -> Results:
        """ Performs k-nearest neighbors search around query.

        :param query: point around which to look.
        :param k: number of closest neighbors to look for.
        :param max_depth: optional maximum search depth.
        :return: dictionary of index-of-neighbor -> distance-to-neighbor.
        """
        if k < 1:
            raise ValueError(f'k must be a positive integer. Got {k}')

        radius: float = (k / self.root.cardinality) * self.root.radius
        assert radius > 0, f'expected a positive value for radius. Got {radius:.2e} instead.'

        hits: Results = self.rnn(query, radius, max_depth=max_depth)
        while len(hits) == 0:  # make sure to have non-zero hits
            radius *= 2
            hits = self.rnn(query, radius, max_depth=max_depth)

        while len(hits) < k:  # make sure to have more at least k hits
            # calculate lfd in ball around query
            if len(hits) == 1:
                lfd = 1
            else:
                half_count = len([point for point, distance in hits.items() if distance <= radius / 2])
                lfd = 1 if half_count == 0 else numpy.log2(len(hits) / half_count)

            # increase radius as guided by lfd
            factor = (k / len(hits)) ** (1 / (lfd + constants.EPSILON))
            assert factor > 1, f'expected factor to be greater than 1. Got {factor:.2e} instead.'
            radius *= factor

            # rerun rnn search
            hits = self.rnn(query, radius, max_depth=max_depth)

        # sort hits in non-decreasing order of distance to query
        results: list[tuple[float, int]] = list(sorted([(distance, point) for point, distance in hits.items()]))
        return {point: distance for distance, point in results[:k]}

    def knn_points(self, query: types.Datum, k: int, *, max_depth: int = None) -> types.Dataset:
        """ Same as knn, but returns the raw points rather than a dictionary of indices and distances. """
        results = [(d, p) for p, d in self.knn(query, k, max_depth=max_depth).items()]
        return self.data[[p for _, p in sorted(results)]]

    def _check_shape(self, points: numpy.ndarray) -> None:
        if len(points.shape) != len(self.data.shape):
            raise ValueError(f'wrong number of dimensions in shape. Expected {len(self.data.shape)}, but got {len(points.shape)}')
        elif points.shape[1:] != self.data.shape[1:]:
            raise ValueError(f'wrong dimensionality of points. Expected {self.data.shape[1:]}, but got {points.shape[1:]}')
        else:
            return

    def _parse_query(self, query: types.Datum) -> numpy.ndarray:
        if isinstance(query, int):
            query: numpy.ndarray = numpy.asarray([self.data[query]])
        query = numpy.expand_dims(query, 0)
        self._check_shape(query)
        return query

    def tree_search(self, query: types.Datum, radius: float, *, start: Cluster = None, max_depth: int = None) -> ClusterResults:
        """ Performs tree-search for the query, starting at the given cluster.

        Consider the sphere around 'query' with the given 'radius'.
        This method performs a breadth-first search over the tree and, at each iteration,
        discards the clusters that do not have overlap with this sphere.
        Once the search terminates, the remaining clusters are returned.

        :param query: point around which to search.
        :param radius: distance from point within which to look.
        :param start: starting cluster for search.
        :param max_depth: optional maximum search depth starting at Cluster.
        :return: dictionary of cluster -> distance between query and cluster center
        """
        return self.tree_search_history(query, radius, start=start, max_depth=max_depth)[1]

    def tree_search_history(
            self,
            query: types.Datum,
            radius: float,
            *,
            start: Cluster = None,
            max_depth: int = None,
    ) -> tuple[ClusterResults, ClusterResults]:
        """ Same as tree-search, except that it also returns the history of candidate clusters at each depth. """
        start = start or self.root

        if max_depth is None:
            max_depth = self.manifold.depth
        elif max_depth < 0:
            raise ValueError(f'max_depth must be a non-negative integer. Got {max_depth} instead.')
        else:
            max_depth = min(self.manifold.depth, start.depth + max_depth)

        query = self._parse_query(query)
        candidates: ClusterResults = {start: -1}
        history: ClusterResults = dict()
        hits: ClusterResults = dict()
        for depth in range(start.depth, max_depth + 1):
            logger.debug(f'Searching tree at {depth = }')

            if len(candidates) == 0:
                break
            # find distances to centers of candidate clusters
            clusters: list[Cluster] = list(candidates.keys())
            centers: numpy.ndarray = numpy.asarray([cluster.center for cluster in clusters])
            self._check_shape(centers)
            distances: list[float] = list(self.distance(query, centers)[0])

            candidates = {  # filter out clusters that are too far away
                cluster: distance for cluster, distance in zip(clusters, distances)
                if distance <= (radius + cluster.radius)
            }
            subsumed: set[Cluster] = {  # set of clusters that lie completely within radius of query
                cluster for cluster, distance in candidates.items()
                if (distance + cluster.radius) <= radius
            }
            hits.update({  # add subsumed clusters to hits, singletons should already be included here
                cluster: distance
                for cluster, distance in candidates.items()
                if (cluster in subsumed) or (len(cluster.children) == 0)
            })

            history.update(candidates)  # add candidates from this depth to history

            candidates = {  # expand search to children
                child: -1
                for cluster in candidates.keys()
                for child in cluster.children
                if cluster not in hits
            }

        logger.debug(f'Found {len(hits)} candidate clusters from tree-search.')
        return history, hits

    def leaf_search(self, query: types.Datum, radius: float, clusters: ClusterResults) -> Results:
        """ Performs leaf search for query on given clusters.

        :param query: point around which to look.
        :param radius: distance from point within which to look.
        :param clusters: candidate clusters which need to be exhaustively searched.
        :return: dictionary of index-of-hit -> distance-to-hit.
        """
        query = self._parse_query(query)

        # get indices of all points in candidate clusters
        argpoints: list[int] = [point for cluster in clusters for point in cluster.argpoints]
        logger.debug(f'Performing leaf-search over {len(argpoints)} points.')

        # get distances from query to candidate points
        points: numpy.ndarray = self.data[argpoints]
        self._check_shape(points)
        distances: list[float] = list(self.distance(query, points)[0])

        # Filter hits with points within radius of query
        hits = {point: distance for point, distance in zip(argpoints, distances) if distance <= radius}
        return hits
