from typing import Dict, Optional, List, Set, Tuple

import numpy as np

import pyclam.criterion as criterion
from pyclam.manifold import Manifold, Cluster, EPSILON
from pyclam.types import Data, Metric, Datum

ClusterResults = Dict[Cluster, float]
Results = Dict[int, float]


class Search:
    def __init__(self, data: Data, metric: Metric):
        self.data: Data = data
        self.metric: Metric = metric
        self.manifold: Manifold = Manifold(self.data, self.metric)
        self.distance = self.manifold.distance
        self.root: Cluster = self.manifold.root

    @property
    def depth(self) -> int:
        """ returns the depth of the search tree. """
        return self.manifold.depth

    def build(self, *, max_depth: Optional[int] = None) -> 'Search':
        """ Builds the search tree upto leaves, or an optional maximum depth.

        This method can be called repeatedly, with higher depth values, to further increase the depth of the tree.

        :param max_depth: optional maximum depth of search tree
        :return: the modified Search object.
        """
        if max_depth is None:
            self.manifold.build(criterion.Layer[-1])
        elif max_depth < 1:
            raise ValueError(f'Expected a positive integer for max_depth. Got {max_depth} instead.')
        elif max_depth > self.depth:
            self.manifold.build_tree(criterion.MaxDepth(max_depth), criterion.Layer(-1))
        return self

    def rnn(self, query: Datum, radius: float, *, max_depth: Optional[int] = None) -> Results:
        """ Performs rho-nearest neighbors search around query with given radius.

        :param query: point around which to search.
        :param radius: distance from query within which te return all hits.
        :param max_depth: optional maximum search depth.
        :return: dictionary of index-of-hit -> distance-to-hit.
        """
        candidate_clusters: ClusterResults = self.tree_search(query, radius, self.root, max_depth=max_depth)
        return self.leaf_search(query, radius, candidate_clusters)

    def rnn_points(self, query: Datum, radius: float, *, max_depth: Optional[int] = None) -> Data:
        """ Same as rnn, but returns the raw points rather than a dictionary of indices and distances. """
        results = [(d, p) for p, d in self.rnn(query, radius, max_depth=max_depth).items()]
        return self.data[[p for _, p in sorted(results)]]

    def knn(self, query: Datum, k: int, *, max_depth: Optional[int] = None) -> Results:
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
                lfd = 1 if half_count == 0 else np.log2(len(hits) / half_count)

            # increase radius as guided by lfd
            factor = (k / len(hits)) ** (1 / (lfd + EPSILON))
            assert factor > 1, f'expected factor to be greater than 1. Got {factor:.2e} instead.'
            radius *= factor

            # rerun rnn search
            hits = self.rnn(query, radius, max_depth=max_depth)

        # sort hits in non-decreasing order of distance to query
        results: List[Tuple[float, int]] = list(sorted([(distance, point) for point, distance in hits.items()]))
        return {point: distance for distance, point in results[:k]}

    def knn_points(self, query: Datum, k: int, *, max_depth: Optional[int] = None) -> Data:
        """ Same as knn, but returns the raw points rather than a dictionary of indices and distances. """
        results = [(d, p) for p, d in self.knn(query, k, max_depth=max_depth).items()]
        return self.data[[p for _, p in sorted(results)]]

    def _check_shape(self, points: np.ndarray) -> None:
        if len(points.shape) != len(self.data.shape):
            raise ValueError(f'wrong number of dimensions in shape. Expected {len(self.data.shape)}, but got {len(points.shape)}')
        elif points.shape[1:] != self.data.shape[1:]:
            raise ValueError(f'wrong dimensionality of points. Expected {self.data.shape[1:]}, but got {points.shape[1:]}')
        else:
            return

    def _parse_query(self, query: Datum) -> np.ndarray:
        if isinstance(query, int):
            query: np.ndarray = np.asarray([self.data[query]])
        query = np.expand_dims(query, 0)
        self._check_shape(query)
        return query

    def tree_search(self, query: Datum, radius: float, start: Cluster, *, max_depth: Optional[int] = None) -> ClusterResults:
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
        return self.tree_search_history(query, radius, start, max_depth=max_depth)[1]

    def tree_search_history(self, query: Datum, radius: float, start: Cluster, *, max_depth: Optional[int] = None) -> Tuple[ClusterResults, ClusterResults]:
        """ Same as tree-search, except that it also returns the history of candidate clusters at each depth. """
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
            if len(candidates) == 0:
                break
            # find distances to centers of candidate clusters
            clusters: List[Cluster] = list(candidates.keys())
            centers: np.ndarray = np.asarray([cluster.center for cluster in clusters])
            self._check_shape(centers)
            distances: List[float] = list(self.distance(query, centers)[0])

            candidates = {  # filter out clusters that are too far away
                cluster: distance for cluster, distance in zip(clusters, distances)
                if distance <= (radius + cluster.radius)
            }
            subsumed: Set[Cluster] = {  # set of clusters that lie completely within radius of query
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

        return history, hits

    def leaf_search(self, query: Datum, radius: float, clusters: ClusterResults) -> Results:
        """ Performs leaf search for query on given clusters.

        :param query: point around which to look.
        :param radius: distance from point within which to look.
        :param clusters: candidate clusters which need to be exhaustively searched.
        :return: dictionary of index-of-hit -> distance-to-hit.
        """
        query = self._parse_query(query)

        # initialize hits
        hits: Results = dict()

        # get indices of all points in candidate clusters
        argpoints: List[int] = [point for cluster in clusters for point in cluster.argpoints]

        # get distances from query to candidate points
        points: np.ndarray = self.data[argpoints]
        self._check_shape(points)
        distances: List[float] = list(self.distance(query, points)[0])

        # update hits with points within radius of query
        hits.update({point: distance for point, distance in zip(argpoints, distances) if distance <= radius})

        return hits
