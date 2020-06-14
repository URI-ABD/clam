import logging
from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.distance import cdist

from pyclam.manifold import Cluster, Graph


class Criterion(ABC):
    pass


class GraphCriterion(Criterion):

    @abstractmethod
    def __call__(self, graph: Graph):
        pass


class ClusterCriterion(Criterion):

    @abstractmethod
    def __call__(self, cluster: Cluster):
        pass


class MaxDepth(ClusterCriterion):
    """ Allows clustering up until the given depth.
    """

    def __init__(self, depth):
        self.depth = depth

    def __call__(self, cluster: Cluster):
        return cluster.depth < self.depth


class AddLevels(ClusterCriterion):
    """ Allows clustering up until current.depth + depth.
    """

    def __init__(self, depth):
        self.depth = depth
        self.start = None

    def __call__(self, cluster: Cluster):
        if self.start is None:
            self.start = cluster.depth
        return cluster.depth < (self.start + self.depth)


class MinPoints(ClusterCriterion):
    """ Allows clustering up until there are fewer than points.
    """

    def __init__(self, points):
        self.min_points = points

    def __call__(self, cluster: Cluster):
        return cluster.cardinality > self.min_points


# class MinRadius(ClusterCriterion):
#     """ Allows clustering until cluster.radius is less than radius.
#     """
#
#     def __init__(self, radius):
#         self.radius = radius
#         MIN_RADIUS = radius
#
#     def __call__(self, cluster: Cluster):
#         if cluster.radius <= self.radius:
#             return False
#         return True


# class LeavesSubgraph(ClusterCriterion):
#     """ Allows clustering until the cluster has left the subgraph of the parent.
#     """
#
#     def __init__(self, manifold: _Manifold):
#         self.manifold = manifold
#         return
#
#     def __call__(self, cluster: _Cluster):
#         parent_subgraph = self.manifold.layers[cluster.depth - 1].subgraph(self.manifold.select(cluster.name[:-1]))
#         return any((c.overlaps(cluster.medoid, cluster.radius) for c in parent_subgraph))


# class MinCardinality(ClusterCriterion):
#     """ Allows clustering until cardinality of cluster's subgraph is less than given.
#     """
#
#     def __init__(self, cardinality):
#         self.cardinality = cardinality
#
#     def __call__(self, cluster: _Cluster):
#         return any((
#             # If there are fewer points than needed, we don't check cardinality.
#             cluster.manifold.layers[cluster.depth].cardinality <= self.cardinality,
#             cluster.manifold.layers[cluster.depth].subgraph(cluster).cardinality >= self.cardinality
#         ))


# class MinNeighborhood(ClusterCriterion):
#     """ Allows clustering until the size of the neighborhood drops below threshold.
#     """
#
#     def __init__(self, starting_depth: int, threshold: int):
#         self.starting_depth = starting_depth
#         self.threshold = threshold
#         return
#
#     def __call__(self, cluster: _Cluster) -> bool:
#         return cluster.depth < self.starting_depth or len(cluster.neighbors) >= self.threshold


# class NewSubgraph(ClusterCriterion):
#     """ Cluster until a new subgraph is created. """
#
#     def __init__(self, manifold: _Manifold):
#         self.manifold = manifold
#         self.starting = len(manifold.layers[-1].subgraphs)
#         return
#
#     def __call__(self, _):
#         return len(self.manifold.layers[-1].subgraphs) == self.starting


class MedoidNearCentroid(ClusterCriterion):
    def __init__(self):
        return

    def __call__(self, cluster: Cluster) -> bool:
        distance = cdist(np.expand_dims(cluster.centroid, 0), np.expand_dims(cluster.medoid, 0), cluster.metric)[0][0]
        logging.debug(f'Cluster {str(cluster)} distance: {distance}')
        return any((
            cluster.depth < 1,
            distance > (cluster.radius * 0.1)
        ))


class UniformDistribution(ClusterCriterion):
    def __init__(self):
        return

    def __call__(self, cluster: Cluster) -> bool:
        distances = cdist(np.expand_dims(cluster.medoid, 0), cluster.samples)[0] / (cluster.radius + 1e-15)
        logging.debug(f'Cluster: {cluster}. Distances: {distances}')
        freq, bins = np.histogram(distances, bins=[i / 10 for i in range(1, 10)])
        ideal = np.full_like(freq, distances.shape[0] / bins.shape[0])
        from scipy.stats import wasserstein_distance
        distance = wasserstein_distance(freq, ideal)
        return distance > 0.25

# TODO: class ChildTooSmall which checks % of parent owned by child, relative populations of child parent and root.
# TODO: RE above, Sparseness which looks at % owned / radius or similar
