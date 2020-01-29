import logging

import numpy as np
from scipy.spatial.distance import cdist

import chess
from chess.manifold import Cluster as _Cluster, Manifold as _Manifold


class MaxDepth:
    """ Allows clustering up until the given depth.
    """

    def __init__(self, depth):
        self.depth = depth

    def __call__(self, cluster: _Cluster):
        return cluster.depth < self.depth


class AddLevels:
    """ Allows clustering up until current.depth + depth.
    """

    def __init__(self, depth):
        self.depth = depth
        self.start = None

    def __call__(self, cluster: _Cluster):
        if self.start is None:
            self.start = cluster.depth
        return cluster.depth < (self.start + self.depth)


class MinPoints:
    """ Allows clustering up until there are fewer than points.
    """

    def __init__(self, points):
        self.points = points

    def __call__(self, cluster: _Cluster):
        return len(cluster) > self.points


class MinRadius:
    """ Allows clustering until cluster.radius is less than radius.
    """

    def __init__(self, radius):
        self.radius = radius
        chess.manifold.MIN_RADIUS = radius

    def __call__(self, cluster: _Cluster):
        if cluster.radius <= self.radius:
            cluster.__dict__['_min_radius'] = self.radius
            return False
        return True


class LeavesSubgraph:
    """ Allows clustering until the cluster has left the subgraph of the parent.
    """

    def __init__(self, manifold: _Manifold):
        self.manifold = manifold
        return

    def __call__(self, cluster: _Cluster):
        parent_subgraph = self.manifold.graphs[cluster.depth - 1].subgraph(self.manifold.select(cluster.name[:-1]))
        return any((c.overlaps(cluster.medoid, cluster.radius) for c in parent_subgraph))


class MinCardinality:
    """ Allows clustering until cardinality of cluster's subgraph is less than given.
    """

    def __init__(self, cardinality):
        self.cardinality = cardinality

    def __call__(self, cluster: _Cluster):
        return any((
            # If there are fewer points than needed, we don't check cardinality.
            len(cluster.manifold.graphs[cluster.depth]) <= self.cardinality,
            len(cluster.manifold.graphs[cluster.depth].subgraph(cluster)) >= self.cardinality
        ))


class MinNeighborhood:
    """ Allows clustering until the size of the neighborhood drops below threshold.
    """

    def __init__(self, starting_depth: int, threshold: int):
        self.starting_depth = starting_depth
        self.threshold = threshold
        return

    def __call__(self, cluster: _Cluster) -> bool:
        return cluster.depth < self.starting_depth or len(cluster.neighbors) >= self.threshold


class NewSubgraph:
    """ Cluster until a new subgraph is created. """

    def __init__(self, manifold: _Manifold):
        self.manifold = manifold
        self.starting = len(manifold.graphs[-1].subgraphs)
        return

    def __call__(self, _):
        return len(self.manifold.graphs[-1].subgraphs) == self.starting


class MedoidNearCentroid:
    def __init__(self):
        return

    def __call__(self, cluster: _Cluster) -> bool:
        distance = cdist(np.expand_dims(cluster.centroid, 0), np.expand_dims(cluster.medoid, 0), cluster.metric)[0][0]
        logging.debug(f'Cluster {str(cluster)} distance: {distance}')
        return any((
            cluster.depth < 1,
            distance > (cluster.radius * 0.1)
        ))


class UniformDistribution:
    def __init__(self):
        return

    def __call__(self, cluster: _Cluster) -> bool:
        distances = cdist(np.expand_dims(cluster.medoid, 0), cluster.samples)[0] / (cluster.radius + 1e-15)
        logging.debug(f'Cluster: {cluster}. Distances: {distances}')
        freq, bins = np.histogram(distances, bins=[i / 10 for i in range(1, 10)])
        ideal = np.full_like(freq, distances.shape[0] / bins.shape[0])
        from scipy.stats import wasserstein_distance
        distance = wasserstein_distance(freq, ideal)
        return distance > 0.25

# TODO: class ChildTooSmall which checks % of parent owned by child, relative populations of child parent and root.
# TODO: RE above, Sparseness which looks at % owned / radius or similar
