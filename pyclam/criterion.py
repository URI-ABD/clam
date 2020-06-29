import logging
from abc import ABC, abstractmethod
from typing import Set, Tuple, List

import numpy as np
from scipy.spatial.distance import cdist

from pyclam.manifold import Cluster, Graph

# TODO: class ChildTooSmall which checks % of parent owned by child, relative populations of child parent and root.
# TODO: RE above, Sparseness which looks at % owned / radius or similar


class Criterion(ABC):
    pass


class ClusterCriterion(Criterion):

    @abstractmethod
    def __call__(self, cluster: Cluster) -> bool:
        pass


class SelectionCriterion(Criterion):

    @abstractmethod
    def __call__(self, root: Cluster) -> Set[Cluster]:
        pass


class GraphCriterion(Criterion):

    @abstractmethod
    def __call__(self, graph: Graph) -> Graph:
        pass


class MaxDepth(ClusterCriterion):
    """ Allows clustering up until the given depth.
    """

    def __init__(self, depth):
        self.depth = depth

    def __call__(self, cluster: Cluster) -> bool:
        return cluster.depth < self.depth


class AddLevels(ClusterCriterion):
    """ Allows clustering up until current.depth + depth.
    """

    def __init__(self, depth):
        self.depth = depth
        self.start = None

    def __call__(self, cluster: Cluster) -> bool:
        if self.start is None:
            self.start = cluster.depth
        return cluster.depth < (self.start + self.depth)


class MinPoints(ClusterCriterion):
    """ Allows clustering up until there are fewer than points.
    """

    def __init__(self, points):
        self.min_points = points

    def __call__(self, cluster: Cluster) -> bool:
        return cluster.cardinality > self.min_points


class MinRadius(ClusterCriterion):
    """ Allows clustering until cluster.radius is less than radius.
    """

    def __init__(self, radius):
        self.radius = radius

    def __call__(self, cluster: Cluster) -> bool:
        return cluster.radius > self.radius


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


class LFDRange(SelectionCriterion):

    def __init__(self, upper: float, lower: float):
        if not (0. < lower <= upper < 100.):
            raise ValueError(f'LFDRange expected 0 < lower <= upper < 100 for upper and lower thresholds.'
                             f'Got: lower: {lower:.2f}, and upper: {upper:.2f}')

        self.upper: float = float(upper)
        self.lower: float = float(lower)

    def __call__(self, root: Cluster) -> Set[Cluster]:
        upper, lower, grace_depth = self._range(root)
        return self._select(root, upper, lower, grace_depth)

    def _range(self, root: Cluster) -> Tuple[float, float, int]:
        manifold = root.manifold
        lfd_range = [], []
        for depth in range(1, manifold.depth):
            if manifold.layers[depth + 1].cardinality < 2 ** (depth + 1):
                clusters: List[Cluster] = [cluster for cluster in manifold.layers[depth] if cluster.cardinality > 2]
                if len(clusters) > 0:
                    lfds = np.percentile(
                        a=[c.local_fractal_dimension for c in clusters],
                        q=[self.upper, self.lower],
                    )
                    lfd_range[0].append(lfds[0]), lfd_range[1].append(lfds[1])
        if len(lfd_range[0]) > 0:
            upper: float = float(np.median(lfd_range[0]))
            lower: float = float(np.median(lfd_range[1]))
            depth: int = manifold.depth - len(lfd_range[0])
            return upper, lower, depth
        else:
            lfds = np.percentile(
                a=[cluster.local_fractal_dimension for cluster in manifold.layers[-1]],
                q=[self.upper, self.lower],
            )
            return float(lfds[0]), float(lfds[1]), manifold.depth
        pass

    @staticmethod
    def _select(root: Cluster, upper: float, lower: float, grace_depth: int) -> Set[Cluster]:
        # childless clusters at or before grace depth are automatically selected
        selected: Set[Cluster] = {
            cluster
            for cluster in root.manifold.layers[grace_depth].clusters
            if not cluster.children
        }
        inactive: Set[Cluster] = {
            cluster
            for cluster in root.manifold.layers[grace_depth].clusters
            if cluster.children
        }
        active: Set[Cluster] = set()

        while active or inactive:
            # Select childless clusters from inactive set
            childless: Set[Cluster] = {cluster for cluster in inactive if len(cluster.children) == 0}
            # Select childless clusters from active set
            childless.update({cluster for cluster in active if len(cluster.children) == 0})
            selected.update(childless)
            inactive -= childless
            active -= childless

            # Select active clusters that fall below the lower threshold
            selections: Set[Cluster] = {cluster for cluster in active if cluster.local_fractal_dimension <= lower}
            selected.update(selections)
            active -= selections

            # activate branches that rise above the upper threshold
            activations: Set[Cluster] = {cluster for cluster in inactive if cluster.local_fractal_dimension >= upper}
            active.update(activations)
            inactive -= activations

            # replace active and inactive sets with child clusters
            active = {child for cluster in active for child in cluster.children}
            inactive = {child for cluster in inactive for child in cluster.children}

        return selected
