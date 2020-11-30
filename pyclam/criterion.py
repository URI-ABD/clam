import heapq
import logging
from abc import ABC
from abc import abstractmethod
from collections import Counter
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple

import numpy as np
from scipy.spatial.distance import cdist

from pyclam.manifold import Cluster
from pyclam.manifold import Manifold
from pyclam.utils import normalize


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


class Leaves(SelectionCriterion):
    """ Selects the leaves for building the graph.
    """
    def __init__(self):
        return

    def __call__(self, root: Cluster) -> Set[Cluster]:
        return {cluster for cluster in root.manifold.layers[-1].clusters}


class Layer(SelectionCriterion):
    """ Selects the layer at the specified depth.
    """
    def __init__(self, depth: int):
        if depth < -1:
            raise ValueError(f'expected a \'-1\' or a non-negative depth. got: {depth}')
        self.depth = depth

    def __call__(self, root: Cluster) -> Set[Cluster]:
        manifold: Manifold = root.manifold
        if (self.depth == -1) or (manifold.depth <= self.depth):
            return {cluster for cluster in manifold.layers[-1].clusters}
        else:
            return {cluster for cluster in manifold.layers[self.depth].clusters}


class MetaMLSelect(SelectionCriterion):
    """ Uses a decision function from a trained meta-ml model to select best clusters for graph.
    """
    def __init__(self, predict_auc: Callable[[np.array], float], min_depth: int = 4):
        """ A Meta-ML model to select Clusters for a Graph.

        :param predict_auc: A function that takes a cluster's ratios and returns a predicted auc for selecting that cluster.
        """
        self.predict_auc: Callable[[np.array], float] = predict_auc
        if min_depth < 1:
            raise ValueError(f'min-depth must be a positive integer.')
        self.min_depth = min_depth

    def __call__(self, root: Cluster) -> Set[Cluster]:
        def predict_auc(cluster):
            # adjust predicted auc by cardinality to slightly favor clusters with high cardinalities.
            return self.predict_auc(manifold.cluster_ratios(cluster)) * (1 + cluster.cardinality / root.cardinality / 10)

        manifold: Manifold = root.manifold
        predicted_auc: Dict[Cluster, float] = {
            cluster: predict_auc(cluster)
            for cluster in manifold.ordered_clusters
            if cluster.depth >= self.min_depth
        }
        normalized_auc = normalize(np.asarray([-v for v in predicted_auc.values()]), mode='gaussian')
        # Turn dict into max priority queue
        items = list(zip(normalized_auc, predicted_auc.keys()))
        heapq.heapify(items)

        selected: Set[Cluster] = set()
        excluded: Set[int] = set()
        while len(items) > 0:
            _, cluster = heapq.heappop(items)
            if len(excluded.intersection(set(cluster.argpoints))) > 0:
                continue
            else:
                selected.add(cluster)
                excluded.update(set(cluster.argpoints))
        return selected


class LFDRange(SelectionCriterion):
    """ Selects clusters based on when their ancestry crosses certain thresholds of LFD values.
    This works a bit like a two-state machine.
    """
    def __init__(self, upper: float, lower: float):
        if not (0. < lower <= upper <= 100.):
            raise ValueError(f'LFDRange expected 0 < lower <= upper <= 100 for upper and lower thresholds.'
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


class PropertyThreshold(SelectionCriterion):
    """ Selects clusters when the given property crosses the given percentile.

    If mode is 'above', the cluster qualifies if the property goes above the percentile,
    If mode is 'below', the cluster qualifies if the property goes below the percentile.

    During a BFT of the tree, keep track of the given cluster property.
    Select a cluster only if:
        * it is a leaf, or
        * it qualifies by 'mode' as above, or
        * all children qualify by 'mode'.

    :param value: The cluster property to keep track of. Must be one of 'cardinality', 'radius', or 'lfd'.
    :param percentile: the percentile, in the (0, 100) range, to be crossed.
    :param mode: select when 'above' or 'below' the percentile.
    """
    def __init__(self, value: str, percentile: float, mode: str):
        # TODO: Use Literal for value and mode.
        if value == 'cardinality':
            self.value = lambda c: c.cardinality
        elif value == 'radius':
            self.value = lambda c: c.radius
        elif value == 'lfd':
            self.value = lambda c: c.local_fractal_dimension
        else:
            raise ValueError(f'value must be one of \'cardinality\', \'radius\', or \'lfd\'. Got {value} instead.')

        if 0 < percentile < 100:
            self.percentile: float = percentile
        else:
            raise ValueError(f'percentile must be in the (0, 100) range. Gor {percentile:.2f} instead.')

        if mode == 'above':
            self.mode = lambda c, v: self.value(c) > v
        elif mode == 'below':
            self.mode = lambda c, v: self.value(c) < v
        else:
            raise ValueError(f'mode must be \'above\' or \'below\'. Got {mode} instead.')

    def __call__(self, root: Cluster) -> Set[Cluster]:
        threshold = float(np.percentile(
            [self.value(cluster) for cluster in root.descendents if cluster.cardinality > 1],
            self.percentile,
        ))
        selected: Set[Cluster] = set()
        frontier: Set[Cluster] = {root}
        while frontier:
            cluster = frontier.pop()
            if (
                    (cluster.children is None)  # select the leaves
                    or self.mode(cluster, threshold)  # if cluster qualifies
                    or all((map(lambda c: self.mode(c, threshold), cluster.children)))  # if both children qualify
            ):
                selected.add(cluster)
            else:  # add children to frontier
                frontier.update(cluster.children)
        return selected


class Labels(SelectionCriterion):
    """ Uses a given dictionary of index -> label to select optimal clusters.

    During a BFT of the tree, when a cluster has an overwhelming majority of points with the same label,
    that cluster is selected.
    If some data points are unlabelled, then when a cluster contains no labelled points, it too is selected.

    Let 'f' be the fraction of labels in the smallest class in the data,
    and let 't' be a given threshold in the (0, 1] range.
    Then the 'overwhelming majority' is defined as (1 - f * t).
    """
    def __init__(self, labels: Dict[int, Any], threshold: float = 0.1):
        """ Instantiates the Labels Selector.

        :param labels: A dictionary of index -> label for som or all of the points in the data.
        :param threshold: A threshold, in the (0, 1] range for determining the size of the 'overwhelming majority.'

        """
        if not (0 < threshold <= 1):
            ValueError(f'threshold must be in the (0, 1] range.')
        self.labels: Dict[int, Any] = labels
        self.min_fraction: float = min(dict(Counter(labels.values())).values()) / len(labels)
        self.threshold: float = 1 - self.min_fraction * threshold

    def __call__(self, root: Cluster) -> Set[Cluster]:
        selected: Set[Cluster] = set()
        frontier: Set[Cluster] = {root}
        while frontier:
            cluster = frontier.pop()
            if cluster.children is None:
                selected.add(cluster)
            else:
                labels: List[Any] = [self.labels[p] for p in cluster.argpoints if p in self.labels]
                if len(labels) > 0:
                    max_fraction = max(dict(Counter(labels)).values()) / cluster.cardinality
                    if max_fraction > self.threshold:
                        selected.add(cluster)
                    else:
                        frontier.update(cluster.children)
                else:
                    # TODO: Should we revert to some other selection criterion here?
                    selected.add(cluster)
        return selected
