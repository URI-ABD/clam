import logging
from abc import ABC, abstractmethod
from typing import Set, Tuple, List, Union, Dict, Callable

import numpy as np
from scipy.spatial.distance import cdist

from pyclam.manifold import Cluster, Manifold


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
        if depth <= 0:
            raise ValueError(f'expected a positive depth. got: {depth}')
        self.depth = depth

    def __call__(self, root: Cluster) -> Set[Cluster]:
        manifold: Manifold = root.manifold
        if manifold.depth <= self.depth:
            return {cluster for cluster in manifold.layers[-1].clusters}
        else:
            return {cluster for cluster in manifold.layers[self.depth].clusters}


def _percentile_selection(manifold: Manifold, predicted_auc: Dict[Cluster, float]) -> Set[Cluster]:
    """ Selects clusters for graph when their predicted auc is greater than the p-th percentile of the predicted auc of their descendents.
    """
    graph: Set[Cluster] = set()
    clusters: List[Cluster] = [child for child in manifold.root.children]
    while clusters:
        new_clusters: List[Cluster] = list()
        for cluster in clusters:
            if cluster.children:
                subtree_auc = [predicted_auc[c] for c in cluster.descendents]
                if predicted_auc[cluster] >= np.percentile(subtree_auc, q=75):
                    graph.add(cluster)
                else:
                    new_clusters.extend(cluster.children)
            else:
                graph.add(cluster)
        clusters = new_clusters
    return graph


def _ranked_selection(manifold: Manifold, predicted_auc: Dict[Cluster, float]) -> Set[Cluster]:
    """ Rank-orders clusters by predicted_auc and peel off the top to build the graph.
    """
    rankings: Dict[Cluster, float] = {cluster: 0 for cluster in predicted_auc.keys()}
    for leaf in manifold.layers[-1].clusters:
        # sort ancestors by predicted auc in non-increasing order, break ties by smaller depth
        ancestors = list(sorted(manifold.ancestry(leaf), key=lambda c: (predicted_auc[c], c), reverse=True))
        # add index of cluster in sorted list of ancestors to rankings
        for i, cluster in enumerate(ancestors):
            rankings[cluster] += (i * cluster.cardinality)
    rankings.pop(manifold.select(''))  # remove root from rankings because ratios are undefined

    # TODO: misses some points/clusters. breaks graph invariant
    # sort clusters by mean ranking in non-decreasing order
    potentials: List[Cluster] = list(sorted(list(rankings.keys()), key=lambda c: (rankings[c] / c.cardinality, c)))
    selected: Set[Cluster] = set()
    excluded: Set[Cluster] = set()
    # select best clusters for graph
    for cluster in potentials:
        if cluster in excluded:
            continue
        else:
            selected.add(cluster)
            excluded.add(cluster)
            excluded.update(cluster.ancestors)
            excluded.update(cluster.descendents)

    # TODO: remove this after fixing the missed clusters in the initial pass
    missed: Set[int] = set(manifold.root.argpoints) - {p for c in selected for p in c.argpoints}
    for layer in manifold.layers[1:]:
        for cluster in layer.clusters:
            if all((p in missed for p in cluster.argpoints)):
                selected.add(cluster)
                missed.difference_update(cluster.argpoints)

    return selected


SELECTION_MODES = {
    'percentile': _percentile_selection,
    'ranked': _ranked_selection,
}


class LinearRegressionConstants(SelectionCriterion):
    """ Uses constants from a meta-ml model using linear regression.
    There are 6 constants, one for each of the ratios and ema_ratios in Cluster.
    """
    # TODO: Change this to use a parsed function, just like Regression Tree Criterion.

    def __init__(self, constants: Union[np.array, List[float]], *, mode: str = 'ranked'):
        """ A Linear Regression based Meta-ML model to select Clusters for a Graph.

        The constants correspond to:
            * child/parent lfd ratio
            * child/parent cardinality ratio
            * child/parent radii ratio
            * lfd-ratio exponential-moving-average (ema)
            * cardinality-ratio ema
            * radii-ratio ema

        :param constants: 6 constants from the LR model.
        :param mode: which selection method to use. Must be one of 'ranked' or 'percentile'.
        """
        if mode not in SELECTION_MODES.keys():
            raise ValueError(f'mode must be one of {list(SELECTION_MODES.keys())}. Gor {mode} instead.')
        self.mode = SELECTION_MODES[mode]

        num_constants = 6
        constants = np.asarray(constants, dtype=float)
        if constants.shape != (num_constants,):
            raise ValueError(f'expected a vector of {num_constants} elements. Got {constants.shape} instead.')
        self.constants: np.array = constants

    def __call__(self, root: Cluster) -> Set[Cluster]:
        predicted_auc: Dict[Cluster, float] = self._predict_auc(root.manifold)
        return self.mode(root.manifold, predicted_auc)

    def _predict_auc(self, manifold: Manifold) -> Dict[Cluster, float]:
        predicted_auc: Dict[Cluster, float] = {
            cluster: float(np.dot(self.constants, np.concatenate([
                np.asarray(cluster.ratios, dtype=float),
                np.asarray(cluster.ema_ratios, dtype=float)
            ])))
            for layer in manifold.layers
            for cluster in layer.clusters
            if cluster.depth == layer.depth
        }
        return predicted_auc


class RegressionTreePaths(SelectionCriterion):
    """ Uses the decision paths of the Regression Tree meta-ml models to select best clusters for graph.
    """
    def __init__(self, predict_auc: Callable[[Cluster], float], *, mode: str = 'ranked'):
        """ A Regression Tree based Meta-ML model to select Clusters for a Graph.

        :param predict_auc: A function, parsed from a decision tree output, to assign auc prediction to each cluster
        :param mode: which selection method to use. Must be one of 'ranked' or 'percentile'.
        """
        if mode not in ['ranked', 'percentile']:
            raise ValueError(f'mode must be either \'ranked\' or \'percentile\'. Got {mode} instead.')
        self.mode = SELECTION_MODES[mode]
        self.predict_auc: Callable[[Cluster], float] = predict_auc

    def __call__(self, root: Cluster) -> Set[Cluster]:
        predicted_auc: Dict[Cluster, float] = {
            cluster: self.predict_auc(cluster)
            for layer in root.manifold.layers
            for cluster in layer.clusters
            if cluster.depth == layer.depth
        }
        return self.mode(root.manifold, predicted_auc)


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
