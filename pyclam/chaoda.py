""" This module provides the CHAODA algorithms implemented on top of CLAM.
"""
import logging
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

import numpy as np

from pyclam.criterion import Criterion
from pyclam.criterion import MaxDepth
from pyclam.criterion import MetaMLSelect
from pyclam.criterion import MinPoints
from pyclam.criterion import PropertyThreshold
from pyclam.manifold import Cluster
from pyclam.manifold import Graph
from pyclam.manifold import Manifold
from pyclam.types import Metric
from pyclam.utils import catch_normalization_mode
from pyclam.utils import normalize

ClusterScores = Dict[Cluster, float]
Scores = Dict[int, float]
VOTING_MODES = [
    'mean',
    'product',
    'median',
    'min',
    'max',
    'p25',
    'p75',
]


class CHAODA:
    method_names = [
        'cluster_cardinality',
        'component_cardinality',
        'graph_neighborhood',
        'parent_cardinality',
        'stationary_probabilities',
        'vertex_degree',
    ]
    """ This class provides implementations of the CHAODA algorithms on top of the CLAM framework.
    """
    # TODO: Allow weights for each method in ensemble.
    # TODO: Allow meta-ml with users' own datasets.
    # TODO: Look at DSD (diffusion-state-distance) as a possible individual-method.
    # noinspection PyTypeChecker
    def __init__(
            self, *,
            metrics: Optional[List[Metric]] = None,
            max_depth: int = 25,
            min_points: int = 1,
            meta_ml_functions: Optional[List[Tuple[str, str, Callable[[np.array], float]]]] = None,
            cardinality_percentile: Optional[float] = None,
            radius_percentile: Optional[float] = None,
            lfd_percentile: Optional[float] = None,
            normalization_mode: Optional[str] = 'gaussian',
            speed_threshold: Optional[int] = 128,
    ):
        """ Creates and initializes a CHAODA object.

        :param metrics: A list of distance metrics to use for creating manifolds.
                        A metric must deterministically produce positive real numbers for each pair of instances.
                        A metric need not obey the triangle inequality.
                        Any such metrics allowed by scipy are allowed here, as are user-defined functions (so long as scipy accepts them).
        :param max_depth: The max-depth of the cluster-trees in the manifolds.
        :param min_points: The minimum number of points in a cluster before it can be partitioned further.
        :param meta_ml_functions: A list of tuples of (metric-name, method-name, meta-ml-ranking-function).
        :param cardinality_percentile: The percentile of cluster cardinalities immediately under which clusters wil be selected for graphs.
        :param radius_percentile: The percentile of cluster radii immediately under which clusters wil be selected for graphs.
        :param lfd_percentile: The percentile of cluster local-fractal-dimensions immediately above which clusters wil be selected for graphs.
        :param normalization_mode: What normalization mode to use. Must be one of 'linear', 'gaussian', or 'sigmoid'.
        :param speed_threshold: number of clusters above which to skip the slow methods.
        """
        self.metrics: List[Metric] = ['euclidean', 'cityblock'] if metrics is None else metrics

        # Set criteria for building manifolds
        self.max_depth: int = max_depth
        self.min_points: int = min_points

        if meta_ml_functions is None:
            cardinality_percentile = 50 if cardinality_percentile is None else cardinality_percentile
            radius_percentile = 50 if radius_percentile is None else radius_percentile
            lfd_percentile = 25 if lfd_percentile is None else lfd_percentile
            criteria = [
                PropertyThreshold('cardinality', cardinality_percentile, 'below'),
                PropertyThreshold('radius', radius_percentile, 'below'),
                PropertyThreshold('lfd', lfd_percentile, 'above'),
            ]
            self._criteria = {
                metric: {method: criteria for method in self.method_names}
                for metric in self.metrics
            }
        else:
            metrics = {metric for metric, _, _ in meta_ml_functions}
            if not metrics <= set(self.metrics):
                raise ValueError(f'some meta-ml-functions reference metrics not being used by CHAODA. {set(self.metrics) - metrics}')
            methods = {method for _, method, _ in meta_ml_functions}
            if not methods <= set(self.method_names):
                raise ValueError(f'some meta-ml-functions reference methods not being used by CHAODA. {set(self.method_names) - methods}')

            self._criteria = {
                metric: {method: list() for method in self.method_names}
                for metric in self.metrics
            }
            for metric, method, function in meta_ml_functions:
                self._criteria[metric][method].append(MetaMLSelect(function))

        self.speed_threshold: Optional[int] = speed_threshold

        if normalization_mode is not None:
            catch_normalization_mode(normalization_mode)
        self.normalization_mode: Optional[str] = normalization_mode

        # dict of name -> individual-method
        self._names: Dict[str, Callable[[Graph], ClusterScores]] = {
            'cluster_cardinality': self._cluster_cardinality,
            'component_cardinality': self._component_cardinality,
            'graph_neighborhood': self._graph_neighborhood,
            'parent_cardinality': self._parent_cardinality,
            'stationary_probabilities': self._stationary_probabilities,
            'vertex_degree': self._vertex_degree,
        }
        self.slow_methods: List[str] = [
            'component_cardinality',
            'graph_neighborhood',
            'stationary_probabilities',
        ]

        # Values to be set later

        # list of manifolds build during the fit method.
        self._manifolds: Optional[List[Manifold]] = None

        # These graphs are used with the individual-methods.
        #  This is a list of tuples os (method-name, graph).
        self._graphs: Optional[List[Tuple[str, List[Graph]]]] = None

        # A list of all outlier-scores arrays to be voted amongst.
        self._individual_scores: Optional[List[np.array]] = None

        # outlier-scores for each point in the fitted data, after voting.
        self._scores: Optional[np.array] = None

    @property
    def manifolds(self) -> List[Manifold]:
        """ Returns the list of manifolds being used for anomaly-detection. """
        return self._manifolds

    @property
    def scores(self) -> Optional[np.array]:
        """ Returns the scores for data on which the model was last fit. """
        if self._scores is None:
            logging.warning(f'Scores are currently empty. Please call the fit method.')
        return self._scores

    @property
    def individual_scores(self) -> Optional[List[np.array]]:
        if self._individual_scores is None:
            logging.warning(f'Individual-scores are currently empty. Please call the fit method.')
        return self._individual_scores

    def build_manifolds(self, data: np.array) -> List[Manifold]:
        """ Builds the list of manifolds for the class.

        :param data: numpy array of data where the rows are instances and the columns are features.
        :return: The list of manifolds.
        """
        criteria: List[Criterion] = [MaxDepth(self.max_depth), MinPoints(self.min_points)]
        self._manifolds = [Manifold(data, metric).build(*criteria) for metric in self.metrics]
        return self._manifolds

    def fit(self, data: np.array, *, voting: str = 'mean') -> 'CHAODA':
        """ Fits the anomaly detector to the data.

        :param data: numpy array of data where the rows are instances and the columns are features.
        :param voting: How to vote among scores.
        :return: the fitted CHAODA model.
        """
        self.build_manifolds(data)
        self._graphs = list()
        for manifold in self._manifolds:
            for method, meta_ml_criteria in self._criteria[manifold.metric].items():
                manifold.add_graphs(*meta_ml_criteria)
                for graph in manifold.graphs[-len(meta_ml_criteria):]:
                    if method in self.slow_methods and graph.cardinality > self.speed_threshold:
                        continue
                    self._graphs.append((method, graph))
        self._ensemble(voting)
        return self

    def predict(self, *, queries: np.array) -> np.array:
        # TODO: Handle seeing unseen data for live, online anomaly detection
        raise NotImplementedError

    def vote(self, voting: str = 'mean', replace: bool = True) -> np.array:
        """ Get ensemble scores with custom voting and normalization

        :param voting: voting mode to use. Must be one of VOTING_MODES.
        :param replace: whether to replace internal scores with new scores.
        :return: 1-d array of outlier scores for fitted data.
        """
        scores = self._vote(np.stack(self._individual_scores), voting)
        if replace:
            self._scores = scores
        return scores

    def cluster_cardinality(self, graph: Graph) -> Scores:
        """ Determines outlier scores for points by considering the relative cardinalities of the clusters in the graph.

        Points in clusters with relatively low cardinalities are the outliers.

        :param graph: Graph on which to calculate outlier scores.
        :return: A dict of index -> outlier score
        """
        return self._score_points(self._cluster_cardinality(graph))

    def component_cardinality(self, graph: Graph) -> Scores:
        """ Determines outlier scores by considering the relative cardinalities of the connected components of the graph.

        Points in components of relatively low cardinalities are the outliers

        :param graph: Graph on which to calculate outlier scores.
        :return: A dict of index -> outlier score
        """
        return self._score_points(self._component_cardinality(graph))

    def graph_neighborhood(self, graph: Graph, eccentricity_fraction: float = 0.25) -> Scores:
        """ Determines outlier scores by the considering the relative graph-neighborhood of clusters.

        Subsumed clusters are assigned the highest score of all subsuming clusters.
        Points in clusters with relatively small neighborhoods are the outliers.

        :param graph: Graph on which to calculate outlier scores.
        :param eccentricity_fraction: The fraction, in the (0, 1] range, of a cluster's eccentricity for which to compute the size of the neighborhood.
        :return: A dict of index -> outlier score
        """
        return self._score_points(self._graph_neighborhood(graph, eccentricity_fraction))

    def parent_cardinality(self, graph: Graph, weight: Callable[[int], float] = None) -> Scores:
        """ Determines outlier scores for points by considering ratios of cardinalities of parent-child clusters.

        The ratios are weighted by the child's depth in the tree, and are then accumulated for each point in each cluster in the graph.
        Points with relatively high accumulated ratios are the outliers.

        :param graph: Graph on which to calculate outlier scores.
        :param weight: A function for weighing the contribution of each depth.
                       Takes an integer depth and returns a weight.
                       Defaults to 1 / sqrt(depth).
        :return: A dict of index -> outlier score
        """
        return self._score_points(self._parent_cardinality(graph, weight))

    def stationary_probabilities(self, graph: Graph, steps: int = 16) -> Scores:
        """ Compute the Outlier scores based on the convergence of a random walk with weighted edges on each component of the Graph.

        For each component on the graph, compute the convergent transition matrix for that graph.
        Clusters with low values in that matrix are the outliers.

        :param graph: The graph on which to compute outlier scores.
        :param steps: number of steps to wait for convergence.
        :return: A dict of index -> outlier score
        """
        return self._score_points(self._stationary_probabilities(graph, steps))

    def vertex_degree(self, graph: Graph) -> Scores:
        """ Compute the Outlier scores based on the degree of each vertex of the Graph.

        For each cluster in teh graph, its outlier score is inversely proportional to its degree in the graph.

        :param graph: The graph on which to compute outlier scores.
        :return: A dict of index -> outlier score
        """
        return self._score_points(self._vertex_degree(graph))

    def _ensemble(self, voting: str):
        """ Ensemble of individual methods.

        :param voting: How to vote among scores.
        """
        argpoints = self._manifolds[0].root.argpoints
        self._individual_scores = list()
        for name, graph in self._graphs:
            method = self._names[name]
            scores: Scores = self._score_points(method(graph))
            self._individual_scores.append(np.asarray([scores[i] for i in argpoints], dtype=float))

        # store scores for fitted data.
        self._scores = self.vote(voting)
        return

    # noinspection PyMethodMayBeStatic
    def _vote(self, scores: np.array, mode: str):
        """ Vote among all individual-scores for ensemble-score.

        :param scores: An array of shape (num_graphs, num_points) of scores to be voted among.
        :param mode: Voting mode to use. Must be one of VOTING_MODES
        """
        if mode not in VOTING_MODES:
            raise ValueError(f'voting mode {mode} is invalid. Must be one of {VOTING_MODES}.')

        if mode == 'mean':
            scores = np.mean(scores, axis=0)
        elif mode == 'product':
            scores = np.product(scores, axis=0)
        elif mode == 'median':
            scores = np.median(scores, axis=0)
        elif mode == 'min':
            scores = np.min(scores, axis=0)
        elif mode == 'max':
            scores = np.max(scores, axis=0)
        elif mode == 'p25':
            scores = np.percentile(scores, 25, axis=0)
        elif mode == 'p75':
            scores = np.percentile(scores, 75, axis=0)
        else:
            # TODO: Investigate other voting methods.
            pass

        return scores

    # noinspection PyMethodMayBeStatic
    def _score_points(self, scores: ClusterScores) -> Scores:
        """ Translate scores for clusters into scores for points. """
        scores = {point: float(score) for cluster, score in scores.items() for point in cluster.argpoints}
        for i in self._manifolds[0].root.argpoints:
            if i not in scores:
                # TODO: Rip off this band-aid and investigate source of missing points
                #  So far, this only triggers for exactly one point in exactly one dataset from our suite of datasets.
                #  No idea why.
                scores[i] = 0.5
        return scores

    def _normalize_scores(self, scores: Union[ClusterScores, Scores], high: bool) -> Union[ClusterScores, Scores]:
        """ Normalizes scores to lie in the [0, 1] range.

        :param scores: A dictionary of outlier rankings for clusters.
        :param high: True if high scores denote outliers, False if low scores denote outliers.

        :return: A dict of cluster -> normalized outlier score.
        """
        if self.normalization_mode is None:
            normalized_scores: List[float] = [(s if high is True else (1 - s)) for s in scores.values()]
        else:
            sign = 1 if high is True else -1
            normalized_scores: List[float] = list(normalize(
                values=sign * np.asarray([score for score in scores.values()], dtype=float),
                mode=self.normalization_mode,
            ))

        return {cluster: float(score) for cluster, score in zip(scores.keys(), normalized_scores)}

    def _cluster_cardinality(self, graph: Graph) -> ClusterScores:
        logging.info(f'with metric {graph.metric} on graph {list(graph.depth_range)} with {graph.cardinality} clusters.')
        return self._normalize_scores({cluster: cluster.cardinality for cluster in graph.clusters}, False)

    def _component_cardinality(self, graph: Graph) -> ClusterScores:
        logging.info(f'with metric {graph.metric} on graph {list(graph.depth_range)} with {graph.cardinality} clusters.')
        scores: Dict[Cluster, int] = {
            cluster: component.cardinality
            for component in graph.components
            for cluster in component.clusters
        }
        return self._normalize_scores(scores, False)

    # noinspection PyMethodMayBeStatic
    def _inherit_subsumed(self, subsumed_neighbors, scores):
        for master, subsumed in subsumed_neighbors.items():
            for cluster in subsumed:
                if cluster in scores:
                    scores[cluster] = max(scores[cluster], scores[master])
                else:
                    scores[cluster] = scores[master]
        return scores

    def _graph_neighborhood(self, graph: Graph, eccentricity_fraction: float = 0.25) -> ClusterScores:
        logging.info(f'Running method GN with metric {graph.metric} on graph {list(graph.depth_range)} with {graph.cardinality} clusters.')
        if not (0 < eccentricity_fraction <= 1):
            raise ValueError(f'eccentricity fraction must be in the (0, 1] range. Got {eccentricity_fraction:.2f} instead.')

        pruned_graph, subsumed_neighbors = graph.pruned_graph

        def _neighborhood_size(start: Cluster, steps: int) -> int:
            """ Returns the number of clusters within 'steps' of 'start'. """
            visited: Set[Cluster] = set()
            frontier: Set[Cluster] = {start}
            for _ in range(steps):
                if frontier:
                    visited.update(frontier)
                    frontier = {
                        neighbor for cluster in frontier
                        for neighbor in pruned_graph.neighbors(cluster)
                        if neighbor not in visited
                    }
                else:
                    break
            return len(visited)

        scores: Dict[Cluster, int] = {
            cluster: _neighborhood_size(cluster, int(pruned_graph.eccentricity(cluster) * eccentricity_fraction))
            for cluster in pruned_graph.clusters
        }
        scores = self._inherit_subsumed(subsumed_neighbors, scores)
        return self._normalize_scores(scores, False)

    def _parent_cardinality(self, graph: Graph, weight: Callable[[int], float] = None) -> ClusterScores:
        logging.info(f'Running method PC with metric {graph.metric} on graph {list(graph.depth_range)} with {graph.cardinality} clusters.')

        weight = (lambda d: 1 / (d ** 0.5)) if weight is None else weight
        scores: ClusterScores = {cluster: 0 for cluster in graph.clusters}
        for cluster in graph:
            ancestry = graph.manifold.ancestry(cluster)
            for i in range(1, len(ancestry)):
                scores[cluster] += (weight(i) * ancestry[i-1].cardinality / ancestry[i].cardinality)
        return self._normalize_scores(scores, True)

    def _stationary_probabilities(self, graph: Graph, steps: int = 16) -> ClusterScores:
        logging.info(f'Running method SP with metric {graph.metric} on graph {list(graph.depth_range)} with {graph.cardinality} clusters.')
        scores: Dict[Cluster, float] = {cluster: -1 for cluster in graph.clusters}
        pruned_graph, subsumed_neighbors = graph.pruned_graph
        for component in pruned_graph.components:
            if component.cardinality > 1:
                clusters, matrix = component.as_matrix
                matrix = matrix / np.sum(matrix, axis=1)[:, None]

                for _ in range(steps):
                    # TODO: Go until convergence. For now, matrix ^ (2 ^ 16) ought to be enough.
                    matrix = np.linalg.matrix_power(matrix, 2)
                steady = matrix
                scores.update({cluster: score for cluster, score in zip(clusters, np.sum(steady, axis=0))})
            else:
                scores.update({cluster: 0 for cluster in component.clusters})
        scores = self._inherit_subsumed(subsumed_neighbors, scores)
        return self._normalize_scores(scores, False)

    def _vertex_degree(self, graph: Graph) -> ClusterScores:
        logging.info(f'Running method VD with metric {graph.metric} on graph {list(graph.depth_range)} with {graph.cardinality} clusters.')
        scores: Dict[Cluster, float] = {cluster: graph.vertex_degree(cluster) for cluster in graph.clusters}
        return self._normalize_scores(scores, False)
