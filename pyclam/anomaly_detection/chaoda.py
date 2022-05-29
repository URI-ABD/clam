""" This module provides the CHAODA algorithms implemented on top of CLAM.
"""
import logging
import typing

import numpy

from .pretrained_models import META_ML_MODELS
from ..core import Cluster
from ..core import criterion
from ..core import Graph
from ..core import Manifold
from ..core import types
from ..search import CAKES
from ..utils import constants
from ..utils import helpers

logger = logging.getLogger(__name__)
logger.setLevel(constants.LOG_LEVEL)

ClusterScores = dict[Cluster, float]
Scores = dict[int, float]
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
            metrics: list[types.Metric] = None,
            min_depth: int = 4,
            max_depth: int = 25,
            min_points: int = 1,
            meta_ml_functions: list[tuple[str, str, typing.Callable[[numpy.array], float]]] = None,
            normalization_mode: typing.Optional[str] = 'gaussian',
            speed_threshold: typing.Optional[int] = 128,
    ):
        """ Creates and initializes a CHAODA object.

        :param metrics: A list of distance metrics to use for creating manifolds.
                        A metric must deterministically produce positive real numbers for each pair of instances.
                        A metric need not obey the triangle inequality.
                        Any such metrics allowed by scipy are allowed here, as are user-defined functions (so long as scipy accepts them).
        :param min_depth: The minimum depth clusters that can be selected for optimal graphs using meta-ml.
        :param max_depth: The max-depth of the cluster-trees in the manifolds.
        :param min_points: The minimum number of points in a cluster before it can be partitioned further.
        :param meta_ml_functions: A list of tuples of (metric-name, method-name, meta-ml-ranking-function).
        :param normalization_mode: What normalization mode to use. Must be one of 'linear', 'gaussian', or 'sigmoid'.
        :param speed_threshold: number of clusters above which to skip the slow methods.
        """
        self.metrics: list[types.Metric] = metrics or ['euclidean', 'cityblock']

        # Set criteria for building manifolds
        self.max_depth: int = max_depth
        self.min_points: int = min_points

        # Set meta-ml selection criteria
        self.min_depth: int = min_depth

        meta_ml_functions = META_ML_MODELS if meta_ml_functions is None else meta_ml_functions

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
            self._criteria[metric][method].append(criterion.MetaMLSelect(function, self.min_depth))

        self.speed_threshold: typing.Optional[int] = speed_threshold

        if normalization_mode is not None:
            helpers.catch_normalization_mode(normalization_mode)
        self.normalization_mode: typing.Optional[str] = normalization_mode

        # dict of name -> individual-method
        self._names: dict[str, typing.Callable[[Graph], ClusterScores]] = {
            'cluster_cardinality': self._cluster_cardinality,
            'component_cardinality': self._component_cardinality,
            'graph_neighborhood': self._graph_neighborhood,
            'parent_cardinality': self._parent_cardinality,
            'stationary_probabilities': self._stationary_probabilities,
            'vertex_degree': self._vertex_degree,
        }
        self.slow_methods: list[str] = [
            'component_cardinality',
            'graph_neighborhood',
            'stationary_probabilities',
        ]

        # Values to be set later

        self._voting_mode: str = None

        # list of manifolds build during the fit method.
        self._manifolds: list[Manifold] = None

        # These graphs are used with the individual-methods.
        #  This is a list of tuples os (method-name, graph).
        self._graphs: list[tuple[str, Graph]] = None

        # Each item in the list corresponds to a Graph and is a dictionary of
        # the outlier-scores for the Clusters in that Graph.
        self._cluster_scores: list[ClusterScores] = None

        # A list of all outlier-scores arrays to be voted amongst.
        self._individual_scores: list[numpy.array] = None

        # outlier-scores for each point in the fitted data, after voting.
        self._scores: numpy.array = None

    @property
    def manifolds(self) -> list[Manifold]:
        """ Returns the list of manifolds being used for anomaly-detection. """
        return self._manifolds

    @property
    def scores(self) -> numpy.ndarray:
        """ Returns the scores for data on which the model was last fit. """
        if self._scores is None:
            raise ValueError(f'Scores are currently empty. Please call the fit method.')
        return self._scores

    @property
    def individual_scores(self) -> list[numpy.ndarray]:
        if self._individual_scores is None:
            raise ValueError(f'Individual-scores are currently empty. Please call the fit method.')
        return self._individual_scores

    def build_manifolds(self, data: numpy.array, indices: list[int] = None) -> list[Manifold]:
        """ Builds the list of manifolds for the class.

        :param data: numpy array of data where the rows are instances and the columns are features.
        :param indices: Optional. List of indexes of data to which to restrict the Manifolds.
        :return: The list of manifolds.
        """
        indices = list(range(data.shape[0])) if indices is None else indices
        criteria: list[criterion.Criterion] = [criterion.MaxDepth(self.max_depth), criterion.MinPoints(self.min_points)]
        self._manifolds = [Manifold(data, metric, indices).build(*criteria) for metric in self.metrics]
        return self._manifolds

    def fit(self, data: numpy.array, *, indices: list[int] = None, voting: str = 'mean') -> 'CHAODA':
        """ Fits the anomaly detector to the data.

        :param data: numpy array of data where the rows are instances and the columns are features.
        :param indices: Optional. List of indexes of data to which to restrict CHAODA.
        :param voting: How to vote among scores.
        :return: the fitted CHAODA model.
        """
        if voting not in VOTING_MODES:
            raise ValueError(f'\'voting\' must be one of {VOTING_MODES}. Got {voting} instead.')
        self._voting_mode = voting

        self.build_manifolds(data, indices)
        self._graphs = list()
        for manifold in self._manifolds:
            for method, meta_ml_criteria in self._criteria[manifold.metric].items():
                manifold.add_graphs(*meta_ml_criteria)
                for graph in manifold.graphs[-len(meta_ml_criteria):]:
                    if method in self.slow_methods and graph.cardinality > self.speed_threshold:
                        continue
                    self._graphs.append((method, graph))
        self._ensemble()
        return self

    def predict_single(self, query: numpy.array) -> float:
        """ Predict the anomaly score for a single query.
        """
        scores = list()
        for cluster_scores in self._cluster_scores:
            manifold = next(iter(cluster_scores.keys())).manifold
            searcher = CAKES.from_manifold(manifold)
            hits = list(searcher.tree_search_history(query, radius=0)[0].keys())
            intersection = [cluster for cluster in hits if cluster in cluster_scores]
            if len(intersection) > 0:
                individual_scores = [cluster_scores[cluster] for cluster in intersection]
            else:
                individual_scores = [1.]
            scores.append(self._vote(individual_scores))
        score = self._vote(scores)
        return score

    def predict(self, queries: numpy.array) -> numpy.array:
        """ Predict the anomaly score for a 2d array of queries.
        """
        scores = list()
        for i in range(queries.shape[0]):
            logger.info(f'Predicting anomaly score for query {i} ...')
            scores.append(self.predict_single(queries[i]))
        return numpy.array(scores, dtype=numpy.float32)

    def vote(self, replace: bool = True) -> numpy.array:
        """ Get ensemble scores with custom voting and normalization

        :param replace: whether to replace internal scores with new scores.
        :return: 1-d array of outlier scores for fitted data.
        """
        scores = self._vote(numpy.stack(self._individual_scores))
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

    def parent_cardinality(self, graph: Graph, weight: typing.Callable[[int], float] = None) -> Scores:
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

    def _ensemble(self):
        """ Ensemble of individual methods.
        """
        self._cluster_scores = list()
        argpoints = self._manifolds[0].root.argpoints
        self._individual_scores = list()
        for name, graph in self._graphs:
            method = self._names[name]
            self._cluster_scores.append(method(graph))
            scores: Scores = self._score_points(self._cluster_scores[-1])
            self._individual_scores.append(numpy.asarray([scores[i] for i in argpoints], dtype=float))

        # store scores for fitted data.
        self._scores = self.vote()
        return

    def _vote(self, scores: numpy.array):
        """ Vote among all individual-scores for ensemble-score.

        :param scores: An array of shape (num_graphs, num_points) of scores to be voted among.
        """
        if self._voting_mode == 'mean':
            scores = numpy.mean(scores, axis=0)
        elif self._voting_mode == 'product':
            scores = numpy.product(scores, axis=0)
        elif self._voting_mode == 'median':
            scores = numpy.median(scores, axis=0)
        elif self._voting_mode == 'min':
            scores = numpy.min(scores, axis=0)
        elif self._voting_mode == 'max':
            scores = numpy.max(scores, axis=0)
        elif self._voting_mode == 'p25':
            scores = numpy.percentile(scores, 25, axis=0)
        elif self._voting_mode == 'p75':
            scores = numpy.percentile(scores, 75, axis=0)
        else:
            # TODO: Investigate other voting methods.
            raise NotImplementedError(f'voting mode {self._voting_mode} is not implemented. Try one of {VOTING_MODES}.')

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

    def _normalize_scores(self, scores: typing.Union[ClusterScores, Scores], high: bool) -> typing.Union[ClusterScores, Scores]:
        """ Normalizes scores to lie in the [0, 1] range.

        :param scores: A dictionary of outlier rankings for clusters.
        :param high: True if high scores denote outliers, False if low scores denote outliers.

        :return: A dict of cluster -> normalized outlier score.
        """
        if self.normalization_mode is None:
            normalized_scores: list[float] = [(s if high is True else (1 - s)) for s in scores.values()]
        else:
            sign = 1 if high is True else -1
            normalized_scores: list[float] = list(helpers.normalize(
                values=sign * numpy.asarray([score for score in scores.values()], dtype=float),
                mode=self.normalization_mode,
            ))

        return {cluster: float(score) for cluster, score in zip(scores.keys(), normalized_scores)}

    def _cluster_cardinality(self, graph: Graph) -> ClusterScores:
        logger.info(f'with metric {graph.metric} on graph {list(graph.depth_range)} with {graph.cardinality} clusters.')
        return self._normalize_scores({cluster: cluster.cardinality for cluster in graph.clusters}, False)

    def _component_cardinality(self, graph: Graph) -> ClusterScores:
        logger.info(f'with metric {graph.metric} on graph {list(graph.depth_range)} with {graph.cardinality} clusters.')
        scores: dict[Cluster, int] = {
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
        logger.info(f'Running method GN with metric {graph.metric} on graph {list(graph.depth_range)} with {graph.cardinality} clusters.')
        if not (0 < eccentricity_fraction <= 1):
            raise ValueError(f'eccentricity fraction must be in the (0, 1] range. Got {eccentricity_fraction:.2f} instead.')

        pruned_graph, subsumed_neighbors = graph.pruned_graph

        def _neighborhood_size(start: Cluster, steps: int) -> int:
            """ Returns the number of clusters within 'steps' of 'start'. """
            visited: set[Cluster] = set()
            frontier: set[Cluster] = {start}
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

        scores: dict[Cluster, int] = {
            cluster: _neighborhood_size(cluster, int(pruned_graph.eccentricity(cluster) * eccentricity_fraction))
            for cluster in pruned_graph.clusters
        }
        scores = self._inherit_subsumed(subsumed_neighbors, scores)
        return self._normalize_scores(scores, False)

    def _parent_cardinality(self, graph: Graph, weight: typing.Callable[[int], float] = None) -> ClusterScores:
        logger.info(f'Running method PC with metric {graph.metric} on graph {list(graph.depth_range)} with {graph.cardinality} clusters.')

        weight = (lambda d: 1 / (d ** 0.5)) if weight is None else weight
        scores: ClusterScores = {cluster: 0 for cluster in graph.clusters}
        for cluster in graph:
            ancestry = graph.manifold.ancestry(cluster)
            for i in range(1, len(ancestry)):
                scores[cluster] += (weight(i) * ancestry[i-1].cardinality / ancestry[i].cardinality)
        return self._normalize_scores(scores, True)

    def _stationary_probabilities(self, graph: Graph, steps: int = 16) -> ClusterScores:
        logger.info(f'Running method SP with metric {graph.metric} on graph {list(graph.depth_range)} with {graph.cardinality} clusters.')
        scores: dict[Cluster, float] = {cluster: -1 for cluster in graph.clusters}
        pruned_graph, subsumed_neighbors = graph.pruned_graph
        for component in pruned_graph.components:
            if component.cardinality > 1:
                clusters, matrix = component.as_matrix
                matrix = matrix / numpy.sum(matrix, axis=1)[:, None]

                for _ in range(steps):
                    # TODO: Go until convergence. For now, matrix ^ (2 ^ 16) ought to be enough.
                    matrix = numpy.linalg.matrix_power(matrix, 2)
                steady = matrix
                scores.update({cluster: score for cluster, score in zip(clusters, numpy.sum(steady, axis=0))})
            else:
                scores.update({cluster: 0 for cluster in component.clusters})
        scores = self._inherit_subsumed(subsumed_neighbors, scores)
        return self._normalize_scores(scores, False)

    def _vertex_degree(self, graph: Graph) -> ClusterScores:
        logger.info(f'Running method VD with metric {graph.metric} on graph {list(graph.depth_range)} with {graph.cardinality} clusters.')
        scores: dict[Cluster, float] = {cluster: graph.vertex_degree(cluster) for cluster in graph.clusters}
        return self._normalize_scores(scores, False)
