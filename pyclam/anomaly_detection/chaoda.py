""" This module provides the CHAODA algorithms implemented on top of CLAM.
"""
import logging
import typing

import numpy

from . import graph_selectors
from . import individual_algorithms
from .. import core
from .. import search
from .. import utils
from ..core import criterion
from ..utils import constants

logger = logging.getLogger(__name__)
logger.setLevel(constants.LOG_LEVEL)

VotingMode = typing.Literal[
    'mean',
    'product',
    'median',
    'min',
    'max',
    'p25',
    'p75',
]


class CHAODA:
    # TODO: Allow weights for each method in ensemble. Perhaps with an Ensembler class?

    def __init__(
            self, *,
            metrics: typing.Optional[list[str]] = None,  # TODO: convert this to a metric class to allow for custom metrics from users.
            partition_criteria: typing.Optional[list[criterion.ClusterCriterion]] = None,
            selector_scorers: typing.Optional[list[tuple[graph_selectors.GraphSelector, list[individual_algorithms.GraphScorer]]]] = None,
            normalization_mode: str = 'gaussian',
            use_speed_threshold: bool = True,
            voting_mode: VotingMode = 'mean',
    ):
        """ Creates and initializes a CHAODA object.

        Args:
            metrics: list of distance metrics to use for constructing manifolds.
            partition_criteria: list of criteria for partitioning clusters when building trees in the manifolds.
            selector_scorers: list of 2-tuples whose items are
                - a trained meta-ml model for selecting a graph from a manifold.
                - a list of individual algorithms to run on that graph.
            normalization_mode: What normalization mode to use. Must be one of 'linear', 'gaussian', or 'sigmoid'.
            use_speed_threshold: number of clusters above which to skip the slow methods.
            voting_mode: voting method to use to aggregate scores in the ensemble
        """
        self.metrics: list[str] = metrics or ['euclidean', 'cityblock']
        self.partition_criteria: list[criterion.ClusterCriterion] = partition_criteria or [
            criterion.MaxDepth(25),
            criterion.MinPoints(1),
        ]

        self.selector_scorers: list[tuple[graph_selectors.GraphSelector, list[individual_algorithms.GraphScorer]]] = selector_scorers or list()
        if len(self.selector_scorers) == 0:
            for selector in graph_selectors.DEFAULT_SELECTORS:
                scorers = [
                    scorer for scorer in individual_algorithms.DEFAULT_SCORERS
                    if scorer.name == selector.scorer_name
                ]
                self.selector_scorers.append((selector, scorers))

        self.normalization_mode = normalization_mode
        self.use_speed_threshold = use_speed_threshold
        self.voting_mode = voting_mode

        self.bowls = [
            SingleMetricChaoda(metric, self.partition_criteria, self.selector_scorers, self.normalization_mode, self.use_speed_threshold)
            for metric in self.metrics
        ]

        self.__scores: typing.Union[numpy.ndarray, utils.Unset] = constants.UNSET

    def fit(self, data: numpy.ndarray, *, indices: typing.Optional[list[int]] = None) -> 'CHAODA':
        indices = indices or list(range(data.shape[0]))
        self.bowls = [bowl.fit(data, indices) for bowl in self.bowls]
        individual_scores = numpy.stack([bowl.scores for bowl in self.bowls])
        self.__scores = self.__vote(individual_scores)
        return self

    def __vote(self, scores: numpy.ndarray) -> numpy.ndarray:
        """ Vote among individual scores for ensemble score.
        """
        if self.voting_mode == 'mean':
            scores = numpy.mean(scores, axis=0)
        elif self.voting_mode == 'product':
            scores = numpy.product(scores, axis=0)
        elif self.voting_mode == 'median':
            scores = numpy.median(scores, axis=0)
        elif self.voting_mode == 'min':
            scores = numpy.min(scores, axis=0)
        elif self.voting_mode == 'max':
            scores = numpy.max(scores, axis=0)
        elif self.voting_mode == 'p25':
            scores = numpy.percentile(scores, 25, axis=0)
        elif self.voting_mode == 'p75':
            scores = numpy.percentile(scores, 75, axis=0)
        else:
            # TODO: Investigate other voting methods.
            raise NotImplementedError(f'voting mode {self.voting_mode} is not implemented. Try one of {VotingMode}.')

        return scores

    @property
    def scores(self) -> numpy.ndarray:
        if isinstance(self.__scores, utils.Unset):
            raise ValueError(f'Please call the `fit` method before fetching anomaly scores.')
        return self.__scores

    def fit_predict(self, data: numpy.ndarray, *, indices: typing.Optional[list[int]] = None) -> numpy.ndarray:
        self.fit(data, indices=indices)
        return self.scores

    def predict_single(self, query: numpy.array) -> float:
        """ Predict the anomaly score for a single query.
        """
        scores = list()
        for bowl in self.bowls:
            searcher = search.CAKES.from_manifold(bowl.manifold)

            for cluster_scores in bowl.cluster_scores_list:
                hits = list(searcher.tree_search_history(query, radius=0)[0].keys())

                intersection = [cluster for cluster in hits if cluster in cluster_scores]
                if len(intersection) > 0:
                    individual_scores = [cluster_scores[cluster] for cluster in intersection]
                else:
                    individual_scores = [1.]

                scores.append(self.__vote(numpy.asarray(individual_scores, dtype=numpy.float32)))

        score = self.__vote(numpy.asarray(scores, dtype=numpy.float32))
        return float(score)

    def predict(self, queries: numpy.array) -> numpy.array:
        """ Predict the anomaly score for a 2d array of queries.
        """
        scores = list()
        for i in range(queries.shape[0]):
            logger.info(f'Predicting anomaly score for query {i} ...')
            scores.append(self.predict_single(queries[i]))
        return numpy.array(scores, dtype=numpy.float32)


class SingleMetricChaoda:
    # TODO: metric argument and type: replace with a Metric class
    def __init__(
            self,
            metric: str,
            partition_criteria: list[criterion.ClusterCriterion],
            selectors_scorers: list[tuple[graph_selectors.GraphSelector, list[individual_algorithms.GraphScorer]]],
            normalization_mode: str,
            use_speed_threshold: bool,
    ):
        """ This class is a single-metric version of CHAODA. This is intended to
         be used from the CHAODA class. All input validation should be done in
         CHAODA and not here.

        Args:
            metric: name of the metric used.
            partition_criteria:
            selectors_scorers: list of 2-tuples whose items are
                - a trained meta-ml model for selecting clusters to build graphs.
                - a list of individual algorithms for predicting anomaly scores
                  from that graph.
            normalization_mode: method to use to normalize anomaly scores from different scoring algorithms.
            use_speed_threshold: Whether to only run the fast algorithms in the ensemble.
        """

        self.metric = metric
        self.partition_criteria = partition_criteria
        self.selectors_scorers = selectors_scorers
        self.normalization_mode = normalization_mode
        self.use_speed_threshold = use_speed_threshold

        self.__manifold: typing.Union[core.Manifold, utils.Unset] = constants.UNSET
        self.__cluster_scores_list: typing.Union[list[individual_algorithms.ClusterScores], utils.Unset] = constants.UNSET
        self.__scores: typing.Union[numpy.ndarray, utils.Unset] = constants.UNSET

    @property
    def manifold(self) -> core.Manifold:
        if isinstance(self.__manifold, utils.Unset):
            raise ValueError(f'Please call the `fit` method before fetching the manifold.')
        return self.__manifold

    @property
    def scores(self) -> numpy.ndarray:
        if isinstance(self.__scores, utils.Unset):
            raise ValueError(f'Please call the `fit` method before fetching anomaly scores.')
        return self.__scores

    @property
    def cluster_scores_list(self) -> list[individual_algorithms.ClusterScores]:
        if isinstance(self.__cluster_scores_list, utils.Unset):
            raise ValueError(f'Please call the `fit` method before fetching dict of cluster scores.')
        return self.__cluster_scores_list

    def fit(self, data: numpy.ndarray, indices: list[int]) -> 'SingleMetricChaoda':
        """ Build a Manifold on the given data, create optimal Graphs, and predict anomaly scores for each Graph.

        Args:
            data: A 2d array where rows are instances and columns are features.
            indices: A list of indices of rows from data on which to fit the model.
        """
        self.__manifold = core.Manifold(data, self.metric, indices).build(*self.partition_criteria)
        self.__manifold.layers[-1].build_edges()

        graphs_scorers = [
            (core.Graph(*selector(self.__manifold.root)).build_edges(), scorers)
            for selector, scorers in self.selectors_scorers
        ]

        if self.use_speed_threshold:
            individual_scores = [
                method(graph)
                for graph, scorers in graphs_scorers
                for method in scorers
                if method.should_be_fast(graph)
            ]
            if len(individual_scores) == 0:
                uniform_cluster_scores = {self.__manifold.root: .5}
                uniform_scores = 0.5 * numpy.ones(shape=(len(indices,)))
                individual_scores.append((uniform_cluster_scores, uniform_scores))
        else:
            individual_scores = [
                method(graph)
                for graph, scorers in graphs_scorers
                for method in scorers
            ]

        [cluster_scores, scores] = list(zip(*individual_scores))
        self.__scores = numpy.stack(scores)
        self.__cluster_scores_list = cluster_scores
        return self
