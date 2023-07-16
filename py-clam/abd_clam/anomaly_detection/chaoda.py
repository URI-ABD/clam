"""CHAODA: Clustered Hierarchical Anomaly and Outlier Detection Algorithms."""

import inspect
import math
import typing

import numpy

from .. import core
from .. import search
from ..core import cluster_criteria
from ..core import graph_criteria
from ..utils import constants
from ..utils import helpers
from . import graph_scorers
from . import pretrained_models

logger = helpers.make_logger(__name__)

VotingMode = typing.Literal[
    "mean",
    "product",
    "median",
    "min",
    "max",
    "p25",
    "p75",
]

DEFAULT_GRAPH_CRITERIA = [
    graph_criteria.MetaMLSelect(function, min_depth=4)
    for _, function in inspect.getmembers(pretrained_models, inspect.isfunction)
]

DEFAULT_GRAPH_SCORERS: list[graph_scorers.GraphScorer] = [
    graph_scorers.ClusterCardinality(),
    graph_scorers.ComponentCardinality(),
    graph_scorers.VertexDegree(),
    graph_scorers.ParentCardinality(depth_weight=lambda d: 1 / (d**0.5)),
    graph_scorers.GraphNeighborhood(eccentricity_fraction=0.25),
    graph_scorers.StationaryProbabilities(steps=16),
]


class CHAODA:
    """This class is the main CHAODA class."""

    def __init__(  # noqa: PLR0913
        self,
        metric_spaces: typing.Sequence[core.Space],
        *,
        partition_criteria: typing.Optional[
            typing.Sequence[cluster_criteria.ClusterCriterion]
        ] = None,
        selector_scorers: typing.Optional[
            typing.Sequence[
                tuple[
                    graph_criteria.GraphCriterion,
                    typing.Sequence[graph_scorers.GraphScorer],
                ]
            ]
        ] = None,
        normalization_mode: helpers.NormalizationMode = "gaussian",
        use_speed_threshold: bool = True,
        voting_mode: VotingMode = "mean",
    ) -> None:
        """Creates and initializes a CHAODA object.

        Args:
            metric_spaces: A list of metric spaces to use for anomaly detection.
                All metric spaces should have the same `Dataset` object.
            partition_criteria: list of criteria for partitioning clusters when
                building trees.
            selector_scorers: list of 2-tuples whose items are
                - a trained meta-ml model for selecting a graph.
                - a list of individual algorithms to run on that graph.
            normalization_mode: What normalization mode to use. Must be one of
                - 'linear',
                - 'gaussian', or
                - 'sigmoid'.
            use_speed_threshold: Whether to skip slow graph scorers.
            voting_mode: to use to aggregate scores for the ensemble.
        """
        for i, left in enumerate(metric_spaces):
            for right in metric_spaces[i + 1 :]:
                if left.data != right.data:
                    msg = (
                        f"Metric spaces {left.name} and {right.name} "
                        "had different datasets."
                    )
                    raise ValueError(
                        msg,
                    )

        self.__metric_spaces = metric_spaces

        self.__partition_criteria: typing.Sequence[
            cluster_criteria.ClusterCriterion
        ] = partition_criteria or [
            cluster_criteria.MinPoints(
                int(math.log2(self.__metric_spaces[0].data.cardinality)),
            ),
        ]

        self.__selector_scorers: typing.Sequence[
            tuple[
                graph_criteria.GraphCriterion,
                typing.Sequence[graph_scorers.GraphScorer],
            ]
        ] = selector_scorers or [
            (c, [s for s in DEFAULT_GRAPH_SCORERS if s.name in c.name])
            for c in DEFAULT_GRAPH_CRITERIA
        ]

        self.__normalization_mode = normalization_mode
        self.__use_speed_threshold = use_speed_threshold
        self.__voting_mode = voting_mode

        self.__bowls = [
            SingleSpaceChaoda(
                metric_space,
                self.__partition_criteria,
                self.__selector_scorers,
                self.__normalization_mode,
                self.__use_speed_threshold,
            )
            for metric_space in self.__metric_spaces
        ]

        self.__scores: typing.Union[numpy.ndarray, constants.Unset] = constants.UNSET

    def build(self) -> "CHAODA":
        """Build the CHAODA model and compute anomaly scores."""
        self.__bowls = [bowl.build() for bowl in self.__bowls]
        individual_scores = numpy.concatenate([bowl.scores for bowl in self.__bowls])
        self.__scores = self.__vote(individual_scores)
        return self

    def __vote(self, scores: numpy.ndarray) -> numpy.ndarray:
        """Vote among individual scores for ensemble score."""
        if self.__voting_mode == "mean":
            scores = numpy.mean(scores, axis=0)
        elif self.__voting_mode == "product":
            scores = numpy.product(scores, axis=0)
        elif self.__voting_mode == "median":
            scores = numpy.median(scores, axis=0)
        elif self.__voting_mode == "min":
            scores = numpy.min(scores, axis=0)
        elif self.__voting_mode == "max":
            scores = numpy.max(scores, axis=0)
        elif self.__voting_mode == "p25":
            scores = numpy.percentile(scores, 25, axis=0)
        else:  # self.__voting_mode == 'p75'
            scores = numpy.percentile(scores, 75, axis=0)

        return scores

    @property
    def scores(self) -> numpy.ndarray:
        """The ensemble scores for each point in the dataset."""
        if self.__scores is constants.UNSET:
            msg = "Please call the `fit` method before using this property."
            raise ValueError(msg)
        return self.__scores

    def fit_predict(self) -> numpy.ndarray:
        """Fit the model and return the anomaly scores."""
        self.build()
        return self.__scores

    def predict_single(self, query: typing.Any) -> float:  # noqa: ANN401
        """Predict the anomaly score for a single query."""
        scores = []
        for bowl in self.__bowls:
            for cluster_scores in bowl.cluster_scores_list:
                hits = set(bowl.searcher.tree_search_history(query, 0.0)[0].keys())

                intersection = hits.intersection(set(cluster_scores.keys()))
                if len(intersection) > 0:
                    score = self.__vote(
                        numpy.asarray(
                            [cluster_scores[c] for c in intersection],
                            dtype=numpy.float32,
                        ),
                    )
                else:
                    score = 1.0
                scores.append(1.0 if numpy.isnan(score) else float(score))

        final_score = self.__vote(numpy.asarray(scores, dtype=numpy.float32))
        return float(final_score)

    def predict(self, queries: core.Dataset) -> numpy.ndarray:
        """Predict the anomaly scores for a 2d array of queries."""
        scores = []
        for i in range(queries.cardinality):
            logger.info(f"Predicting anomaly score for query {i} ...")
            scores.append(self.predict_single(queries[i]))
        return numpy.asarray(scores, dtype=numpy.float32)


class SingleSpaceChaoda:
    """This class is a single-metric-space version of CHAODA.

    This is intended to be used from the CHAODA class and does almost no input
    validation.

    This class does not vote among scores to form an ensemble.
    """

    def __init__(  # noqa: PLR0913
        self,
        metric_space: core.Space,
        partition_criteria: typing.Sequence[cluster_criteria.ClusterCriterion],
        selectors_scorers: typing.Sequence[
            tuple[
                graph_criteria.GraphCriterion,
                typing.Sequence[graph_scorers.GraphScorer],
            ]
        ],
        normalization_mode: helpers.NormalizationMode,
        use_speed_threshold: bool,
    ) -> None:
        """Creates single-metric-space version of chaoda.

        Args:
            metric_space: The metric space to use.
            partition_criteria: A list of criteria to use for partitioning
                clusters.
            selectors_scorers: list of 2-tuples whose items are
                - a trained meta-ml model for selecting clusters to build
                  graphs.
                - a list of individual algorithms for predicting anomaly scores
                  from that graph.
            normalization_mode: method to use to normalize anomaly scores from
                different scoring algorithms.
            use_speed_threshold: Whether to only run the fast algorithms in the
                ensemble.
        """
        self.__metric_space = metric_space
        self.__partition_criteria = partition_criteria
        self.__selectors = [s for s, _ in selectors_scorers]
        self.__scorers = [s for _, s in selectors_scorers]
        self.__normalization_mode = normalization_mode
        self.__use_speed_threshold = use_speed_threshold

        self.__root: typing.Union[core.Cluster, constants.Unset] = constants.UNSET
        self.__graphs: typing.Union[list[core.Graph], constants.Unset] = constants.UNSET
        self.__cluster_scores_list: typing.Union[
            list[graph_scorers.ClusterScores],
            constants.Unset,
        ] = constants.UNSET
        self.__scores: typing.Union[numpy.ndarray, constants.Unset] = constants.UNSET
        self.__searcher: typing.Union[search.CAKES, constants.Unset] = constants.UNSET

    @property
    def root(self) -> core.Cluster:
        """The root cluster of the tree."""
        if self.__root is constants.UNSET:
            msg = "Please call the `fit` method before using this property."
            raise ValueError(msg)
        return self.__root  # type: ignore[return-value]

    @property
    def scores(self) -> numpy.ndarray:
        """The individual scores for each point in the dataset."""
        if self.__scores is constants.UNSET:
            msg = "Please call the `fit` method before using this property."
            raise ValueError(msg)
        return self.__scores  # type: ignore[return-value]

    @property
    def cluster_scores_list(self) -> list[graph_scorers.ClusterScores]:
        """The list of cluster scores for each graph."""
        if self.__cluster_scores_list is constants.UNSET:
            msg = "Please call the `fit` method before using this property."
            raise ValueError(msg)
        return self.__cluster_scores_list  # type: ignore[return-value]

    @property
    def searcher(self) -> search.CAKES:
        """The CAKES object for this metric space."""
        if self.__searcher is constants.UNSET:
            msg = "Please call the `fit` method before using this property."
            raise ValueError(msg)
        return self.__searcher  # type: ignore[return-value]

    def build(self) -> "SingleSpaceChaoda":
        """Build a Manifold on the data, create optimal Graphs, and predict scores."""
        self.__root = (
            core.Cluster.new_root(self.__metric_space)
            .build()
            .iterative_partition(self.__partition_criteria)
            .normalize_ratios(mode=self.__normalization_mode)
        )
        self.__searcher = search.CAKES.from_root(self.__root)

        self.__graphs = [
            core.Graph(selector(self.__root)).build() for selector in self.__selectors
        ]

        if self.__use_speed_threshold:
            individual_scores = [
                method(g)
                for g, scorers in zip(self.__graphs, self.__scorers)
                for method in scorers
                if method.should_be_fast(g)
            ]
            if len(individual_scores) == 0:
                uniform_cluster_scores = {self.__root: 0.5}
                uniform_scores = 0.5 * numpy.ones(shape=(self.__root.cardinality,))
                individual_scores.append((uniform_cluster_scores, uniform_scores))
        else:
            individual_scores = [
                method(g)
                for g, scorers in zip(self.__graphs, self.__scorers)
                for method in scorers
            ]

        self.__scores = numpy.stack([s for _, s in individual_scores])
        self.__cluster_scores_list = [d for d, _ in individual_scores]

        return self


__all__ = [
    "VotingMode",
    "CHAODA",
    "SingleSpaceChaoda",
]
