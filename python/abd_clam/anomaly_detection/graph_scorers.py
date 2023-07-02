"""Graph anomaly detection algorithms."""

import abc
import typing

import numpy

from .. import core
from ..utils import constants
from ..utils import helpers

logger = helpers.make_logger(__name__)

ClusterScores = dict[core.Cluster, float]
InstanceScores = dict[int, float]

# TODO: Look at DSD (diffusion-state-distance) as a possible individual-method.
# TODO: Specialize `should_be_fast` for each algorithm based on complexity analysis
# and graph properties.


class GraphScorer(abc.ABC):
    """A base class for all anomaly detection algorithms."""

    def __call__(self, g: core.Graph) -> tuple[ClusterScores, numpy.ndarray]:
        """Use this scorer to generate normalized anomaly scores."""
        logger.info(
            f"Running method {self.name} on a "
            "graph with {g.vertex_cardinality} clusters.",
        )

        cluster_scores = self.score_graph(g)
        if self.normalize_on_clusters:
            cluster_scores = self.__normalize_cluster_scores(cluster_scores)

        instance_scores = self.inherit_scores(cluster_scores)
        if not self.normalize_on_clusters:
            instance_scores = self.__normalize_instance_scores(instance_scores)

        scores_array = self.ordered_scores(g, instance_scores)
        return cluster_scores, scores_array

    @abc.abstractmethod
    def __hash__(self) -> int:
        """A way to uniquely identify each `GraphScorer` object."""
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Full name of the algorithm."""
        pass

    @property
    @abc.abstractmethod
    def short_name(self) -> str:
        """Abbreviated name of the algorithm."""
        pass

    @property
    @abc.abstractmethod
    def normalize_on_clusters(self) -> bool:
        """Whether to normalize scores for clusters or for instances."""
        pass

    @property
    @abc.abstractmethod
    def normalization_mode(self) -> helpers.NormalizationMode:
        """What normalization method to use."""
        pass

    @abc.abstractmethod
    def score_graph(self, g: core.Graph) -> ClusterScores:
        """The main method of the algorithm.

        This should return a dictionary whose keys are `Clusters` in the `Graph`
        and whose values are the pre-normalization anomaly scores.
        """
        pass

    @abc.abstractmethod
    def should_be_fast(self, g: core.Graph) -> bool:
        """Whether this algorithm is expected to run in a reasonably short time.

        This estimate method should made quickly.
        """
        pass

    def __normalize_cluster_scores(
        self,
        scores: ClusterScores,
    ) -> ClusterScores:
        """Normalize scores to be in the [0, 1] range."""
        [keys, scores] = list(zip(*scores.items()))  # type: ignore[assignment]
        scores = helpers.normalize(
            numpy.asarray(scores, dtype=numpy.float32),
            mode=self.normalization_mode,
        )
        return dict(zip(keys, (float(x) for x in scores)))  # type: ignore[arg-type]

    def __normalize_instance_scores(
        self,
        scores: InstanceScores,
    ) -> InstanceScores:
        """Normalize scores to be in the [0, 1] range."""
        [keys, scores] = list(zip(*scores.items()))  # type: ignore[assignment]
        scores = helpers.normalize(
            numpy.asarray(scores, dtype=numpy.float32),
            mode=self.normalization_mode,
        )
        return dict(zip(keys, (float(x) for x in scores)))  # type: ignore[arg-type]

    @staticmethod
    def inherit_scores(scores: ClusterScores) -> InstanceScores:
        """Inherit cluster scores to instance scores."""
        return {i: s for c, s in scores.items() for i in c.indices}

    @staticmethod
    def ordered_scores(g: core.Graph, scores: InstanceScores) -> numpy.ndarray:
        """Return scores in the order of the graph's vertices."""
        [indices, scores_] = list(zip(*sorted(scores.items())))
        if set(indices) != {i for c in g.clusters for i in c.indices}:
            msg = "Graph invariant violated."
            raise ValueError(msg)
        return numpy.asarray(scores_, dtype=numpy.float32)


class ClusterCardinality(GraphScorer):
    """A method that scores clusters by their cardinality.

    The smaller the cluster, the more anomalous it is.
    """

    def __hash__(self) -> int:
        """A way to uniquely identify each `GraphScorer` object by its name."""
        return hash(self.name)

    @property
    def name(self) -> str:
        """Full name of the algorithm."""
        return "cluster_cardinality"

    @property
    def short_name(self) -> str:
        """Abbreviated name of the algorithm."""
        return "cc"

    @property
    def normalize_on_clusters(self) -> bool:
        """Whether to normalize scores for clusters."""
        return True

    @property
    def normalization_mode(self) -> helpers.NormalizationMode:
        """The normalization mode to use."""
        return "gaussian"

    def should_be_fast(self, _: core.Graph) -> bool:
        """This method is expected to be fast for all graphs."""
        return True

    def score_graph(self, g: core.Graph) -> ClusterScores:
        """The main method of the algorithm."""
        return {c: -c.cardinality for c in g.clusters}


class ComponentCardinality(GraphScorer):
    """A method that scores clusters by their component's cardinality.

    The smaller the component, the more anomalous its clusters are.
    """

    def __hash__(self) -> int:
        """A way to uniquely identify each `GraphScorer` object by its name."""
        return hash(self.name)

    @property
    def name(self) -> str:
        """Full name of the algorithm."""
        return "component_cardinality"

    @property
    def short_name(self) -> str:
        """Abbreviated name of the algorithm."""
        return "sc"

    @property
    def normalize_on_clusters(self) -> bool:
        """Whether to normalize scores for clusters."""
        return True

    @property
    def normalization_mode(self) -> helpers.NormalizationMode:
        """The normalization mode to use."""
        return "gaussian"

    def should_be_fast(self, _: core.Graph) -> bool:
        """This method is expected to be fast for all graphs."""
        return True

    def score_graph(self, g: core.Graph) -> ClusterScores:
        """The main method of the algorithm."""
        return {
            c: -component.vertex_cardinality
            for component in g.components
            for c in component.clusters
        }


class VertexDegree(GraphScorer):
    """A method that scores clusters by their vertex degree.

    The smaller the degree, the more anomalous the cluster is.
    """

    def __hash__(self) -> int:
        """A way to uniquely identify each `GraphScorer` object by its name."""
        return hash(self.name)

    @property
    def name(self) -> str:
        """Full name of the algorithm."""
        return "vertex_degree"

    @property
    def short_name(self) -> str:
        """Abbreviated name of the algorithm."""
        return "vd"

    @property
    def normalize_on_clusters(self) -> bool:
        """Whether to normalize scores for clusters."""
        return True

    @property
    def normalization_mode(self) -> helpers.NormalizationMode:
        """The normalization mode to use."""
        return "gaussian"

    def should_be_fast(self, _: core.Graph) -> bool:
        """This method is expected to be fast for all graphs."""
        return True

    def score_graph(self, g: core.Graph) -> ClusterScores:
        """The main method of the algorithm."""
        return {c: -g.vertex_degree(c) for c in g.clusters}


class ParentCardinality(GraphScorer):
    """A method that scores clusters by their parent's cardinality.

    The smaller the cluster relative to its parent, the more anomalous the cluster is.
    """

    def __init__(self, depth_weight: typing.Callable[[int], float]) -> None:
        """Initialize the scorer."""
        self.weight = depth_weight

    def __hash__(self) -> int:
        """A way to uniquely identify each `GraphScorer` object by its name."""
        return hash(self.name)

    @property
    def name(self) -> str:
        """Full name of the algorithm."""
        return "parent_cardinality"

    @property
    def short_name(self) -> str:
        """Abbreviated name of the algorithm."""
        return "pc"

    @property
    def normalize_on_clusters(self) -> bool:
        """Whether to normalize scores for clusters."""
        return True

    @property
    def normalization_mode(self) -> helpers.NormalizationMode:
        """The normalization mode to use."""
        return "gaussian"

    def should_be_fast(self, _: core.Graph) -> bool:
        """This method is expected to be fast for all graphs."""
        return True

    def score_graph(self, g: core.Graph) -> ClusterScores:
        """The main method of the algorithm."""
        scores = {c: 0.0 for c in g.clusters}

        for c in g.clusters:
            ancestry = c.ancestry
            for i in range(1, len(ancestry)):
                scores[c] += (
                    self.weight(i)
                    * ancestry[i - 1].cardinality
                    / ancestry[i].cardinality
                )

        return {c: scores[c] for c in g.clusters}


class GraphNeighborhood(GraphScorer):
    """A method that scores clusters by the number of clusters in their neighborhood.

    The smaller the neighborhood, the more anomalous the cluster is.
    """

    def __init__(self, eccentricity_fraction: float) -> None:
        """Initialize the scorer."""
        if not (0 < eccentricity_fraction <= 1):
            msg = (
                "eccentricity fraction must be in the (0, 1] range. "
                "Got {eccentricity_fraction:.2e} instead."
            )
            raise ValueError(
                msg,
            )

        self.eccentricity_fraction = eccentricity_fraction

    def __hash__(self) -> int:
        """A way to uniquely identify each `GraphScorer` object by its name."""
        return hash(self.name)

    @property
    def name(self) -> str:
        """Full name of the algorithm."""
        return "graph_neighborhood"

    @property
    def short_name(self) -> str:
        """Abbreviated name of the algorithm."""
        return "gn"

    @property
    def normalize_on_clusters(self) -> bool:
        """Whether to normalize scores for clusters."""
        return True

    @property
    def normalization_mode(self) -> helpers.NormalizationMode:
        """The normalization mode to use."""
        return "gaussian"

    def should_be_fast(self, g: core.Graph) -> bool:
        """This method is expected to be fast for small graphs."""
        return g.vertex_cardinality < 256

    def num_steps(self, g: core.Graph, c: core.Cluster) -> int:
        """The number of steps to take away from the cluster."""
        return int(g.eccentricity(c) * self.eccentricity_fraction) + 1

    def score_graph(self, g: core.Graph) -> ClusterScores:
        """The main method of the algorithm."""
        step_sizes: dict[core.Cluster, list[int]] = {
            c: g.frontier_sizes(c)[: self.num_steps(g, c)] for c in g.clusters
        }
        return {c: -float(sum(sizes)) for c, sizes in step_sizes.items()}


class StationaryProbabilities(GraphScorer):
    """A method that scores clusters by their stationary probabilities.

    The stationary probability of a cluster is the probability that a random walk
    starting from a vertex in the cluster will end up in the cluster after a large
    number of steps.

    The smaller the stationary probability, the more anomalous the cluster is.
    """

    def __init__(self, steps: int) -> None:
        """Initialize the scorer."""
        self.steps = steps

    def __hash__(self) -> int:
        """A way to uniquely identify each `GraphScorer` object by its name."""
        return hash(self.name)

    @property
    def name(self) -> str:
        """Full name of the algorithm."""
        return "stationary_probabilities"

    @property
    def short_name(self) -> str:
        """Abbreviated name of the algorithm."""
        return "sp"

    @property
    def normalize_on_clusters(self) -> bool:
        """Whether to normalize scores for clusters."""
        return True

    @property
    def normalization_mode(self) -> helpers.NormalizationMode:
        """The normalization mode to use."""
        return "gaussian"

    def should_be_fast(self, _: core.Graph) -> bool:
        """This method is expected to be fast for small graphs."""
        return True

    def score_graph(self, g: core.Graph) -> ClusterScores:
        """The main method of the algorithm."""
        scores = {}

        for component in g.components:
            if component.vertex_cardinality > 1:
                clusters, matrix = component.distance_matrix

                # TODO: Figure out how to go until convergence.
                sums = numpy.sum(matrix, axis=1)[:, None] + constants.EPSILON
                matrix = matrix / sums
                for _ in range(self.steps):
                    matrix = numpy.linalg.matrix_power(matrix, 2)

                scores.update(
                    dict(zip(clusters, numpy.sum(matrix, axis=0))),
                )

            else:  # Component with only one Cluster means the Cluster is anomalous.
                scores.update({c: 0 for c in component.clusters})

        scores = {c: -s for c, s in scores.items()}

        return scores


__all__ = [
    "ClusterScores",
    "InstanceScores",
    "GraphScorer",
    "ClusterCardinality",
    "ComponentCardinality",
    "VertexDegree",
    "ParentCardinality",
    "GraphNeighborhood",
    "StationaryProbabilities",
]
