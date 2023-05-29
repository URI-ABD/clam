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
# TODO: Specialize `should_be_fast` for each algorithm based on complexity analysis and graph properties.


class GraphScorer(abc.ABC):

    def __call__(self, g: core.Graph) -> tuple[ClusterScores, numpy.ndarray]:
        """ Use this scorer to generate normalized anomaly scores.
        """
        logger.info(f'Running method {self.name} on a graph with {g.vertex_cardinality} clusters.')

        cluster_scores = self.score_graph(g)
        if self.normalize_on_clusters:
            cluster_scores = self.__normalize_scores(cluster_scores)

        instance_scores = self.inherit_scores(cluster_scores)
        if not self.normalize_on_clusters:
            instance_scores = self.__normalize_scores(instance_scores)

        scores_array = self.ordered_scores(g, instance_scores)
        return cluster_scores, scores_array

    @abc.abstractmethod
    def __hash__(self):
        """ A way to uniquely identify each `GraphScorer` object.
        """
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """ Full name of the algorithm.
        """
        pass

    @property
    @abc.abstractmethod
    def short_name(self) -> str:
        """ Abbreviated name of the algorithm.
        """
        pass

    @property
    @abc.abstractmethod
    def normalize_on_clusters(self) -> bool:
        """ Whether to normalize scores for clusters or for instances.
        """
        pass

    @property
    @abc.abstractmethod
    def normalization_mode(self) -> helpers.NormalizationMode:
        """ What normalization method to use.
        """
        pass

    @abc.abstractmethod
    def score_graph(self, g: core.Graph) -> ClusterScores:
        """ The method with which to assign anomaly scores to `Clusters` in the
        `Graph`. This should return a dictionary whose keys are `Clusters` in
        the `Graph` and whose values are the pre-normalization anomaly scores.
        """
        pass

    @abc.abstractmethod
    def should_be_fast(self, g: core.Graph) -> bool:
        """ Whether this algorithm is expected to run in a reasonably short
        time on the given `Graph`. This method should make a quick estimate.
        """
        pass

    def __normalize_scores(
            self,
            scores: typing.Union[ClusterScores, InstanceScores],
    ) -> typing.Union[ClusterScores, InstanceScores]:
        [keys, scores] = list(zip(*scores.items()))
        scores = helpers.normalize(numpy.asarray(scores, dtype=numpy.float32), mode=self.normalization_mode)
        return {c: s for c, s in zip(keys, map(float, scores))}

    @staticmethod
    def inherit_scores(scores: ClusterScores) -> InstanceScores:
        return {i: s for c, s in scores.items() for i in c.indices}

    @staticmethod
    def ordered_scores(g: core.Graph, scores: InstanceScores) -> numpy.ndarray:
        [indices, scores] = list(zip(*sorted(scores.items())))
        assert set(indices) == {i for c in g.clusters for i in c.indices}
        return numpy.asarray(scores, dtype=numpy.float32)


class ClusterCardinality(GraphScorer):

    def __hash__(self):
        return hash(self.name)

    @property
    def name(self) -> str:
        return 'cluster_cardinality'

    @property
    def short_name(self) -> str:
        return 'cc'

    @property
    def normalize_on_clusters(self) -> bool:
        return True

    @property
    def normalization_mode(self) -> helpers.NormalizationMode:
        return 'gaussian'

    def should_be_fast(self, _) -> bool:
        return True

    def score_graph(self, g: core.Graph) -> ClusterScores:
        return {c: -c.cardinality for c in g.clusters}


class ComponentCardinality(GraphScorer):

    def __hash__(self):
        return hash(self.name)

    @property
    def name(self) -> str:
        return 'component_cardinality'

    @property
    def short_name(self) -> str:
        return 'sc'

    @property
    def normalize_on_clusters(self) -> bool:
        return True

    @property
    def normalization_mode(self) -> helpers.NormalizationMode:
        return 'gaussian'

    def should_be_fast(self, _) -> bool:
        return True

    def score_graph(self, g: core.Graph) -> ClusterScores:
        return {
            c: -component.vertex_cardinality
            for component in g.components
            for c in component.clusters
        }


class VertexDegree(GraphScorer):

    def __hash__(self):
        return hash(self.name)

    @property
    def name(self) -> str:
        return 'vertex_degree'

    @property
    def short_name(self) -> str:
        return 'vd'

    @property
    def normalize_on_clusters(self) -> bool:
        return True

    @property
    def normalization_mode(self) -> helpers.NormalizationMode:
        return 'gaussian'

    def should_be_fast(self, _) -> bool:
        return True

    def score_graph(self, g: core.Graph) -> ClusterScores:
        return {c: -g.vertex_degree(c) for c in g.clusters}


class ParentCardinality(GraphScorer):

    def __init__(self, depth_weight: typing.Callable[[int], float]):
        self.weight = depth_weight

    def __hash__(self):
        return hash(self.name)

    @property
    def name(self) -> str:
        return 'parent_cardinality'

    @property
    def short_name(self) -> str:
        return 'pc'

    @property
    def normalize_on_clusters(self) -> bool:
        return True

    @property
    def normalization_mode(self) -> helpers.NormalizationMode:
        return 'gaussian'

    def should_be_fast(self, _) -> bool:
        return True

    def score_graph(self, g: core.Graph) -> ClusterScores:

        scores = {c: 0 for c in g.clusters}

        for c in g.clusters:
            ancestry = c.ancestry
            for i in range(1, len(ancestry)):
                scores[c] += (self.weight(i) * ancestry[i - 1].cardinality / ancestry[i].cardinality)

        return {c: scores[c] for c in g.clusters}


class GraphNeighborhood(GraphScorer):

    def __init__(self, eccentricity_fraction: float):
        if not (0 < eccentricity_fraction <= 1):
            raise ValueError(f'eccentricity fraction must be in the (0, 1] range. Got {eccentricity_fraction:.2e} instead.')

        self.eccentricity_fraction = eccentricity_fraction

    def __hash__(self):
        return hash(self.name)

    @property
    def name(self) -> str:
        return 'graph_neighborhood'

    @property
    def short_name(self) -> str:
        return 'gn'

    @property
    def normalize_on_clusters(self) -> bool:
        return True

    @property
    def normalization_mode(self) -> helpers.NormalizationMode:
        return 'gaussian'

    def should_be_fast(self, g: core.Graph) -> bool:
        return g.vertex_cardinality < 256

    def num_steps(self, g: core.Graph, c: core.Cluster) -> int:
        return int(g.eccentricity(c) * self.eccentricity_fraction) + 1

    def score_graph(self, g: core.Graph) -> ClusterScores:

        step_sizes: dict[core.Cluster, list[int]] = {
            c: g.frontier_sizes(c)[:self.num_steps(g, c)]
            for c in g.clusters
        }
        scores = {c: -float(sum(sizes)) for c, sizes in step_sizes.items()}

        return scores


class StationaryProbabilities(GraphScorer):

    def __init__(self, steps: int):
        self.steps = steps

    def __hash__(self):
        return hash(self.name)

    @property
    def name(self) -> str:
        return 'stationary_probabilities'

    @property
    def short_name(self) -> str:
        return 'sp'

    @property
    def normalize_on_clusters(self) -> bool:
        return True

    @property
    def normalization_mode(self) -> helpers.NormalizationMode:
        return 'gaussian'

    def should_be_fast(self, g: core.Graph) -> bool:
        return True

    def score_graph(self, g: core.Graph) -> ClusterScores:

        scores = dict()

        for component in g.components:

            if component.vertex_cardinality > 1:
                clusters, matrix = component.distance_matrix

                # TODO: Figure out how to go until convergence.
                sums = numpy.sum(matrix, axis=1)[:, None] + constants.EPSILON
                matrix = matrix / sums
                for _ in range(self.steps):
                    matrix = numpy.linalg.matrix_power(matrix, 2)

                scores.update({c: s for c, s in zip(clusters, numpy.sum(matrix, axis=0))})

            else:  # Component with only one Cluster means that the Cluster is anomalous.
                scores.update({c: 0 for c in component.clusters})

        scores = {c: -s for c, s in scores.items()}

        return scores


__all__ = [
    'ClusterScores',
    'InstanceScores',
    'GraphScorer',
    'ClusterCardinality',
    'ComponentCardinality',
    'VertexDegree',
    'ParentCardinality',
    'GraphNeighborhood',
    'StationaryProbabilities',
]
