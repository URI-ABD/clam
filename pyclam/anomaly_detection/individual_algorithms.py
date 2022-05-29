import abc
import logging
import typing

import numpy

from .. import core
from ..utils import constants
from ..utils import helpers

logger = logging.getLogger(__name__)
logger.setLevel(constants.LOG_LEVEL)

ClusterScores = dict[core.Cluster, float]
DatumScores = dict[int, float]

# TODO: Look at DSD (diffusion-state-distance) as a possible individual-method.


class GraphScorer(abc.ABC):

    def __init__(self):
        self.__cluster_scores: ClusterScores = dict()
        self.__datum_scores: list[float] = list()

    def __call__(self, graph: core.Graph) -> tuple[ClusterScores, numpy.ndarray]:
        """ Use this scorer to generate normalized anomaly scores for the given graph.
        """
        logger.info(f'Running method {self.name} on a graph with {graph.cardinality} clusters.')
        self.__cluster_scores = self.score_graph(graph)
        self.__normalize_scores()
        self.__inherit_scores(graph.manifold.argpoints)
        self.__individual_scores = numpy.asarray(self.__datum_scores, dtype=numpy.float32)
        return self.__cluster_scores, self.__individual_scores

    @property
    def cluster_scores(self) -> ClusterScores:
        return self.__cluster_scores

    @property
    def individual_scores(self) -> numpy.ndarray:
        return self.__individual_scores

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

    @abc.abstractmethod
    def score_graph(self, graph: core.Graph) -> ClusterScores:
        """ The method with which to assign anomaly scores to Clusters in the
         Graph. This should return a dictionary whose keys are Clusters in the
         Graph and whose values are the anomaly scores (before normalization).
        """
        pass

    @staticmethod
    def should_be_fast(graph: core.Graph) -> bool:
        """ Whether this algorithm is expected to run in a reasonably short
         time on the given Graph. This method should make a quick estimate.
        """
        # TODO: Specialize this for each algorithm based on complexity analysis and graph properties.
        return graph.cardinality <= 128

    def __normalize_scores(self):
        if len(self.__cluster_scores) == 0:
            raise ValueError(f'This scoring method needs to be called on a Graph before scores can be normalized.')

        scores = numpy.asarray(list(self.__cluster_scores.values()), dtype=numpy.float32)
        scores = helpers.normalize(scores, mode='gaussian')
        self.__cluster_scores = {cluster: score for cluster, score in zip(self.__cluster_scores.keys(), map(float, scores))}
        return

    def __inherit_scores(self, manifold_indices: list[int]):
        if len(self.__cluster_scores) == 0:
            raise ValueError(f'This scoring method needs to be called on a Graph before scores can be assigned from Clusters to Instances.')

        scores = {index: score for cluster, score in self.__cluster_scores.items() for index in cluster.argpoints}

        self.__datum_scores = [scores[i] for i in manifold_indices]
        return

    @staticmethod
    def score_subsumed(
            subsumer_scores: ClusterScores,
            subsumer_subsumed: dict[core.Cluster, set[core.Cluster]],
    ) -> ClusterScores:
        """ Given scores for subsumer Clusters and a mapping between
         subsumer-subsumed Clusters, assigns the scores from the subsumer
         to each subsumed. If a Cluster is subsumed by multiple Clusters, the
         highest score from among the subsumers is assigned to the subsumed.

         Cluster A is subsumed by Cluster B if the volume of A lies entirely
          inside B.

        Args:
            subsumer_scores: dict of scores of Clusters that are not subsumed by any
             other cluster.
            subsumer_subsumed: dict where the key is a subsumer Cluster and the
             value is a set of subsumed Clusters.

        Side effects:
            - modifies subsumer_scores

        Returns:
            Anomaly scores for each Cluster.
        """
        for subsumer, subsumed_set in subsumer_subsumed.items():
            for subsumed in subsumed_set:
                if subsumed in subsumer_scores:
                    subsumer_scores[subsumed] = max(subsumer_scores[subsumed], subsumer_scores[subsumer])
                else:
                    subsumer_scores[subsumed] = subsumer_scores[subsumer]
        return subsumer_scores


class ClusterCardinality(GraphScorer):

    @property
    def name(self) -> str:
        return 'cluster_cardinality'

    @property
    def short_name(self) -> str:
        return 'cc'

    def score_graph(self, graph: core.Graph) -> ClusterScores:
        return {cluster: cluster.cardinality for cluster in graph.clusters}


class ComponentCardinality(GraphScorer):

    @property
    def name(self) -> str:
        return 'component_cardinality'

    @property
    def short_name(self) -> str:
        return 'sc'

    def score_graph(self, graph: core.Graph) -> ClusterScores:
        return {
            cluster: component.cardinality
            for component in graph.components
            for cluster in component.clusters
        }


class VertexDegree(GraphScorer):

    @property
    def name(self) -> str:
        return 'vertex_degree'

    @property
    def short_name(self) -> str:
        return 'vd'

    def score_graph(self, graph: core.Graph) -> ClusterScores:
        return {cluster: graph.vertex_degree(cluster) for cluster in graph.clusters}


class ParentCardinality(GraphScorer):

    def __init__(self, weight: typing.Callable[[int], float] = None):
        super().__init__()
        self.weight = (lambda d: 1 / (d ** 0.5)) if weight is None else weight

    @property
    def name(self) -> str:
        return 'parent_cardinality'

    @property
    def short_name(self) -> str:
        return 'pc'

    def score_graph(self, graph: core.Graph) -> ClusterScores:

        scores = {cluster: 0 for cluster in graph.clusters}

        for cluster in graph.clusters:
            ancestry = graph.manifold.ancestry(cluster)
            for i in range(1, len(ancestry)):
                scores[cluster] += (self.weight(i) * ancestry[i - 1].cardinality / ancestry[i].cardinality)

        return {cluster: -scores[cluster] for cluster in graph.clusters}


class GraphNeighborhood(GraphScorer):

    def __init__(self, eccentricity_fraction: float = 0.25):
        if not (0 < eccentricity_fraction <= 1):
            raise ValueError(f'eccentricity fraction must be in the (0, 1] range. Got {eccentricity_fraction:.2e} instead.')

        super().__init__()
        self.eccentricity_fraction = eccentricity_fraction

    @property
    def name(self) -> str:
        return 'graph_neighborhood'

    @property
    def short_name(self) -> str:
        return 'gn'

    @staticmethod
    def __neighborhood_sizes(start: core.Cluster, steps: int, pruned_graph: core.Graph) -> list[int]:
        """ Returns the number of clusters in the frontier at each step in a
         breadth-first traversal of the graph.
        """
        visited: set[core.Cluster] = set()
        frontier: set[core.Cluster] = {start}
        frontier_sizes = [1]

        for _ in range(steps):
            if len(frontier) > 0:
                visited.update(frontier)
                frontier = {
                    neighbor for cluster in frontier
                    for neighbor in pruned_graph.neighbors(cluster)
                    if neighbor not in visited
                }
                frontier_sizes.append(len(frontier))
            else:
                break

        return frontier_sizes

    def score_graph(self, graph: core.Graph) -> ClusterScores:

        pruned_graph, subsumer_subsumed = graph.pruned_graph

        step_sizes: dict[core.Cluster, list[int]] = {
            cluster: self.__neighborhood_sizes(
                start=cluster,
                steps=int(pruned_graph.eccentricity(cluster) * self.eccentricity_fraction),
                pruned_graph=pruned_graph,
            )
            for cluster in pruned_graph.clusters
        }
        subsumer_scores = {cluster: -float(sum(sizes)) for cluster, sizes in step_sizes.items()}

        return self.score_subsumed(subsumer_scores, subsumer_subsumed)


class StationaryProbabilities(GraphScorer):

    def __init__(self, steps: int = 16):
        super().__init__()
        self.steps = steps

    @property
    def name(self) -> str:
        return 'stationary_probabilities'

    @property
    def short_name(self) -> str:
        return 'sp'

    def score_graph(self, graph: core.Graph) -> ClusterScores:

        pruned_graph, subsumer_subsumed = graph.pruned_graph
        subsumer_scores = dict()

        for component in pruned_graph.components:

            if component.cardinality > 1:
                clusters, matrix = component.as_matrix
                matrix = matrix / numpy.sum(matrix, axis=1)[:, None]

                for _ in range(self.steps):
                    # TODO: Go until convergence. For now, matrix ^ (2 ^ 16) ought to be enough.
                    matrix = numpy.linalg.matrix_power(matrix, 2)

                subsumer_scores.update({cluster: score for cluster, score in zip(clusters, numpy.sum(matrix, axis=0))})

            else:  # Component with only one Cluster means that the Cluster is anomalous.
                subsumer_scores.update({cluster: 0 for cluster in component.clusters})

        subsumer_scores = {cluster: -score for cluster, score in subsumer_scores.items()}

        return self.score_subsumed(subsumer_scores, subsumer_subsumed)


DEFAULT_SCORERS = [
    ClusterCardinality(),
    ComponentCardinality(),
    VertexDegree(),
    ParentCardinality(),
    GraphNeighborhood(),
    StationaryProbabilities(),
]
