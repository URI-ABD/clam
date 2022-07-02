import typing

import numpy

from . import cluster
from . import space
from .. import utils
from ..utils import constants
from ..utils import helpers

logger = helpers.make_logger(__name__)

AdjacencyDict = dict[cluster.Cluster, set['Edge']]


class Edge:
    """ An `Edge` represents a relationship between two `Cluster`s in a `Graph`.
     Two clusters in a graph have an edge between them if there is any overlap
     between the volumes of the two clusters.

    In CLAM, we do not allow edges from a cluster to itself.
    In CLAM, edges are bidirectional.
    """

    __slots__ = [
        '__left',
        '__right',
        '__distance',
    ]

    def __init__(self, left: cluster.Cluster, right: cluster.Cluster, distance: float):
        """ Create an edge using two clusters and the distance between their
         centers.
        """
        self.__left, self.__right = (left, right) if left < right else (right, left)
        self.__distance = distance

    def __eq__(self, other: 'Edge') -> bool:
        """ Two edges are equal if their clusters are equal.
        """
        return self.clusters == other.clusters

    def __str__(self) -> str:
        return f'{str(self.__left)} -- {str(self.__right)}'

    def __repr__(self) -> str:
        return f'{str(self.__left)} -- {str(self.__right)} -- {self.__distance:.6e}'

    def __hash__(self) -> int:
        return hash(str(self))

    def __contains__(self, c: cluster.Cluster) -> bool:
        return (c == self.__left) or (c == self.__right)

    @property
    def clusters(self) -> tuple[cluster.Cluster, cluster.Cluster]:
        return self.__left, self.__right

    @property
    def left(self) -> cluster.Cluster:
        return self.__left

    @property
    def right(self) -> cluster.Cluster:
        return self.__right

    @property
    def distance(self) -> float:
        return self.__distance

    @property
    def to_self(self) -> bool:
        """ Whether the `left` and `right` clusters are equal.
        """
        return self.left == self.right

    def neighbor(self, c: cluster.Cluster) -> cluster.Cluster:
        """ Returns the neighbor of the given cluster, if the cluster is in 
        the edge. Otherwise raises a ValueError.
        """

        if c == self.__right:
            return self.__left

        elif c == self.__left:
            return self.__right

        else:
            raise ValueError(f'Cluster {c.name} is not in this edge {str(self)}.')


class Graph:
    """ A `Graph` is a collection of `Cluster`s with `Edge`s between clusters
    with overlapping volumes. We can select clusters from a tree to build a
    graph.

    All clusters in a graph must come from the same tree and the same
    `MetricSpace`. If using any of the `SelectionCriteria` from CLAM, this
    property will hold. Otherwise, it is upon the user to ensure.

    In CLAM, a `Graph` has a useful invariant:
        - Every instance in the dataset is in exactly one cluster in the graph.
        - No cluster is an ancestor/descendent of another cluster in the graph.

    If clusters were selected using any of the `SelectionCriteria`, this
    invariant will hold. Otherwise, the user must verify this manually.
    This invariant does not hold for sub-graphs such as connected components.
    """

    __slots__ = [
        '__clusters',
        '__root',
        '__metric_space',
        '__population',
        '__min_depth',
        '__max_depth',
        '__edges',
        '__adjacency_dict',
        '__neighborhood_sizes',
        '__components',
        '__distance_matrix',
    ]

    def __init__(self, clusters: set[cluster.Cluster], edges: typing.Optional[set[Edge]] = None):
        """ Creates a graph from the given clusters. The edges need not be
        provided because we have efficient internal algorithms for finding all
        edges.

        Args:
            clusters: The set of clusters with which to build the graph.
            edges: Optional. The set of edges to use.
        """
        if len(clusters) == 0:
            raise ValueError(f'Cannot create a graph with an empty set of clusters.')

        self.__clusters = clusters

        _c = next(iter(clusters))
        self.__root = _c if _c.depth == 0 else _c.ancestry[0]
        self.__metric_space = _c.metric_space

        self.__population = sum(c.cardinality for c in self.__clusters)
        self.__min_depth = min(c.depth for c in self.__clusters)
        self.__max_depth = max(c.depth for c in self.__clusters)

        self.__edges: typing.Union[set[Edge], utils.Unset] = edges or constants.UNSET
        self.__adjacency_dict: typing.Union[AdjacencyDict, utils.Unset] = constants.UNSET
        self.__neighborhood_sizes: dict[cluster.Cluster, list[int]] = dict()
        self.__components: typing.Union[list['Graph'], utils.Unset] = constants.UNSET
        self.__distance_matrix: typing.Union[numpy.ndarray, utils.Unset] = constants.UNSET

    @property
    def root(self) -> cluster.Cluster:
        return self.__root

    @property
    def clusters(self) -> set[cluster.Cluster]:
        return self.__clusters

    @property
    def edges(self) -> set[Edge]:
        if self.__edges is constants.UNSET:
            raise ValueError(f'Please call `build` before accessing this property.')
        return self.__edges

    @property
    def metric_space(self) -> space.Space:
        return self.__metric_space

    @property
    def vertex_cardinality(self) -> int:
        return len(self.__clusters)

    @property
    def edge_cardinality(self) -> int:
        return len(self.edges)

    @property
    def population(self) -> int:
        """ The total number of instances across all clusters in the graph.
        """
        return self.__population

    @property
    def min_depth(self) -> int:
        """ The minimum depth of any cluster in the graph.
        """
        return self.__min_depth

    @property
    def max_depth(self) -> int:
        """ The maximum depth of any cluster in the graph.
        """
        return self.__max_depth

    @property
    def depth_range(self) -> tuple[int, int]:
        """ The range of depths of the clusters in the graph.
        """
        return self.__min_depth, self.__max_depth

    @property
    def adjacency_dict(self) -> AdjacencyDict:
        """ A dictionary of clusters and their sets of edges.
        """
        if self.__adjacency_dict is constants.UNSET:
            raise ValueError(f'Please call `build` before accessing this property.')
        return self.__adjacency_dict

    @property
    def indices(self) -> list[int]:
        """ Indices of instances in all clusters in the graph.
        """
        return [i for c in self.__clusters for i in c.indices]

    @property
    def diameter(self) -> int:
        """ The maximum eccentricity of any cluster in the graph.
        """
        return max(self.eccentricity(c) for c in self.__clusters)

    @property
    def components(self) -> list['Graph']:
        """ List of connected components, as sub-`Graph`s of this graph. If
        there is more than one component, no component will uphold the graph
        invariant.
        """

        if self.__components is constants.UNSET:

            self.__components = list()
            unvisited = self.__clusters.copy()

            while len(unvisited) > 0:
                start = unvisited.pop()
                visited, _ = self.__traverse(start)
                unvisited = unvisited - visited
                self.__components.append(self.subgraph_with(visited))

        return self.__components

    @property
    def distance_matrix(self) -> tuple[list[cluster.Cluster], numpy.ndarray]:
        """ A list of clusters in the graph and a square matrix of edge lengths.
        """

        if self.__distance_matrix is constants.UNSET:

            clusters = list(self.__clusters)
            indices: dict[cluster.Cluster, int] = {c: i for i, c in enumerate(clusters)}

            matrix = numpy.zeros(shape=(self.vertex_cardinality, self.vertex_cardinality))
            for e in self.__edges:
                i, j = indices[e.left], indices[e.right]
                matrix[i, j] = e.distance
                matrix[j, i] = e.distance

            self.__distance_matrix = matrix.astype(numpy.float32)

        return list(self.__clusters), self.__distance_matrix

    @property
    def adjacency_matrix(self) -> tuple[list[cluster.Cluster], numpy.ndarray]:
        """ A list of clusters in the graph and a binary square matrix of edges.
        """
        clusters, matrix = self.distance_matrix
        matrix = (matrix > 0).astype(numpy.uint8)
        return clusters, matrix

    def build(self) -> 'Graph':
        """ Builds the `edges` and `adjacency_dict` of the graph, sets the
        properties and returns the modified graph.
        """
        logger.info(f'Building graph with {self.vertex_cardinality} clusters in a depth range of {self.depth_range} ...')

        if self.__edges is constants.UNSET:
            self.__edges: set[Edge] = {
                Edge(c, n, d) for c in self.__clusters
                for n, d in c.candidate_neighbors.items()
                if (n != c) and (n in self.__clusters) and (d <= (c.radius + n.radius))
            }

        self.__adjacency_dict: AdjacencyDict = {c: set() for c in self.__clusters}
        for e in self.__edges:
            self.__adjacency_dict[e.left].add(e)
            self.__adjacency_dict[e.right].add(e)

        return self

    def assert_contains(self, c: cluster.Cluster):
        """ Raises a ValueError if `c` is not in the graph.
        """
        if c not in self.__clusters:
            raise AssertionError(f'Cluster {c} is not in this Graph.')
        return

    def jaccard(self, other: 'Graph') -> float:
        """ Compute the Jaccard similarity with another graph.
        """
        self_indices = set(self.indices)
        other_indices = set(other.indices)

        intersection = self_indices.intersection(other_indices)
        union = self_indices.union(other_indices)

        return len(intersection) / len(union)

    def vertex_degree(self, c: cluster.Cluster) -> int:
        self.assert_contains(c)
        return len(self.adjacency_dict[c])

    def edges_of(self, c: cluster.Cluster) -> list[Edge]:
        self.assert_contains(c)
        return list(self.adjacency_dict[c])

    def neighbors_of(self, c: cluster.Cluster) -> list[cluster.Cluster]:
        self.assert_contains(c)
        return [e.neighbor(c) for e in self.adjacency_dict[c]]

    def edge_distances_of(self, c: cluster.Cluster) -> list[float]:
        self.assert_contains(c)
        return [e.distance for e in self.adjacency_dict[c]]

    def __traverse(self, start: cluster.Cluster) -> tuple[set[cluster.Cluster], list[int]]:
        """ Performs a traversal, in arbitrary order, starting at the given
        cluster. The traversal continues until no new clusters can be visited.

        Args:
            start: where the traversal will start.

        Returns:
            A 2-tuple of:
                - set of visited clusters.
                - list of number of clusters in the frontier at each iteration
                  of the traversal.
        """

        visited: set[cluster.Cluster] = set()
        frontier: set[cluster.Cluster] = {start}
        frontier_sizes = list()

        while len(frontier) > 0:
            new_frontier = {
                n for c in frontier
                for n in self.neighbors_of(c)
                if not ((n in visited) or (n in frontier))
            }
            visited.update(frontier)
            frontier = new_frontier
            frontier_sizes.append(len(frontier))

        return visited, frontier_sizes

    def frontier_sizes(self, c: cluster.Cluster) -> list[int]:
        self.assert_contains(c)
        if c not in self.__neighborhood_sizes:
            _, n = self.__traverse(c)
            self.__neighborhood_sizes[c] = n
        return self.__neighborhood_sizes[c]

    def eccentricity(self, c: cluster.Cluster) -> int:
        return len(self.frontier_sizes(c))

    def subgraph_with(self, clusters: set[cluster.Cluster]) -> 'Graph':
        [self.assert_contains(c) for c in clusters]
        edges = {
            e for e in self.__edges
            if (e.left in clusters) and (e.right in clusters)
        }
        return Graph(clusters, edges).build()

    def component_containing(self, c: cluster.Cluster) -> 'Graph':
        self.assert_contains(c)
        for component in self.components:
            if c in component.__clusters:
                return component
        else:
            raise ValueError(f'Cluster is the graph but not in any component. This is a bug.')

    def as_dot_string(self) -> str:
        raise NotImplementedError

    @staticmethod
    def from_dot_string(dot_string) -> 'Graph':
        raise NotImplementedError

    def __add(self, c: cluster.Cluster):
        raise NotImplementedError

    def __remove(self, c: cluster.Cluster):
        raise NotImplementedError

    def replace_clusters(self, removals: set[cluster.Cluster], additions: set[cluster.Cluster]) -> 'Graph':
        raise NotImplementedError


__all__ = [
    'AdjacencyDict',
    'Edge',
    'Graph',
]
