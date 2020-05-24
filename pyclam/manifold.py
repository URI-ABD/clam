""" Clustered Learning of Approximate Manifolds.
"""
import concurrent.futures
import logging
import pickle
from collections import deque
from operator import itemgetter
from typing import Set, Dict, Iterable, BinaryIO, List, Union, Tuple, IO

import numpy as np
from scipy.spatial.distance import cdist

from pyclam.types import Data, Radius, Vector, Metric, Edge

SUBSAMPLE_LIMIT = 100
BATCH_SIZE = 10_000
LOG_LEVEL = logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s:%(levelname)s:%(name)s:%(module)s.%(funcName)s:%(message)s"
)


class Cluster:
    """ A cluster of points.

    Clusters maintain:
        references to their their children,
        the manifold to which they belong,
        and the indices of the points they are responsible for.

    You can compare clusters, hash them, partition them, perform tree search, prune them, and more.
    Cluster implements methods that create and utilize the underlying tree structure used by Manifold.
    """

    def __init__(self, manifold: 'Manifold', argpoints: Vector, name: str, **kwargs):
        """
        A Cluster needs to know the manifold it belongs to and the indexes of the points it contains.
        The name of a Cluster indicates its position in the tree.

        :param manifold: The manifold to which the cluster belongs.
        :param argpoints: A list of indexes of the points that belong to the cluster.
        :param name: The name of the cluster indicating its position in the tree.
        """
        logging.debug(f"Cluster(name={name}, argpoints={argpoints})")
        self.manifold: 'Manifold' = manifold
        self.argpoints: Vector = argpoints
        self.name: str = name
        self.children: Set['Cluster'] = set()

        # Reference to the distance function for easier usage
        self.distance = self.manifold.distance

        # list of candidate neighbors.
        # This helps with a highly efficient neighbor search for clusters in the tree.
        self.candidates: Union[List[Cluster], None] = None

        self.__dict__['_optimal'] = False  # whether cluster depth is optimal
        self.__dict__.update(**kwargs)

        # This is used while reading clusters from file during Cluster.from_json().
        if not argpoints and self.children:
            self.argpoints = [p for child in self.children for p in child.argpoints]
        elif not argpoints:
            raise ValueError(f'Cluster {name} needs argpoints')
        return

    def __eq__(self, other: 'Cluster') -> bool:
        """ Two clusters are identical if they have the same name and the same set of points. """
        return all((
            self.name == other.name,
            set(self.argpoints) == set(other.argpoints),
        ))

    def __bool__(self) -> bool:
        return self.cardinality > 0

    def __hash__(self):
        """ Be careful to use this only with other clusters in the same tree. """
        return hash(self.name)

    def __str__(self) -> str:
        return self.name or 'root'

    def __repr__(self) -> str:
        return '-'.join([self.name, ', '.join(map(str, self.argpoints))])

    def __iter__(self) -> Vector:
        # Iterates in batches, instead of by element.
        for i in range(0, self.cardinality, BATCH_SIZE):
            yield self.argpoints[i:i + BATCH_SIZE]

    def __contains__(self, point: Data) -> bool:
        """ Check weather the given point could be inside this cluster. """
        return self.overlaps(point=point, radius=0.)

    @property
    def cardinality(self) -> int:
        return len(self.argpoints)

    @property
    def metric(self) -> str:
        """ The metric used in the manifold. """
        return self.manifold.metric

    @property
    def depth(self) -> int:
        """ The depth in the tree at which the cluster exists. """
        return len(self.name)

    def distance_from(self, x1: Union[List[int], Data]) -> np.ndarray:
        """ Helper to ease calculation of distance from the cluster center. """
        return self.distance([self.argmedoid], x1)[0]

    @property
    def points(self) -> Data:
        """ An iterator, in batches, over the points in the Clusters. """
        for i in range(0, self.cardinality, BATCH_SIZE):
            yield self.manifold.data[self.argpoints[i:i + BATCH_SIZE]]

    @property
    def samples(self) -> Data:
        """ Returns the samples from the cluster. Samples are used in computing approximate centers and poles.
        """
        return self.manifold.data[self.argsamples]

    @property
    def argsamples(self) -> Vector:
        """ Indices of samples chosen for finding poles.

        Ensures that there are at least 2 different points in samples,
        otherwise returns a single sample that represents the entire cluster.
        i.e., if len(argsamples) == 1, the cluster contains only duplicates.
        """
        if '_argsamples' not in self.__dict__:
            logging.debug(f"building cache for {self}")
            if self.cardinality <= SUBSAMPLE_LIMIT:
                n = len(self.argpoints)
                indices = self.argpoints
            else:
                n = int(np.sqrt(self.cardinality))
                indices = [int(i) for i in np.random.choice(self.argpoints, n, replace=False)]

            # Handle Duplicates.
            if self.distance(indices, indices).max(initial=0.) == 0.:
                indices = np.unique(self.manifold.data[self.argpoints], return_index=True, axis=0)[1]
                indices = [self.argpoints[i] for i in indices][:n]

            # Cache it.
            self.__dict__['_argsamples'] = indices
        return self.__dict__['_argsamples']

    @property
    def nsamples(self) -> int:
        """ The number of samples for the cluster. """
        return len(self.argsamples)

    @property
    def centroid(self) -> Data:
        """ The Geometric Mean of the cluster. """
        return np.average(self.samples, axis=0)

    @property
    def medoid(self) -> Data:
        """ The Geometric Median of the cluster. """
        return self.manifold.data[self.argmedoid]

    @property
    def argmedoid(self) -> int:
        """ The index used to retrieve the medoid. """
        if '_argmedoid' not in self.__dict__:
            logging.debug(f"building cache for {self}")
            _argmedoid = np.argmin(self.distance(self.argsamples, self.argsamples).sum(axis=1))
            self.__dict__['_argmedoid'] = self.argsamples[int(_argmedoid)]
        return self.__dict__['_argmedoid']

    @property
    def radius(self) -> Radius:
        """ The radius of the cluster.

        Computed as distance from medoid to the farthest point in the cluster.
        """
        if '_min_radius' in self.__dict__:
            logging.debug(f'taking min_radius from {self}')
            return self.__dict__['_min_radius']
        elif '_radius' not in self.__dict__:
            logging.debug(f'building cache for {self}')
            _ = self.argradius
        return self.__dict__['_radius']

    @property
    def argradius(self) -> int:
        """ The index of the point which is farthest from the medoid. """
        if ('_argradius' not in self.__dict__) or ('_radius' not in self.__dict__):
            logging.debug(f'building cache for {self}')

            def argmax_max(b):
                distances = self.distance_from(b)
                argmax = int(np.argmax(distances))
                return b[argmax], distances[argmax]

            argradii_radii = [argmax_max(batch) for batch in iter(self)]
            _argradius, _radius = max(argradii_radii, key=itemgetter(1))
            self.__dict__['_argradius'], self.__dict__['_radius'] = int(_argradius), float(_radius)
        return self.__dict__['_argradius']

    @property
    def local_fractal_dimension(self) -> float:
        """ The local fractal dimension of the cluster. """
        if '_local_fractal_dimension' not in self.__dict__:
            logging.debug(f'building cache for {self}')
            if self.nsamples == 1:
                return 0.
            count = [d <= (self.radius / 2)
                     for batch in iter(self)
                     for d in self.distance_from(batch)]
            count = np.sum(count)
            self.__dict__['_local_fractal_dimension'] = count if count == 0. else np.log2(len(self.argpoints) / count)
        return self.__dict__['_local_fractal_dimension']

    @property
    def optimal(self) -> bool:
        return self.__dict__['_optimal']

    def clear_cache(self) -> None:
        """ Clears the cache for the cluster. """
        logging.debug(f'clearing cache for {self}')
        for prop in ['_argsamples', '_argmedoid', '_argradius', '_radius', '_local_fractal_dimension', '_optimal']:
            try:
                del self.__dict__[prop]
            except KeyError:
                pass

    def tree_search(self, point: Data, radius: Radius, depth: int) -> Dict['Cluster', Radius]:
        """ Searches down the tree for clusters that overlap point with radius at depth. """
        # TODO: Rethink with optimal depths
        logging.debug(f'tree_search(point={point}, radius={radius}, depth={depth}')
        if depth == -1:
            depth = len(self.manifold.graphs)
        if depth < self.depth:
            raise ValueError('depth must not be less than cluster.depth')

        results: Dict['Cluster', Radius] = dict()
        if self.depth == depth:
            results = {self: self.distance_from(np.asarray([point]))[0]}
        elif self.overlaps(point, radius):
            results = self._tree_search(point, radius, depth)

        return results

    def _tree_search(self, point: Data, radius: Radius, depth: int) -> Dict['Cluster', Radius]:
        # TODO: rewrite
        distance = self.distance_from(np.asarray([point]))[0]
        assert distance <= radius + self.radius, f'_tree_search was started with no overlap.'
        assert self.depth < depth, f'_tree_search needs to have depth ({depth}) > self.depth ({self.depth}). '

        # results and candidates ONLY contain clusters that have overlap with point
        results: Dict['Cluster', Radius] = dict()
        candidates: Dict['Cluster', Radius] = {self: distance}
        for d_ in range(self.depth, depth):
            # if cluster was not partitioned any further, add it to results.
            results.update({c: d for c, d in candidates.items() if len(c.children) < 1})

            # filter out only those candidates that were partitioned.
            candidates = {c: d for c, d in candidates.items() if len(c.children) > 0}

            # proceed down th tree
            children: List[Cluster] = [c for candidate in candidates.keys() for c in candidate.children]
            if len(children) == 0:
                break

            # filter out clusters that are too far away to possibly contain any hits.
            argcenters = [c.argmedoid for c in children]
            distances = self.distance(np.asarray([point]), argcenters)[0]
            radii = [radius + c.radius for c in children]
            candidates = {c: d for c, d, r in zip(children, distances, radii) if d <= r}
            if len(candidates) == 0:
                break

        assert all((depth >= r.depth for r in results))
        assert all((depth == c.depth for c in candidates))

        # put all potential clusters in one dictionary.
        results.update(candidates)
        # results = {c: d for c, d in results.items() if c.depth == depth}
        return results

    def partition(self, *criterion) -> Iterable['Cluster']:
        # TODO: rewrite
        """ Partitions the cluster into 1 or 2 children.

        2 children are produced if the cluster can be split, otherwise 1 child is produced.
        """
        if not all((
                True if self.depth == 0 else self.name[-1] != '0',
                len(self.argpoints) > 1,
                len(self.argsamples) > 1,
                *(c(self) for c in criterion),
        )):  # Cluster not partitionable
            logging.debug(f'{self} did not partition.')
            self.children = {
                Cluster(
                    self.manifold,
                    self.argpoints,
                    self.name + '0',
                    _argsamples=self.argsamples,
                    _argmedoid=self.argmedoid,
                    _argradius=self.argradius,
                    _radius=self.radius,
                    _local_fractal_dimension=self.local_fractal_dimension,
                )
            }
            return self.children

        # Find the farthest point from argradius
        farthest = self.argsamples[int(np.argmax(self.distance([self.argradius], self.argsamples)[0]))]

        # get lists of argpoints for each child
        p1_idx, p2_idx = list(), list()
        [(p1_idx if p1 < p2 else p2_idx).append(i)
         for batch in iter(self)
         for i, p1, p2 in zip(batch, *self.distance([self.argradius, farthest], batch))]

        # Ensure left child is that one that contains the fewer points.
        p1_idx, p2_idx = (p1_idx, p2_idx) if len(p1_idx) < len(p2_idx) else (p2_idx, p1_idx)
        self.children = {
            Cluster(self.manifold, p1_idx, self.name + '1'),
            Cluster(self.manifold, p2_idx, self.name + '2'),
        } if p1_idx else {
            Cluster(self.manifold, p2_idx, self.name + '0'),
        }
        logging.debug(f'{self} was partitioned.')

        return self.children

    def overlaps(self, point: Data, radius: Radius) -> bool:
        """ Checks if point is within radius + self.radius of cluster. """
        return self.distance_from(np.asarray([point]))[0] <= (self.radius + radius)

    def mark(self, max_lfd: float, min_lfd: float, active: bool = False):
        """ Mark optimal Clusters via a modified depth-first traversal of the tree. """
        if active is False:
            if self.local_fractal_dimension > max_lfd:  # Mark branch as active if above given threshold
                active = True
        elif self.local_fractal_dimension < min_lfd:
            self.__dict__['_optimal'] = True  # Active branches that fall under given threshold is marked optimal
            return  # only one cluster per branch of the tree is marked optimal

        if len(self.children) > 1:  # If there are multiple children, recurse on all children.
            [child.mark(max_lfd, min_lfd, active) for child in self.children]
        else:
            self.__dict__['_optimal'] = True  # The first childless cluster in a branch is optimal.
        return

    def json(self):
        """ This is used for writing the manifold to disk. """
        data = {
            'name': self.name,
            'argpoints': None,  # Do not save argpoints until at leaves.
            'children': [],
            '_radius': self.radius,
            '_argradius': self.argradius,
            '_argsamples': self.argsamples,
            '_argmedoid': self.argmedoid,
            '_local_fractal_dimension': self.local_fractal_dimension,
            '_candidates': None if self.candidates is None else [c.name for c in self.candidates],
            '_optimal': self.optimal,
        }
        if self.children:
            data['children'] = [c.json() for c in self.children]
        else:
            data['argpoints'] = self.argpoints
        return data

    @staticmethod
    def from_json(manifold, data):
        children = set([Cluster.from_json(manifold, c) for c in data.pop('children', [])])
        return Cluster(manifold, children=children, **data)


class Graph:
    """
    Nodes in the Graph are Clusters.
    Two clusters have an edge if they have overlapping volumes.
    """
    # TODO: Consider writing dump/load methods for Graph.

    def __init__(self, *clusters):
        logging.debug(f'Graph(clusters={[str(c) for c in clusters]})')
        assert all(isinstance(c, Cluster) for c in clusters)

        # self.clusters is a dictionary of the clusters in the graph and the list of edges from that cluster.
        # An Edge is a named tuple of Neighbor, Distance, and Transition Probability.
        # Neighbor is is the neighboring cluster.
        # Distance is the distance to that neighbor.
        # Transition Probability is the probability that the edge gets picked during a random walk.
        self.clusters: Dict[Cluster, Set[Edge]] = {c: None for c in clusters}
        return

    def __eq__(self, other: 'Graph') -> bool:  # TODO: Cover, Consider comparing edges as well.
        """ Two graphs are identical if they are composed of the same clusters. """
        return set(self.clusters.keys()) == set(other.clusters.keys())

    def __bool__(self) -> bool:  # TODO: Cover
        return self.cardinality > 0

    def __iter__(self) -> Iterable[Cluster]:
        """ An iterator over the clusters in the graph. """
        yield from self.clusters.keys()

    def __str__(self) -> str:
        if '_str' not in self.__dict__:  # Cashing value because sort can be expensive on many clusters.
            self.__dict__['_str'] = ', '.join(sorted([str(c) for c in self.clusters.keys()]))
        return self.__dict__['_str']

    def __repr__(self) -> str:
        if '_repr' not in self.__dict__:  # Cashing value because sort can be expensive on many clusters.
            self.__dict__['_repr'] = '\n'.join(sorted([repr(c) for c in self.clusters.keys()]))
        return self.__dict__['_repr']

    def __hash__(self):
        return hash(str(self))

    def __contains__(self, cluster: 'Cluster') -> bool:
        return cluster in self.clusters.keys()

    @property
    def cardinality(self) -> int:
        return len(self.clusters.keys())

    @property
    def population(self) -> int:
        return sum((c.cardinality for c in self.clusters))

    @property
    def manifold(self) -> 'Manifold':
        return next(iter(self.clusters.keys())).manifold

    @property
    def metric(self) -> Metric:
        return next(iter(self.clusters.keys())).metric

    def _find_neighbors(self, cluster: Cluster):
        # Dict of candidate neighbors and distances to neighbors.
        candidates: Dict[Cluster, float] = dict()
        radius: float = cluster.manifold.root.radius

        ancestry: List[Cluster] = self.manifold.ancestry(cluster)
        for depth in range(cluster.depth + 1):
            if ancestry[depth].optimal:
                self.clusters[cluster] = {Edge(c, d, 0.) for c, d in candidates.items()
                                          if (d <= ancestry[depth].radius + c.radius) and c.optimal}
            else:
                if ancestry[depth + 1].radius > 0:
                    radius = ancestry[depth + 1].radius

                # This ensures that candidates are calculated once per cluster
                if ancestry[depth + 1].candidates is None:
                    # Keep optimal clusters as candidate neighbors
                    candidates = {c: 0. for c in ancestry[depth].candidates if c.optimal}

                    # Get all children of optimal clusters at the same depth.
                    candidates.update({child: 0. for c in ancestry[depth].candidates
                                       for child in c.children if c.optimal and c.depth == depth})

                    # Get children of non-optimal clusters.
                    candidates.update({child: 0. for c in ancestry[depth].candidates
                                       for child in c.children if not c.optimal})

                    distances = ancestry[depth + 1].distance_from([c.argmedoid for c in candidates])
                    candidates = {c: d for c, d in zip(candidates.keys(), distances) if d <= c.radius + radius * 4}

                    ancestry[depth + 1].candidates = list(candidates.keys())
        return

    def build_edges(self) -> None:
        """ Calculates edges for the graph. """
        logging.info(f'building edges for {len(self.clusters.keys())} clusters')
        self.manifold.root.candidates = [self.manifold.root]

        [self._find_neighbors(c) for c in self.clusters]  # build edges

        for cluster in self.clusters:  # handshake between all neighbors
            for (neighbor, distance, transition_probability) in self.clusters[cluster]:
                self.clusters[neighbor].add(Edge(cluster, distance, transition_probability))

        for cluster in self.clusters:
            if (cluster, 0., 0.) in self.clusters[cluster]:  # Remove edges to self
                self.clusters[cluster].remove(Edge(cluster, 0., 0.))

            if len(self.clusters[cluster]) > 0:  # Compute transition probabilities, only after handshakes.
                _sum = sum([1 / edge.distance for edge in self.clusters[cluster]])
                self.clusters[cluster] = {Edge(edge.neighbor, edge.distance, 1 / (edge.distance * _sum)) for edge in self.clusters[cluster]}

                _sum = sum([edge.transition_probability for edge in self.clusters[cluster]])
                assert abs(_sum - 1.) <= 1e-6, f'transition probabilities did not sum to 1 for cluster {cluster.name}. Got {_sum:.8f} instead.'
        return

    @property
    def edges(self) -> Set[Edge]:
        # TODO: Change return type to indicate source cluster for each edge.
        """ Returns all edges within the graph. """
        if '_edges' not in self.__dict__:
            logging.debug(f'building _edges cache for {self}')
            if any((edges is None for edges in self.clusters.values())):
                self.build_edges()

            edges: Set[Edge] = set()
            [edges.update(e) for e in self.clusters.values()]
            self.__dict__['_edges'] = edges

        return self.__dict__['_edges']

    @property
    def subgraphs(self) -> Set['Graph']:
        """ Returns all subgraphs within the graph. """
        if '_subgraphs' not in self.__dict__:
            self.__dict__['_subgraphs'] = set()
            if any((edges is None for edges in self.clusters.values())):
                self.build_edges()

            unvisited = {c for c in self.clusters}
            while unvisited:
                component = self.traverse(unvisited.pop())
                unvisited -= component
                self.__dict__['_subgraphs'].add(Graph(*component))

        return self.__dict__['_subgraphs']

    def subgraph(self, cluster: 'Cluster') -> 'Graph':  # TODO: Cover
        """ Returns the subgraph to which the cluster belongs. """
        for subgraph in self.subgraphs:
            if cluster in subgraph.clusters:
                return subgraph
        else:
            raise ValueError(f'cluster {cluster.name} not found in and subgraph.')

    def clear_cache(self) -> None:
        """ Clears the cache of the graph. """
        for prop in ['_edges', '_str', '_repr', '_subgraphs']:
            logging.debug(str(self.clusters))
            try:
                del self.__dict__[prop]
            except KeyError:
                pass
        # Clear all cached edges.
        self.clusters = {c: None for c in self.clusters.keys()}
        return

    def neighbors(self, cluster: Cluster) -> List[Cluster]:
        """ return all neighbors of a given cluster. """
        if self.clusters[cluster] is None:
            self.build_edges()
        return [edge.neighbor for edge in self.clusters[cluster]]

    def distances(self, cluster: Cluster) -> List[float]:
        """ return distances to each neighbor of a given cluster. """
        if self.clusters[cluster] is None:
            self.build_edges()
        return [edge.distance for edge in self.clusters[cluster]]

    def transition_probabilities(self, cluster: Cluster) -> List[float]:
        """ return transition probabilities to each neighbor of a given cluster. """
        if self.clusters[cluster] is None:
            self.build_edges()
        return [edge.transition_probability for edge in self.clusters[cluster]]

    def random_walks(
            self,
            clusters: Union[str, List[str], Cluster, List[Cluster]],
            steps: int
    ) -> Dict[Cluster, int]:
        """ Performs random walks, counting visitations of each cluster.

        :param clusters: Clusters at which to start the random walks.
        :param steps: number of steps to take per walk.
        :returns a dictionary of cluster to visit count.
        """
        if self.cardinality < 2:
            return {c: 1 for c in self.clusters}  # TODO: Cover

        if type(clusters) in {Cluster, str}:
            clusters = [clusters]  # TODO: Cover
        if type(clusters) is list and type(clusters[0]) is str:
            clusters = [self.manifold.select(cluster) for cluster in clusters]  # TODO: Cover

        if any((edges is None for edges in self.clusters.values())):
            self.build_edges()

        counts = {c: 0 for c in self.clusters}
        counts.update({c: 1 for c in clusters})

        # initialize walk locations.
        walks = [cluster for cluster in clusters if len(self.clusters[cluster]) > 0]
        for _ in range(steps):
            # update walk locations
            walks = [np.random.choice(a=self.neighbors(cluster), p=self.transition_probabilities(cluster)) for cluster in walks]
            for c in walks:  # increment visit count for each location
                counts[c] += 1
        return counts

    def traverse(self, start: Cluster) -> Set[Cluster]:
        """ Graph traversal starting at start. """
        logging.debug(f'starting traversal from {start}')
        visited: Set[Cluster] = set()
        frontier: Set[Cluster] = {start}
        while frontier:
            visited.update(frontier)
            frontier = {neighbor for cluster in frontier for neighbor in (set(self.neighbors(cluster)) - visited)}
        return visited

    def bft(self, start: Cluster) -> Set[Cluster]:
        """ Breadth-First Traversal starting at start. """
        logging.debug(f'starting breadth-first-traversal from {start}')
        visited = set()
        queue = deque([start])
        while queue:
            cluster = queue.popleft()
            if cluster not in visited:
                visited.add(cluster)
                [queue.append(neighbor) for neighbor in self.neighbors(cluster)]
        return visited

    def dft(self, start: Cluster) -> Set[Cluster]:
        """ Depth-First Traversal starting at start. """
        logging.debug(f'starting depth-first-traversal from {start}')
        visited = set()
        stack: List[Cluster] = [start]
        while stack:
            cluster = stack.pop()
            if cluster not in visited:
                visited.add(cluster)
                stack.extend(self.neighbors(cluster))
        return visited


class Manifold:
    """
    The Manifold's main job is to organize the underlying Clusters and Graphs.
    It does this by providing the ability to reset the build the Cluster-tree, the Graph-stack, and the optimal Graph.
    With the tree and the graphs, Manifold provides utilities for:
        rho-nearest neighbors search,
        k-nearest neighbors search,
    """

    def __init__(self, data: Data, metric: Metric, argpoints: Union[Vector, float] = None, **kwargs):
        """ A Manifold needs the data from which to learn the manifold, and a distance function to use while doing so.

        :param data: The data to learn. This should be a numpy.ndarray or a numpy.memmap.
        :param metric: The distance function to use for the data.
                       Any distance function allowed by scipy.spatial.distance is allowed here.
        :param argpoints: Optional. List of indexes or portion of data to which to restrict Manifold.
        """
        logging.debug(f'Manifold(data={data.shape}, metric={metric}, argpoints={argpoints})')
        self.data: Data = data
        self.metric: Metric = metric

        if argpoints is None:
            self.argpoints = list(range(self.data.shape[0]))
        elif type(argpoints) is list:
            self.argpoints = list(map(int, argpoints))
        elif type(argpoints) is float:
            self.argpoints = np.random.choice(
                self.data.shape[0],
                int(self.data.shape[0] * argpoints),
                replace=False
            )
            self.argpoints = list(map(int, self.argpoints))
        else:
            raise ValueError(f"Invalid argument to argpoints. {argpoints}")

        self.root: Cluster = Cluster(self, self.argpoints, '')
        self.graphs: List[Graph] = [Graph(self.root)]
        self.optimal_graph: Graph = Graph(self.root)

        self.__dict__.update(**kwargs)
        return

    def __eq__(self, other: 'Manifold') -> bool:
        """ Two manifolds are identical if they have the same metric and the same leaf-clusters. """
        return self.metric == other.metric and set(self.graphs[-1]) == set(other.graphs[-1])

    def __iter__(self) -> Iterable[Graph]:
        yield from self.graphs

    def __getitem__(self, depth: int) -> Graph:
        return self.graphs[depth]

    def __str__(self) -> str:
        return f'{self.metric}-{", ".join((str(p) for p in self.argpoints))}'

    def __repr__(self) -> str:
        return f'{str(self)}\n\n' + '\n\n'.join((repr(graph) for graph in self.graphs))

    @property
    def depth(self) -> int:
        return len(self.graphs) - 1

    def distance(self, x1: Union[List[int], Data], x2: Union[List[int], Data]) -> np.ndarray:
        """ Calculates the pairwise distances between all points in x1 and x2.

        This DOES NOT do any batching.

        The metric given to Manifold should have the following properties:
            * dist(p1, p2) = 0 if and only if p1 = p2.
            * dist(p1, p2) = dist(p2, p1)

        :param x1: a list of indices, or a 2D matrix of data points
        :param x2: a list of indices, or a 2D matrix of data points
        :return: matrix of pairwise distances.
        """
        x1, x2 = np.asarray(x1), np.asarray(x2)
        # Fetch data if given indices.
        if len(x1.shape) < 2:
            x1 = self.data[x1 if x1.ndim == 1 else np.expand_dims(x1, 0)]
        if len(x2.shape) < 2:
            x2 = self.data[x2 if x2.ndim == 1 else np.expand_dims(x2, 0)]

        return cdist(x1, x2, metric=self.metric)

    def lfd_range(self, percentiles: Tuple[float, float] = (90, 10)) -> Tuple[float, float]:
        """ Computes the lfd range used for marking optimal clusters. """
        lfd_range = [], []
        for graph in self.graphs:
            clusters: List[Cluster] = [cluster for cluster in graph if cluster.cardinality > 2]
            if len(clusters) > 0:
                lfds = np.percentile(
                    a=[c.local_fractal_dimension for c in clusters],
                    q=percentiles,
                )
                lfd_range[0].append(lfds[0]), lfd_range[1].append(lfds[1])
        return float(np.median(lfd_range[0])), float(np.median(lfd_range[1]))

    def build(self, *criterion) -> 'Manifold':
        """ Rebuilds the Cluster-tree and the Graph-stack. """
        self.graphs = [Graph(self.root)]
        self.build_tree(*criterion)

        max_lfd, min_lfd = self.lfd_range(percentiles=(90, 10))
        self.root.mark(max_lfd, min_lfd)

        self.build_graph()
        return self

    def build_tree(self, *criterion) -> 'Manifold':
        """ Builds the Cluster-tree. """
        while True:
            logging.info(f'depth: {self.depth}, {self.graphs[-1].cardinality} clusters')
            clusters = self._partition_threaded(criterion)
            if self.graphs[-1].cardinality < len(clusters):
                self.graphs.append(Graph(*clusters))
            else:
                # TODO: Figure out how to avoid this extra partition
                [c.children.clear() for c in self.graphs[-1]]
                break
        return self

    def build_graph(self):
        """ Builds the graph at the optimal depth. """
        clusters: List[Cluster] = []
        [clusters.extend([c for c in graph if c.optimal]) for graph in self.graphs]

        logging.info(f'depths: ({min([c.depth for c in clusters])}, {max([c.depth for c in clusters])}), clusters: {len(clusters)}')
        self.optimal_graph = Graph(*clusters)
        self.optimal_graph.build_edges()
        return

    def _partition_single(self, criterion):
        return [child for cluster in self.graphs[-1] for child in cluster.partition(*criterion)]

    def _partition_threaded(self, criterion):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_cluster = [executor.submit(c.partition, *criterion) for c in self.graphs[-1]]
            clusters = []
            [clusters.extend(v.result()) for v in concurrent.futures.as_completed(future_to_cluster)]

        return clusters

    def ancestry(self, cluster: Union[str, Cluster]) -> List[Cluster]:
        """ Returns the sequence of clusters that needs to be traversed to reach the requested cluster.

        :param cluster: A cluster or the name of a cluster.
        :return: The lineage of the cluster starting at the root.
        """
        if type(cluster) is Cluster:
            cluster = cluster.name

        if len(cluster) > self.depth:
            raise ValueError(f'depth of requested cluster must not be greater than depth of cluster-tree. '
                             f'Got {cluster}, max-depth: {self.depth}')

        lineage: List[Cluster] = [self.root]
        for depth in range(len(cluster) + 1):
            for child in lineage[-1].children:
                if child.name == cluster[:depth]:
                    lineage.append(child)
                    break
        assert cluster == lineage[-1].name, f'wanted {cluster} but got {lineage[-1].name}.'
        return lineage

    def select(self, name: str) -> Cluster:
        """ Returns the cluster with the given name. """
        return self.ancestry(name)[-1]

    def find_points(self, point: Data, radius: Radius) -> List[Tuple[int, Radius]]:
        """ Returns all indices of points that are within radius of point. """
        candidates: List[int] = [p for c in self.find_clusters(point, radius, len(self.graphs))
                                 for p in c.argpoints]
        results: Dict[int, Radius] = dict()
        point = np.expand_dims(point, axis=0)
        for i in range(0, len(candidates), BATCH_SIZE):
            batch = candidates[i:i + BATCH_SIZE]
            distances = self.distance(point, batch)[0]
            results.update({p: d for p, d in zip(batch, distances) if d <= radius})
        return sorted([(p, d) for p, d in results.items()], key=itemgetter(1))

    def find_clusters(self, point: Data, radius: Radius, depth: int) -> Dict['Cluster', Radius]:
        """ Returns all clusters that contain points within radius of point at depth. """
        return {r: d for c in self.graphs[0] for r, d in c.tree_search(point, radius, depth).items()}

    def find_knn(self, point: Data, k: int) -> List[Tuple[int, Radius]]:
        """ Finds and returns the k-nearest neighbors of point. """
        radius: Radius = np.float64(np.mean([c.radius for c in self.graphs[-1]]))
        radius = np.float64(max(radius, 1e-16))
        results = self.find_points(point, radius)
        while len(results) < k:
            radius *= 2
            results = self.find_points(point, radius)

        return sorted(results, key=itemgetter(1))[:k]

    def dump(self, fp: Union[BinaryIO, IO[bytes]]) -> None:
        pickle.dump({
            'metric': self.metric,
            'root': self.root.json(),
        }, fp, protocol=pickle.HIGHEST_PROTOCOL)
        return

    @staticmethod
    def load(fp: Union[BinaryIO, IO[bytes]], data: Data) -> 'Manifold':
        d = pickle.load(fp)
        manifold = Manifold(data, metric=d['metric'])

        manifold.root = Cluster.from_json(manifold, d['root'])
        manifold.graphs = [Graph(manifold.root)]
        while True:
            for cluster in manifold.graphs[-1]:
                if cluster.__dict__['_candidates'] is None:
                    cluster.candidates = None
                else:
                    cluster.candidates = [manifold.select(c) for c in cluster.__dict__['_candidates']]

            graph = [child for cluster in manifold.graphs[-1] for child in cluster.children]
            if graph:
                manifold.graphs.append(Graph(*[c for c in graph]))
            else:
                break

        manifold.optimal_graph = Graph(*[cluster for graph in manifold.graphs for cluster in graph if cluster.optimal])
        manifold.optimal_graph.build_edges()

        return manifold
