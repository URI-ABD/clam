""" Clustered Learning of Approximate Manifolds.
"""
import concurrent.futures
import logging
import pickle
from collections import deque
from operator import itemgetter
from typing import Set, Dict, Iterable, BinaryIO, List, Union, Tuple, IO, Any

import numpy as np
from scipy.spatial.distance import cdist

from pyclam.types import Data, Radius, Vector, Metric, Edge, CacheEdge

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
        self.children: Union[None, List['Cluster']] = None

        # Reference to the distance function for easier usage
        self.distance = self.manifold.distance

        # list of candidate neighbors.
        # This helps with a highly efficient neighbor search for clusters in the tree.
        # Candidates have the following properties:
        #     candidate.depth <= self.depth
        #     self.distance_from([candidate.argmedoid]) <= candidate.radius + self.radius * 4
        self.candidates: Union[Dict['Cluster', float], None] = None

        self.cache: Dict[str, Any] = {'optimal': False}
        self.cache.update(**kwargs)

        # This is used while reading clusters from file during Cluster.from_json().
        if not argpoints:
            if 'children' in self.cache:
                self.children = {child for child in self.cache['children']}
                self.argpoints = [p for child in self.children for p in child.argpoints]
            else:
                raise ValueError(f'Cluster {name} needs argpoints of children when reading from file')
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
        if 'repr' not in self.cache:
            self.cache['repr'] = ': '.join([self.name, ', '.join(map(str, sorted(self.argpoints)))])
        return self.cache['repr']

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
        if 'depth' not in self.cache:
            self.cache['depth'] = self.name.count('0')
        return self.cache['depth']

    def distance_from(self, x1: Union[List[int], Data]) -> np.ndarray:
        """ Helper to ease calculation of distance from the cluster center. """
        return self.distance([self.argmedoid], x1)[0]

    @property
    def points(self) -> Data:
        """ An iterator, in batches, over the points in the Clusters. """
        for i in range(0, self.cardinality, BATCH_SIZE):
            yield self.manifold.data[self.argpoints[i:i + BATCH_SIZE]]

    @property
    def argsamples(self) -> Vector:
        """ Indices of samples chosen for finding poles.

        Ensures that there are at least 2 different points in samples,
        otherwise returns a single sample that represents the entire cluster.
        i.e., if len(argsamples) == 1, the cluster contains only duplicates.
        """
        if 'argsamples' not in self.cache:
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
            self.cache['argsamples'] = indices
        return self.cache['argsamples']

    @property
    def samples(self) -> Data:
        """ Returns the samples from the cluster. Samples are used in computing approximate centers and poles.
        """
        return self.manifold.data[self.argsamples]

    @property
    def nsamples(self) -> int:
        """ The number of samples for the cluster. """
        return len(self.argsamples)

    @property
    def argmedoid(self) -> int:
        """ The index used to retrieve the medoid. """
        if 'argmedoid' not in self.cache:
            logging.debug(f"building cache for {self}")
            argmedoid = np.argmin(self.distance(self.argsamples, self.argsamples).sum(axis=1))
            self.cache['argmedoid'] = self.argsamples[int(argmedoid)]
        return self.cache['argmedoid']

    @property
    def centroid(self) -> Data:
        """ The Geometric Mean of the cluster. """
        return np.average(self.samples, axis=0)

    @property
    def medoid(self) -> Data:
        """ The Geometric Median of the cluster. """
        return self.manifold.data[self.argmedoid]

    @property
    def argradius(self) -> int:
        """ The index of the point which is farthest from the medoid. """
        if ('argradius' not in self.cache) or ('radius' not in self.cache):
            logging.debug(f'building cache for {self}')

            def argmax_max(b):
                distances = self.distance_from(b)
                argmax = int(np.argmax(distances))
                return b[argmax], distances[argmax]

            argradii_radii = [argmax_max(batch) for batch in iter(self)]
            argradius, radius = max(argradii_radii, key=itemgetter(1))
            self.cache['argradius'], self.cache['radius'] = int(argradius), float(radius)
        return self.cache['argradius']

    @property
    def radius(self) -> Radius:
        """ The radius of the cluster.

        Computed as distance from medoid to the farthest point in the cluster.
        """
        if 'radius' not in self.cache:
            logging.debug(f'building cache for {self}')
            _ = self.argradius
        return self.cache['radius']

    @property
    def local_fractal_dimension(self) -> float:
        """ The local fractal dimension of the cluster. """
        if 'local_fractal_dimension' not in self.cache:
            logging.debug(f'building cache for {self}')
            if self.nsamples == 1:
                return 0.
            count = [d <= (self.radius / 2)
                     for batch in iter(self)
                     for d in self.distance_from(batch)]
            count = np.sum(count)
            self.cache['local_fractal_dimension'] = count if count == 0. else np.log2(len(self.argpoints) / count)
        return self.cache['local_fractal_dimension']

    @property
    def optimal(self) -> bool:
        return self.cache['optimal']

    def clear_cache(self) -> None:
        """ Clears the cache for the cluster. """
        logging.debug(f'clearing cache for {self}')
        self.cache = {'optimal': False}
        return

    def overlaps(self, point: Data, radius: Radius) -> bool:
        """ Checks if point is within radius + self.radius of cluster. """
        return self.distance_from(np.asarray([point]))[0] <= (self.radius + radius)

    def _find_poles(self) -> List[int]:
        """ Poles are approximately the two farthest points in the cluster.

        :return: list of indexes of the poles.
        """
        assert len(self.argsamples) > 1, f'must have more than one unique point before poles can be chosen'

        if len(self.argsamples) > 2:
            farthest = self.argsamples[int(np.argmax(self.distance([self.argradius], self.argsamples)[0]))]
            poles = [self.argradius, farthest]
        else:
            poles = [p for p in self.argsamples]

        assert len(set(poles)) == len(poles), f'poles cannot contain duplicate points.'
        return poles

    def partition(self, *criterion) -> List['Cluster']:
        """ Partition cluster into children.

        If the cluster can be partitioned, partition it and return list of children.
        Otherwise, return empty list.

        :param criterion: criteria to use to determine if a Cluster can be partitioned.
        :return: List of children.
        """
        if not all((
            len(self.argsamples) > 1,
            *(c(self) for c in criterion),
        )):  # cluster cannot be partitioned
            logging.debug(f'{self} cannot be partitioned.')
            self.children = list()
        else:
            poles: List[int] = self._find_poles()
            child_argpoints: List[List[int]] = [[p] for p in poles]

            for batch in iter(self):
                argpoints = [p for p in batch if p not in poles]
                if len(argpoints) > 0:
                    distances = self.distance(argpoints, poles)
                    [child_argpoints[int(np.argmin(row))].append(p) for p, row in zip(argpoints, distances)]

            child_argpoints.sort(key=len)
            self.children = [Cluster(self.manifold, argpoints, self.name + '0' + '1' * i) for i, argpoints in enumerate(child_argpoints)]
            logging.debug(f'{self} was partitioned into {len(self.children)} child clusters.')

        return self.children

    def _tree_search(self, point: Data, radius: Radius, depth: int) -> Dict['Cluster', Radius]:
        distance = self.distance_from(np.asarray([point]))[0]
        assert distance <= radius + self.radius, f'_tree_search was started with no overlap.'
        assert self.depth < depth, f'_tree_search needs to have depth ({depth}) > self.depth ({self.depth}). '

        # results and candidates ONLY contain clusters that have overlap with point
        results: Dict['Cluster', Radius] = dict()
        candidates: Dict['Cluster', Radius] = {self: distance}
        for _ in range(self.depth, depth):
            # if cluster was not partitioned any further, add it to results.
            results.update({cluster: distance for cluster, distance in candidates.items() if not cluster.children})

            # filter out only those candidates that were partitioned.
            candidates = {cluster: distance for cluster, distance in candidates.items() if cluster.children}

            # proceed down the tree
            children: List[Cluster] = [child for candidate in candidates.keys() for child in candidate.children]
            if len(children) == 0:
                break

            # filter out clusters that are too far away to possibly contain any hits.
            argcenters = [child.argmedoid for child in children]
            distances = self.distance(np.asarray([point]), argcenters)[0]
            radii = [radius + child.radius for child in children]
            candidates = {cluster: distance for cluster, distance, radius in zip(children, distances, radii) if distance <= radius}

            if len(candidates) == 0:
                break

        # put all potential clusters in one dictionary.
        results.update(candidates)
        assert all((depth >= cluster.depth for cluster in results.keys()))
        return results

    def tree_search(self, point: Data, radius: Radius, depth: int) -> Dict['Cluster', Radius]:
        """ Searches down the tree for clusters that overlap point with radius at depth. """
        logging.debug(f'tree_search(point={point}, radius={radius}, depth={depth}')
        if depth == -1:
            depth = len(self.manifold.layers)
        if depth < self.depth:
            raise ValueError('depth must not be less than cluster.depth')

        results: Dict['Cluster', Radius] = dict()
        if self.depth == depth:
            results = {self: self.distance_from(np.asarray([point]))[0]}
        elif self.overlaps(point, radius):
            results = self._tree_search(point, radius, depth)

        return results

    def mark(self, max_lfd: float, min_lfd: float, active: bool = False):
        """ Mark optimal Clusters via a modified depth-first traversal of the tree. """
        if active is False:
            if self.local_fractal_dimension > max_lfd:  # Mark branch as active if above given threshold
                active = True
        elif self.local_fractal_dimension < min_lfd:
            self.cache['optimal'] = True  # Active branches that fall under given threshold is marked optimal
            return  # only one cluster per branch of the tree is marked optimal

        if len(self.children) > 1:  # If there are multiple children, recurse on all children.
            [child.mark(max_lfd, min_lfd, active) for child in self.children]
        else:
            self.cache['optimal'] = True  # The first childless cluster in a branch is optimal.
        return

    def json(self):
        """ This is used for writing the manifold to disk. """
        data = {
            'name': self.name,
            'argpoints': None,  # Do not save argpoints until at leaves.
            'children': [],
            'radius': self.radius,
            'argradius': self.argradius,
            'argsamples': self.argsamples,
            'argmedoid': self.argmedoid,
            'local_fractal_dimension': self.local_fractal_dimension,
            'candidates': None if self.candidates is None else {c.name: d for c, d in self.candidates.items()},
            'optimal': self.optimal,
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
    # TODO: Write dump/load methods for Graph.
    def __init__(self, *clusters):
        logging.debug(f'Graph(clusters={[str(c) for c in clusters]})')
        assert all(isinstance(c, Cluster) for c in clusters)

        # self.edges is a dictionary of:
        #       the clusters in the graph, and
        #       the set of edges from that cluster.
        # An Edge is a named tuple of Neighbor, Distance, and Probability.
        # Edge.neighbor is is the neighboring cluster.
        # Edge.distance is the distance to that neighbor.
        # Edge.probability is sued to pick an edge during a random walk.
        self.edges: Dict[Cluster, Set[Edge]] = {cluster: None
                                                for cluster in clusters}

        self.cache: Dict[str, Any] = {'optimal': False}
        return

    def __eq__(self, other: 'Graph') -> bool:
        """ Two graphs are identical if they have the same clusters and edges.
        """
        return (set(self.clusters) == set(other.clusters)
                and all((
                    set(self.edges[cluster]) == set(other.edges[cluster])
                    for cluster in self.clusters
                )))

    def __bool__(self) -> bool:
        return self.cardinality > 0

    def __iter__(self) -> Iterable[Cluster]:
        """ An iterator over the clusters in the graph. """
        yield from self.edges

    def __str__(self) -> str:
        # Cashing value because sort can be expensive on many clusters.
        if 'str' not in self.cache:
            self.cache['str'] = ','.join(sorted(list(map(str, self.clusters))))
        return self.cache['str']

    def __repr__(self) -> str:
        # Cashing value because sort can be expensive on many clusters.
        if 'repr' not in self.cache:
            self.cache['repr'] = '\n'.join(sorted(list(map(repr, self.clusters))))
        return self.cache['repr']

    def __hash__(self):
        return hash(str(self))

    def __contains__(self, cluster: 'Cluster') -> bool:
        return cluster in self.clusters

    @property
    def cardinality(self) -> int:
        return len(self.edges)

    @property
    def population(self) -> int:
        return sum((cluster.cardinality for cluster in self.clusters))

    @property
    def manifold(self) -> 'Manifold':
        return next(iter(self.edges)).manifold

    @property
    def metric(self) -> Metric:
        return next(iter(self.edges)).metric

    @property
    def depth(self) -> int:
        if 'depth' not in self.cache:
            self.cache['depth'] = max((cluster.depth for cluster in self.clusters))
        return self.cache['depth']

    @property
    def optimal(self) -> bool:
        return self.cache['optimal']

    @property
    def clusters(self) -> Iterable[Cluster]:
        return self.edges.keys()

    @property
    def subsumed_clusters(self) -> Set[Cluster]:
        """ Set of Clusters subsumed by other clusters. """
        if 'subsumed_clusters' not in self.cache:
            self.build_edges()
        return self.cache['subsumed_clusters']

    @property
    def transition_clusters(self) -> Set[Cluster]:
        """ Set of Clusters not subsumed by other clusters. """
        if 'transition_clusters' not in self.cache:
            self.build_edges()
        return self.cache['transition_clusters']

    @property
    def subsumed_edges(self) -> Dict[Cluster, Set[Edge]]:
        """ Dict of all Clusters to set of edges to every subsumed cluster. """
        if 'subsumed_edges' not in self.cache:
            self.build_edges()
        return self.cache['subsumed_edges']

    @property
    def transition_edges(self) -> Dict[Cluster, Set[Edge]]:
        """ Transition Clusters are those not subsumed by any other Cluster. """
        if 'transition_edges' not in self.cache:
            self.build_edges()
        return self.cache['transition_edges']

    def _find_neighbors(self, cluster: Cluster):
        logging.debug(f'building edges for cluster {cluster.name}')

        # Dict of candidate neighbors and distances to neighbors.
        radius: float = cluster.manifold.root.radius

        ancestry: List[Cluster] = self.manifold.ancestry(cluster)
        for depth in range(cluster.depth):
            if ancestry[depth + 1].radius > 0:
                radius = ancestry[depth + 1].radius

            # This ensures that candidates are calculated once per cluster
            if ancestry[depth + 1].candidates is None:
                # Keep candidates from parent
                candidates: Dict[Cluster, float] = {
                    c: 0.
                    for c in ancestry[depth].candidates
                }

                # Get all children of candidates at the same depth.
                candidates.update({
                    child: 0.
                    for c in ancestry[depth].candidates
                    for child in c.children
                    if c.depth == depth
                })

                if len(candidates) > 0:
                    distances = ancestry[depth + 1].distance_from(
                        x1=[c.argmedoid for c in candidates],
                    )
                    ancestry[depth + 1].candidates = {
                        c: float(d)
                        for c, d in zip(candidates, distances)
                        if d <= c.radius + radius * 4
                    }
                else:
                    ancestry[depth + 1].candidates = dict()

        candidates = {
            c: d for c, d in cluster.candidates.items()
            if c in self.clusters
        }
        self.edges[cluster] = {
            Edge(c, d, None)
            for c, d in candidates.items()
            if d <= cluster.radius + c.radius
        }
        return

    def _mark_subsumed_clusters(self):
        logging.debug(f'marking subsumed clusters for graph: '
                      f'depth {self.depth}, clusters : {self.cardinality}')
        # Find the set of Subsumed Clusters.
        self.cache['subsumed_clusters'] = {
            cluster for cluster in self.clusters
            for edge in self.edges[cluster]
            if edge.distance <= edge.neighbor.radius - cluster.radius
        }

        self.cache['transition_clusters'] = set(self.clusters) - self.cache['subsumed_clusters']

        self.cache['subsumed_edges'] = {
            cluster: {
                edge for edge in self.edges[cluster]
                if edge.neighbor in self.cache['subsumed_clusters']
            } for cluster in self.clusters
        }

        # Populate the set of Transition Clusters.
        self.cache['transition_edges'] = {
            cluster: {
                edge for edge in self.edges[cluster]
                if edge.neighbor not in self.cache['subsumed_clusters']
            } for cluster in self.clusters
            if cluster not in self.cache['subsumed_clusters']
        }
        return

    def _compute_transition_probabilities(self):
        logging.debug(f'computing transition probabilities for graph: '
                      f'depth {self.depth}, clusters : {self.cardinality}')
        for cluster in self.cache['transition_edges']:
            # Compute transition probabilities.
            # These only exist among Transition Clusters.
            if len(self.cache['transition_edges'][cluster]) > 0:
                factor = sum((
                    1. / distance
                    for (_, distance, _) in self.cache['transition_edges'][cluster]
                ))
                self.cache['transition_edges'][cluster] = {
                    Edge(neighbor, distance, 1 / (distance * factor))
                    for (neighbor, distance, _) in self.cache['transition_edges'][cluster]
                }

                factor = sum((probability for (_, _, probability) in self.cache['transition_edges'][cluster]))
                assert abs(factor - 1.) <= 1e-6, f'transition probabilities did not sum to 1 for cluster {cluster.name}. Got {factor:.8f} instead.'
        return

    def build_edges(self) -> None:
        """ Calculates edges for the graph. """
        # build edges
        [self._find_neighbors(cluster) for cluster in self.clusters]

        # handshake between all neighbors
        [self.edges[neighbor].add(Edge(cluster, distance, None))
         for cluster, edges in self.edges.items()
         for (neighbor, distance, _) in edges]

        # Remove edges to self
        [self.edges[cluster].remove(Edge(cluster, 0., None))
         for cluster, edges in self.edges.items()
         if (cluster, 0., None) in edges]

        self._mark_subsumed_clusters()
        self._compute_transition_probabilities()
        return

    @property
    def cached_edges(self) -> Set[CacheEdge]:
        """ Returns all edges within the graph. """
        if 'edges' not in self.cache:
            logging.debug(f'building edges cache for {self}')
            if any((edges is None for edges in self.edges.values())):
                self.build_edges()

            self.cache['edges'] = {
                CacheEdge(cluster, *edge)
                for cluster, edges in self.edges.items()
                for edge in edges
            }

        return self.cache['edges']

    @property
    def subgraphs(self) -> Set['Graph']:
        """ Returns all subgraphs within the graph. """
        if any((edges is None for edges in self.edges.values())):
            self.build_edges()

        if 'subgraphs' not in self.cache:
            self.cache['subgraphs'] = set()

            unvisited: Set[Cluster] = {
                cluster for cluster in self.transition_clusters
            }
            while unvisited:
                component = self.traverse(unvisited.pop())
                unvisited = {
                    cluster for cluster in unvisited
                    if cluster not in component
                }

                self.cache['subgraphs'].add(Graph(*component))

        return self.cache['subgraphs']

    def subgraph(self, cluster: 'Cluster') -> 'Graph':
        """ Returns the subgraph to which the cluster belongs. """
        for subgraph in self.subgraphs:
            if cluster in subgraph.clusters:
                return subgraph
        else:
            raise ValueError(f'cluster {cluster.name} not found in and subgraph.')

    def clear_cache(self) -> None:
        """ Clears the cache of the graph. """
        # Clear all cached values and edges.
        self.cache = {'optimal': self.cache['optimal']}
        self.edges = {cluster: None for cluster in self.edges}
        return

    # noinspection DuplicatedCode
    def neighbors(
            self,
            cluster: Cluster,
            *,
            choice: str = 'all',
    ) -> List[Cluster]:
        """ return neighbors of a given cluster.

        :param cluster: source cluster
        :param choice: 'all' for all neighbors,
                       'transition' for only those that are not subsumed,
                       'subsumed' for only those that are subsumed.
        :return:
        """
        if self.edges[cluster] is None:
            self.build_edges()

        if choice == 'all':
            return [edge.neighbor for edge in self.edges[cluster]]
        elif choice == 'transition':
            return [edge.neighbor for edge in self.transition_edges[cluster]]
        elif choice == 'subsumed':
            return [edge.neighbor for edge in self.subsumed_edges[cluster]]
        else:
            raise ValueError(f'choice must be one of: '
                             f'\'all\', '
                             f'\'transition\', or '
                             f'\'subsumed\'. '
                             f'Got: {choice}')

    # noinspection DuplicatedCode
    def distances(
            self,
            cluster: Cluster,
            *,
            choice: str = 'all',
    ) -> List[float]:
        """ return distances to each neighbor of a given cluster. """
        if self.edges[cluster] is None:
            self.build_edges()

        if choice == 'all':
            return [edge.distance for edge in self.edges[cluster]]
        elif choice == 'transition':
            return [edge.distance for edge in self.transition_edges[cluster]]
        elif choice == 'subsumed':
            return [edge.distance for edge in self.subsumed_edges[cluster]]
        else:
            raise ValueError(f'choice must be one of: '
                             f'\'all\', '
                             f'\'transition\', or '
                             f'\'subsumed\'. '
                             f'Got: {choice}')

    # noinspection DuplicatedCode
    def probabilities(
            self,
            cluster: Cluster,
            *,
            choice: str = 'all',
    ) -> List[float]:
        """ return transition probabilities to each neighbor of a given cluster.
        """
        if self.edges[cluster] is None:
            self.build_edges()

        if choice == 'all':
            return [edge.probability for edge in self.edges[cluster]]
        elif choice == 'transition':
            return [edge.probability for edge in self.transition_edges[cluster]]
        elif choice == 'subsumed':
            return [edge.probability for edge in self.subsumed_edges[cluster]]
        else:
            raise ValueError(f'choice must be one of: '
                             f'\'all\', '
                             f'\'transition\', or '
                             f'\'subsumed\'. '
                             f'Got: {choice}')

    def random_walks(
            self,
            starts: Union[str, List[str], Cluster, List[Cluster]],
            steps: int,
    ) -> Dict[Cluster, int]:
        """ Performs random walks, counting visitations of each cluster.

        :param starts: Clusters at which to start the random walks.
        :param steps: number of steps to take per walk.
        :returns a dictionary of cluster to visit count.
        """
        if self.cardinality < 2:
            return {cluster: 1 for cluster in self.clusters}

        if type(starts) in {Cluster, str}:
            starts = [starts]
        if type(starts) is list and type(starts[0]) is str:
            starts = [self.manifold.select(cluster) for cluster in starts]

        if any((edges is None for edges in self.edges.values())):
            self.build_edges()

        if any((cluster not in self.transition_clusters for cluster in starts)):
            raise ValueError(f'random walks may only be started at clusters '
                             f'that are not subsumed by other clusters.')

        counts = {
            cluster: 1 if cluster in starts else 0
            for cluster in self.clusters
        }

        # initialize walk locations.
        walks = [
            cluster for cluster in starts
            if len(self.transition_edges[cluster]) > 0
        ]
        for _ in range(steps):
            # update walk locations
            walks = [
                np.random.choice(
                    a=list(self.neighbors(cluster, choice='transition')),
                    p=list(self.probabilities(cluster, choice='transition')),
                ) for cluster in walks
            ]
            # increment visit counts
            for cluster in walks:
                counts[cluster] += 1

        for cluster in self.transition_clusters:
            counts.update({
                neighbor: counts[cluster] + counts[neighbor]
                for (neighbor, _, _) in self.subsumed_edges[cluster]
            })
        return counts

    def traverse(self, start: Cluster) -> Set[Cluster]:
        """ Graph traversal starting at start. """
        if any((edges is None for edges in self.edges.values())):
            self.build_edges()

        if start in self.subsumed_clusters:
            raise ValueError(f'traversal may not start from subsumed clusters.')

        logging.debug(f'starting traversal from {start}')
        visited: Set[Cluster] = set()
        frontier: Set[Cluster] = {start}
        # visit all reachable transition clusters
        while frontier:
            visited.update(frontier)
            frontier = {
                neighbor for cluster in frontier
                for neighbor in self.neighbors(cluster, choice='transition')
                if neighbor not in visited
            }

        # include the clusters subsumed by visited transition clusters
        visited.update({
            edge.neighbor for cluster in visited
            for edge in self.subsumed_edges[cluster]
        })
        return visited

    def bft(self, start: Cluster) -> Set[Cluster]:
        """ Breadth-First Traversal starting at start. """
        if any((edges is None for edges in self.edges.values())):
            self.build_edges()

        if start not in self.transition_clusters:
            raise ValueError(f'traversal must not start from a subsumed cluster.')

        logging.debug(f'starting breadth-first-traversal from {start}')
        visited = set()
        queue = deque([start])
        while queue:
            cluster = queue.popleft()
            if cluster not in visited:
                visited.add(cluster)
                queue.extend((
                    neighbor
                    for neighbor in self.neighbors(cluster, choice='transition')
                    if neighbor not in visited
                ))

        subsumed_visits = {
            neighbor for cluster in visited
            for neighbor in self.neighbors(cluster, choice='subsumed')
        }
        visited.update(subsumed_visits)
        return visited

    def dft(self, start: Cluster) -> Set[Cluster]:
        """ Depth-First Traversal starting at start. """
        if any((edges is None for edges in self.edges.values())):
            self.build_edges()

        if start not in self.transition_clusters:
            raise ValueError(f'traversal must not start from a subsumed cluster.')

        logging.debug(f'starting depth-first-traversal from {start}')
        visited = set()
        stack: List[Cluster] = [start]
        while stack:
            cluster = stack.pop()
            if cluster not in visited:
                visited.add(cluster)
                stack.extend((
                    neighbor
                    for neighbor in self.neighbors(cluster, choice='transition')
                    if neighbor not in visited
                ))

        subsumed_visits = {
            neighbor for cluster in visited
            for neighbor in self.neighbors(cluster, choice='subsumed')
        }
        visited.update(subsumed_visits)
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
        self.layers: List[Graph] = [Graph(self.root)]
        self.graph: Graph = Graph(self.root)
        self.graph.cache['optimal'] = True

        self.cache: Dict[str, Any] = dict()
        self.cache.update(**kwargs)
        return

    def __eq__(self, other: 'Manifold') -> bool:
        """ Two manifolds are identical if they have the same metric and the same leaf-clusters. """
        return self.metric == other.metric and set(self.layers[-1]) == set(other.layers[-1])

    def __iter__(self) -> Iterable[Graph]:
        yield from self.layers

    def __getitem__(self, depth: int) -> Graph:
        return self.layers[depth]

    def __str__(self) -> str:
        if 'str' not in self.cache:
            self.cache['str'] = f'{self.metric}-{", ".join((str(p) for p in self.argpoints))}'
        return self.cache['str']

    def __repr__(self) -> str:
        if 'repr' not in self.cache:
            self.cache['repr'] = f'{str(self)}\n\n' + '\n\n'.join((repr(graph) for graph in self.layers))
        return self.cache['repr']

    def clear_cache(self):
        self.cache = dict()
        return

    @property
    def depth(self) -> int:
        return len(self.layers) - 1

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

    def lfd_range(self, percentiles: Tuple[float, float] = (90, 10)) -> Tuple[float, float, int]:
        """ Computes the lfd range used for marking optimal clusters. """
        lfd_range = [], []
        for depth in range(1, len(self.layers) - 1):
            if self.layers[depth + 1].cardinality < 2 ** (depth + 1):
                clusters: List[Cluster] = [cluster for cluster in self.layers[depth] if cluster.cardinality > 2]
                if len(clusters) > 0:
                    lfds = np.percentile(
                        a=[c.local_fractal_dimension for c in clusters],
                        q=percentiles,
                    )
                    lfd_range[0].append(lfds[0]), lfd_range[1].append(lfds[1])
        if len(lfd_range[0]) > 0:
            return float(np.median(lfd_range[0])), float(np.median(lfd_range[1])), self.depth - len(lfd_range[0])
        else:
            lfds = np.percentile(
                a=[cluster.local_fractal_dimension for cluster in self.layers[-1]],
                q=percentiles,
            )
            return float(lfds[0]), float(lfds[1]), self.depth

    def build(self, *criterion) -> 'Manifold':
        """ Rebuilds the Cluster-tree and the Graph-stack. """
        from pyclam.criterion import ClusterCriterion, GraphCriterion

        self.layers = [Graph(self.root)]
        self.build_tree(*(c for c in criterion if isinstance(c, ClusterCriterion)))
        self.build_graph(*(c for c in criterion if isinstance(c, GraphCriterion)))
        return self

    def build_tree(self, *criterion) -> 'Manifold':
        """ Builds the Cluster-tree. """
        while True:
            logging.info(f'depth: {self.depth}, {self.layers[-1].cardinality} clusters')
            clusters = self._partition_threaded(criterion)
            if self.layers[-1].cardinality < len(clusters):
                self.layers.append(Graph(*clusters))
            else:
                break
        return self

    def build_graph(self, *criterion):
        """ Builds the graph at the optimal depth, while also building each layer. """
        self.root.candidates = {self.root: 0.}

        # TODO: replace lfd_range and mark methods with SelectionCriteria
        max_lfd, min_lfd, grace_depth = self.lfd_range(percentiles=(90, 10))
        for depth in range(1, len(self.layers) - 1):
            if self.layers[depth + 1].cardinality < 2 ** (depth + 1):
                [cluster.mark(max_lfd, min_lfd) for cluster in self.layers[depth].clusters]
                break
        else:
            for cluster in self.layers[-1]:
                cluster.cache['optimal'] = True

        for depth, layer in enumerate(self.layers):
            logging.info(f'depth: {depth}, clusters: {layer.cardinality}')
            layer.build_edges()

        clusters: List[Cluster] = [
            cluster for layer in self.layers
            for cluster in layer.clusters
            if cluster.optimal and cluster.depth == layer.depth
        ]

        depths = [c.depth for c in clusters]
        logging.info(f'depths: ({min(depths)}, {max(depths)}), clusters: {len(clusters)}')
        self.graph = Graph(*clusters)
        self.graph.cache['optimal'] = True
        self.graph.build_edges()
        [c(self.graph) for c in criterion]
        return

    def _partition_single(self, criterion):
        # filter out clusters not previously partitioned
        new_layer: List[Cluster] = [cluster for cluster in self.layers[-1] if cluster.depth < len(self.layers) - 1]

        # get the deepest clusters. These can potentially be partitioned
        partitionable: List[Cluster] = [cluster for cluster in self.layers[-1] if cluster.depth == len(self.layers) - 1]
        [cluster.partition(*criterion) for cluster in partitionable]

        # extend new_layer to contain all the new clusters
        [new_layer.extend(cluster.children) if cluster.children else new_layer.append(cluster) for cluster in partitionable]
        return new_layer

    def _partition_threaded(self, criterion):
        new_layer: List[Cluster] = [cluster for cluster in self.layers[-1] if cluster.depth < len(self.layers) - 1]
        partitionable: List[Cluster] = [cluster for cluster in self.layers[-1] if cluster.depth == len(self.layers) - 1]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_cluster = [executor.submit(c.partition, *criterion) for c in partitionable]
            [v.result() for v in concurrent.futures.as_completed(future_to_cluster)]

        [new_layer.extend(cluster.children) if cluster.children else new_layer.append(cluster) for cluster in partitionable]
        return new_layer

    def ancestry(self, cluster: Union[str, Cluster]) -> List[Cluster]:
        """ Returns the sequence of clusters that needs to be traversed to reach the requested cluster.

        :param cluster: A cluster or the name of a cluster.
        :return: The lineage of the cluster starting at the root.
        """
        if type(cluster) is Cluster:
            cluster = cluster.name

        if cluster.count('0') > self.depth:
            raise ValueError(f'depth of requested cluster must not be greater than depth of cluster-tree. '
                             f'Got {cluster}, max-depth: {self.depth}')

        lineage: List[Cluster] = [self.root]
        if len(cluster) > 0:
            ancestry_pieces: List[str] = list(cluster.split('0'))
            ancestry = ['']
            for piece in ancestry_pieces[1:]:
                ancestry.append(ancestry[-1] + '0' + piece)

            for ancestor in ancestry[1:]:
                if lineage[-1].children:
                    for child in lineage[-1].children:
                        if child.name == ancestor:
                            lineage.append(child)
                            break
                else:
                    break

        if cluster != lineage[-1].name:
            raise ValueError(f'wanted {cluster} but got {lineage[-1].name}')
        return lineage

    def select(self, name: str) -> Cluster:
        """ Returns the cluster with the given name. """
        return self.ancestry(name)[-1]

    def find_points(self, point: Data, radius: Radius) -> List[Tuple[int, Radius]]:
        """ Returns all indices of points that are within radius of point. """
        candidates: List[int] = [p for c in self.find_clusters(point, radius, len(self.layers))
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
        return {r: d for c in self.layers[0] for r, d in c.tree_search(point, radius, depth).items()}

    def find_knn(self, point: Data, k: int) -> List[Tuple[int, Radius]]:
        """ Finds and returns the k-nearest neighbors of point. """
        radius: Radius = np.float64(np.mean([c.radius for c in self.layers[-1]]))
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
        manifold.layers = [Graph(manifold.root)]
        while True:
            for cluster in manifold.layers[-1]:
                if cluster.cache['candidates'] is None:
                    cluster.candidates = None
                else:
                    cluster.candidates = {manifold.select(c): d for c, d in cluster.cache['candidates'].items()}

            childless = [cluster for cluster in manifold.layers[-1] if not cluster.children]
            with_child = [cluster for cluster in manifold.layers[-1] if cluster.children]
            if with_child:
                graph = childless + [child for cluster in with_child for child in cluster.children]
                manifold.layers.append(Graph(*[c for c in graph]))
            else:
                break

        manifold.graph = Graph(*[cluster for layer in manifold.layers for cluster in layer if cluster.optimal])
        manifold.graph.cache['optimal'] = True
        manifold.graph.build_edges()

        return manifold
