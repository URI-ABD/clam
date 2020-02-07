""" Clustered Learning of Approximate Manifolds.
"""
import concurrent.futures
import logging
import pickle
import random
from collections import deque
from operator import itemgetter
from threading import Thread
from typing import Set, Dict, Iterable, BinaryIO, List, Union, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from pyclam.types import Data, Radius, Vector, Metric

SUBSAMPLE_LIMIT = 100
BATCH_SIZE = 10_000
LOG_LEVEL = logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s:%(levelname)s:%(name)s:%(module)s.%(funcName)s:%(message)s"
)


class Cluster:
    """ A cluster of points.

    Clusters maintain references to their their children, the manifold to which they belong,
    the indices of the points they are responsible for, and neighbors (clusters with which they overlap).

    You can compare clusters, hash them, partition them, perform tree search, prune them, and more.
    In general, they implement methods that create and utilize the underlying tree structure used by Manifold.
    """

    def __init__(self, manifold: 'Manifold', argpoints: Vector, name: str, **kwargs):
        """
        A Cluster needs to know the manifold it belongs to and the indexes of the points it contains.
        The name of a Cluster indicated its position in the tree.

        :param manifold: The manifold to which the cluster belongs.
        :param argpoints: A list of indexes of the points that belong to the cluster.
        :param name: The name of the cluster indicating its position in the tree.
        """
        logging.debug(f"Cluster(name={name}, argpoints={argpoints})")
        self.manifold: 'Manifold' = manifold
        self.argpoints: Vector = argpoints
        self.name: str = name

        self.distance = self.manifold.distance

        self.neighbors: Dict['Cluster', float] = dict()  # key is neighbor, value is distance to neighbor
        self.children: Set['Cluster'] = set()

        self.__dict__.update(**kwargs)

        # This is used during Cluster.from_json().
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
        """ Be careful to use this only with other clusters. """
        return hash(self.name)

    def __str__(self) -> str:
        return self.name or 'root'

    def __repr__(self) -> str:
        return ','.join([self.name, ';'.join(map(str, self.argpoints))])

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
        """ Indices used to retrieve samples.

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
        """ The index used to retrieve the point which is farthest from the medoid. """
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

    def clear_cache(self) -> None:
        """ Clears the cache for the cluster. """
        logging.debug(f'clearing cache for {self}')
        for prop in ['_argsamples', '_argmedoid', '_argradius', '_radius', '_local_fractal_dimension']:
            try:
                del self.__dict__[prop]
            except KeyError:
                pass

    def tree_search(self, point: Data, radius: Radius, depth: int) -> Dict['Cluster', Radius]:
        """ Searches down the tree for clusters that overlap point with radius at depth. """
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
        distance = self.distance_from(np.asarray([point]))[0]
        assert distance <= radius + self.radius, f'_tree_search was started with no overlap.'
        assert self.depth < depth, f'_tree_search needs to have depth ({depth}) > self.depth ({self.depth}). '

        # results and candidates ONLY contain clusters that have overlap with point
        results: Dict['Cluster', Radius] = dict()
        candidates: Dict['Cluster', Radius] = {self: distance}
        for d_ in range(self.depth, depth):
            # if cluster was not partitioned any further, add it to results.
            results.update({c: d for c, d in candidates.items() if len(c.children) < 2})

            # filter out only those candidates that were partitioned.
            candidates = {c: d for c, d in candidates.items() if len(c.children) > 1}

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
        return results

    def partition(self, *criterion) -> Iterable['Cluster']:
        """ Partitions the cluster into 1 or 2 children.

        2 children are produced if the cluster can be split, otherwise 1 child is produced.
        """
        if not all((
                True if self.depth == 0 else self.name[-1] != '0',
                len(self.argpoints) > 1,
                len(self.argsamples) > 1,
                *(c(self) for c in criterion),
        )):
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

        farthest = self.argsamples[int(np.argmax(self.distance([self.argradius], self.argsamples)[0]))]

        p1_idx, p2_idx = list(), list()
        [(p1_idx if p1 < p2 else p2_idx).append(i)
         for batch in iter(self)
         for i, p1, p2 in zip(batch, *self.distance([self.argradius, farthest], batch))]

        # Ensure that p1 contains fewer points than p2
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

    def json(self):
        data = {
            'name': self.name,
            'argpoints': None,  # Do not save argpoints until at leaves.
            'children': [],
            'neighbors': {c.name: d for c, d in self.neighbors.items()},
            '_radius': self.radius,
            '_argradius': self.argradius,
            '_argsamples': self.argsamples,
            '_argmedoid': self.argmedoid,
            '_local_fractal_dimension': self.local_fractal_dimension,
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
    """ A Graph is comprised of clusters. All constituent clusters must be at the same depth in the tree.

    Nodes in the Graph are Clusters. .Two clusters have an edge if they have overlapping volumes.
    The Graph class is responsible for handling operations that occur solely within a layer of Manifold.graphs.
    """

    def __init__(self, *clusters):
        logging.debug(f'Graph(clusters={[str(c) for c in clusters]})')
        assert all(isinstance(c, Cluster) for c in clusters)
        assert all([c.depth == clusters[0].depth for c in clusters[1:]])

        # self.clusters is a dictionary of the clusters in the graph and the connected component subgraph that the cluster belongs to.
        self.clusters: Dict[Cluster: 'Graph'] = {c: None for c in clusters}
        return

    def __eq__(self, other: 'Graph') -> bool:
        """ Two graphs are identical if they are composed of the same clusters. """
        return set(self.clusters.keys()) == set(other.clusters.keys())

    def __bool__(self) -> bool:
        return self.cardinality > 0

    def __iter__(self) -> Iterable[Cluster]:
        """ An iterator over the clusters in the graph. """
        yield from self.clusters.keys()

    def __str__(self) -> str:
        return ';'.join(sorted([str(c) for c in self.clusters.keys()]))

    def __repr__(self) -> str:
        return '\t'.join(sorted([repr(c) for c in self.clusters.keys()]))

    def __hash__(self):
        return hash(str(self))

    def __contains__(self, cluster: 'Cluster') -> bool:
        return cluster in self.clusters.keys()

    @property
    def cardinality(self) -> int:
        return len(self.clusters)

    @property
    def population(self) -> int:
        return sum((c.cardinality for c in self.clusters))

    @property
    def manifold(self) -> 'Manifold':
        return next(iter(self.clusters.keys())).manifold

    @property
    def depth(self) -> int:
        return next(iter(self.clusters.keys())).depth

    @property
    def metric(self) -> Metric:
        return next(iter(self.clusters.keys())).metric

    def build_edges(self) -> None:
        """ Calculates edges for self by relying on tree-search. """
        def _find_neighbors(cluster: Cluster, candidates: List[Cluster], radius: float) -> Tuple[List[Cluster], float]:
            potential_candidates = [c for c in candidates if (c not in cluster.neighbors) and (c.name != cluster.name)]
            try:
                distances = [float(d) for d in cluster.distance_from([c.argmedoid for c in potential_candidates])]
            except IndexError:
                distances = []
            else:
                cluster.neighbors.update({c: d for c, d in zip(potential_candidates, distances) if d <= c.radius + cluster.radius})
                [n.neighbors.update({cluster: d}) for n, d in cluster.neighbors.items()]

            # next_candidates are to cluster.children as candidates are to cluster.
            next_candidates = set()
            if cluster.depth < self.depth:
                # include any child of potential_candidates as potential neighbors of cluster.children
                next_candidates.update([child for c, d in zip(potential_candidates, distances)
                                        for child in c.children
                                        if d <= c.radius + radius * 4])  # why 4? read the paper!
                # include cluster.children as potential neighbors of cluster.children
                next_candidates.update(cluster.children)
                # include children of neighbors as potential neighbors of cluster.children
                next_candidates.update([c for n in cluster.neighbors for c in n.children])
            # this return statement seeds the call to _find_neighbors on each of cluster.children
            # if cluster is a singleton, preserve a non-zero radius because singletons can still have neighbors.
            return list(next_candidates), cluster.radius if cluster.radius > 0. else radius

        root = self.manifold.select('')
        # this seeds the initial arguments to _find_neighbors
        layer: Dict[Cluster, Tuple[List[Cluster], float]] = {cluster: (list(root.children), root.radius) for cluster in root.children}
        for depth in range(1, self.depth + 1):
            layer = {cluster: _find_neighbors(cluster, candidates, radius) for cluster, (candidates, radius) in layer.items()}
            logging.info(f'depth {depth}, {len(layer)} clusters, {len(self.manifold.graphs[depth].subgraphs)} components')
            layer = {child: v for c, v in layer.items() for child in c.children}
        return

    @property
    def edges(self) -> Dict[Set['Cluster'], float]:
        """ Returns all edges within the graph. """
        if '_edges' not in self.__dict__:
            logging.debug(f'building cache for {self}')
            self.__dict__['_edges'] = {frozenset([c, n]): d for c in self.clusters.keys() for n, d in c.neighbors.items()}
        return self.__dict__['_edges']

    @property
    def subgraphs(self) -> Set['Graph']:
        """ Returns all subgraphs within the graph. """
        if any((s is None for s in self.clusters.values())):
            unvisited = {c for c, s in self.clusters.items() if s is None}
            while unvisited:
                component = self.traverse(unvisited.pop())
                unvisited -= component
                subgraph = Graph(*component)
                self.clusters.update({c: subgraph for c in subgraph})
        return set(self.clusters.values())

    def subgraph(self, cluster: 'Cluster') -> 'Graph':
        """ Returns the subgraph to which the cluster belongs. """
        if cluster not in self.clusters.keys():
            raise ValueError(f'Cluster {cluster} not a member of {self}')

        if self.clusters[cluster] is None:
            component = self.traverse(cluster)
            subgraph = Graph(*component)
            self.clusters.update({c: subgraph for c in subgraph})

        return self.clusters[cluster]

    def clear_cache(self) -> None:
        """ Clears the cache of the graph. """
        for prop in ['_edges']:
            logging.debug(str(self.clusters))
            try:
                del self.__dict__[prop]
            except KeyError:
                pass
        # Clear all cached subgraphs.
        self.clusters = {c: None for c in self.clusters.keys()}
        return

    def random_walk(self, steps: int = 5, walks: int = 1) -> Dict[Cluster, int]:
        """ Performs a random walk, returning a modified graph instance.

        :param int steps: number of steps per walk
        :param int walks: number of walks to perform
        :returns a Dict of cluster names to visit counts
        """
        clusters = list(self.clusters.keys())
        results = {c: list() for c in clusters}

        def walk(cluster):
            for _ in range(steps):
                results[cluster].append(1)
                if not cluster.neighbors:
                    break
                cluster = random.sample(cluster.neighbors.keys(), 1)[0]

        # Perform random walks in parallel.
        starts = random.sample(clusters, min(walks, len(clusters)))
        threads = [Thread(target=walk, args=(s,)) for s in starts]
        [t.start() for t in threads]
        [t.join() for t in threads]

        # Gather the results.
        results = {k: len(v) for k, v in results.items()}
        return results

    @staticmethod
    def traverse(start: Cluster) -> Set[Cluster]:
        """ Traversal starting at start. """
        logging.debug(f'starting traversal from {start}')
        visited, frontier = set(), {start}
        while frontier:
            visited.update(frontier)
            frontier = {n for c in frontier for n in c.neighbors.keys() if n not in visited}
        return visited

    @staticmethod
    def bft(start: Cluster) -> Set[Cluster]:
        """ Breadth-First Traversal starting at start. """
        logging.debug(f'starting breadth-first-traversal from {start}')
        visited = set()
        queue = deque([start])
        while queue:
            c = queue.popleft()
            if c not in visited:
                visited.add(c)
                [queue.append(neighbor) for neighbor in c.neighbors.keys()]
        return visited

    @staticmethod
    def dft(start: Cluster) -> Set[Cluster]:
        """ Depth-First Traversal starting at start. """
        logging.debug(f'starting depth-first-traversal from {start}')
        visited = set()
        stack: List[Cluster] = [start]
        while stack:
            c = stack.pop()
            if c not in visited:
                visited.add(c)
                stack.extend(c.neighbors.keys())
        return visited


class Manifold:
    """ Manifold of varying resolution.

    The Manifold class' main job is to organize the underlying Clusters ang Graphs.
    It does this by providing the ability to reset the build the Cluster-tree, and from them the Graph-stack.
    With this Cluster-tree and Graph-stack, Manifold provides utilities for rho-nearest neighbors search, k-nearest neighbors search.
    """

    def __init__(self, data: Data, metric: Metric, argpoints: Union[Vector, float] = None, **kwargs):
        """ A Manifold needs the data to learn the manifold for, and a distance metric to use while doing so.

        :param data: The data to learn. This could be a numpy.ndarray or a numpy.memmap.
        :param metric: The distance metric to use for the data. Any metric allowed by scipy.spatial.distance is allowed here.
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
            self.argpoints = np.random.choice(self.data.shape[0], int(self.data.shape[0] * argpoints), replace=False)
            self.argpoints = list(map(int, self.argpoints))
        else:
            raise ValueError(f"Invalid argument to argpoints. {argpoints}")

        self.graphs: List['Graph'] = [Graph(Cluster(self, self.argpoints, ''))]

        self.__dict__.update(**kwargs)
        return

    def __eq__(self, other: 'Manifold') -> bool:
        """ Two manifolds are identical if they have the same metric and the same leaf-clusters. """
        return self.metric == other.metric and self.graphs[-1] == other.graphs[-1]

    def __getitem__(self, depth: int) -> 'Graph':
        return self.graphs[depth]

    def __iter__(self) -> Iterable[Graph]:
        yield from self.graphs

    def __str__(self) -> str:
        return '\t'.join([self.metric, str(self.graphs[-1])])

    def __repr__(self) -> str:
        return '\n'.join([self.metric, repr(self.graphs[-1])])

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

    def find_points(self, point: Data, radius: Radius) -> List[Tuple[int, Radius]]:
        """ Returns all indices of points that are within radius of point. """
        candidates: List[int] = [p for c in self.find_clusters(point, radius, len(self.graphs)).keys() for p in c.argpoints]
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
        radius: Radius = np.float64(np.mean([c.radius for c in self.graphs[-1].clusters]))
        radius = np.float64(max(radius, 1e-16))
        results = self.find_points(point, radius)
        while len(results) < k:
            radius *= 2
            results = self.find_points(point, radius)

        return sorted(results, key=itemgetter(1))[:k]

    def build(self, *criterion) -> 'Manifold':
        """ Rebuilds the Cluster-tree and the Graph-stack. """
        self.graphs = [Graph(Cluster(self, self.argpoints, ''))]
        self.build_tree(*criterion)
        self.build_graphs()
        return self

    def build_tree(self, *criterion) -> 'Manifold':
        """ Builds the Cluster-tree. """
        while True:
            logging.info(f'depth: {len(self.graphs) - 1}, {len(self.graphs[-1].clusters)} clusters')
            clusters = self._partition_threaded(criterion)
            if self.graphs[-1].cardinality < len(clusters):
                g = Graph(*clusters)
                self.graphs.append(g)
            else:
                [c.children.clear() for c in self.graphs[-1]]
                break
        return self

    def build_graphs(self) -> 'Manifold':
        """ Builds the Graph-stack. """
        self.graphs[-1].build_edges()
        return self

    def build_graph(self, depth: int) -> 'Manifold':
        """ Builds the graph at a given depth. """
        if depth > self.depth:
            raise ValueError(f'depth must not be greater than {self.depth}. Got {depth}.')
        self.graphs[depth].build_edges()
        return self

    def subgraph(self, cluster: Union[str, Cluster]) -> Graph:
        """ Returns the subgraph to which cluster belongs. """
        cluster = self.select(cluster) if type(cluster) is str else cluster
        return self.graphs[cluster.depth].subgraph(cluster)

    def graph(self, cluster: Union[str, Cluster]) -> Graph:
        """ Returns the graph to which cluster belongs. """
        cluster = self.select(cluster) if type(cluster) is str else cluster
        return self.graphs[cluster.depth]

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
            raise ValueError(f'depth of requested cluster must not be greater than depth of cluster-tree. Got {cluster}, max-depth: {self.depth}')

        root: Cluster = next(iter(self.graphs[0]))
        lineage = [root]
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

    def dump(self, fp: BinaryIO) -> None:
        pickle.dump({
            'metric': self.metric,
            'root': [c.json() for c in self.graphs[0]],
        }, fp)
        return

    @staticmethod
    def load(fp: BinaryIO, data: Data) -> 'Manifold':
        d = pickle.load(fp)
        manifold = Manifold(data, metric=d['metric'])
        graphs = [Graph(*[Cluster.from_json(manifold, r) for r in d['root']])]
        while True:
            layer = Graph(*(child for cluster in graphs[-1] for child in cluster.children))
            if not layer:
                break
            else:
                graphs.append(layer)

        manifold.graphs = graphs
        for graph in graphs:
            for cluster in graph.clusters.keys():
                cluster.neighbors = {manifold.select(n): d for n, d in cluster.__dict__['neighbors'].items()}
        return manifold
