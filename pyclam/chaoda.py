""" This module provides the CHAODA algorithms implemented on top of CLAM.
"""

from typing import List, Dict, Set

import numpy as np
from scipy.special import erf

from pyclam import Graph, Cluster

_NORMALIZATION_MODES = ['linear', 'gaussian', 'sigmoid']


def _normalize(scores: np.array, mode: str) -> np.array:
    if mode == 'linear':
        min_v, max_v, = float(np.min(scores)), float(np.max(scores))
        if min_v == max_v:
            max_v += 1.
        scores = (scores - min_v) / (max_v - min_v)
    else:
        mu: float = float(np.mean(scores))
        sigma: float = max(float(np.std(scores)), 1e-3)

        if mode == 'gaussian':
            scores = erf((scores - mu) / (sigma * np.sqrt(2)))
        elif mode == 'sigmoid':
            scores = 1 / (1 + np.exp(-(scores - mu) / sigma))
        else:
            raise ValueError(f'normalization mode {mode} is undefined. Use one of {_NORMALIZATION_MODES}.')

    return scores.ravel().clip(0, 1)


def cluster_cardinality(graph: Graph, normalize: str = 'gaussian') -> np.array:
    """ Determines outlier scores for points by considering the relative cardinalities of the clusters in the graph.

    Points in clusters with relatively low cardinalities are the outliers.

    :param graph: Graph on which to calculate outlier scores.
    :param normalize: which normalization mode to use to get scores in a [0, 1] range.
                      Must be one of 'linear', 'gaussian', or 'sigmoid'.
    :return: A numpy array of outlier scores for each point in the manifold that the graph belongs to.
    """
    if normalize not in _NORMALIZATION_MODES:
        raise ValueError(f'Normalization method {normalize} is undefined. Must by one of {_NORMALIZATION_MODES}.')

    scores: Dict[int, int] = {p: -cluster.cardinality for cluster in graph.clusters for p in cluster.argpoints}
    scores: List[int] = [scores[i] for i in range(len(scores))]
    return _normalize(np.asarray(scores, dtype=float), normalize)


def parent_child(graph: Graph, normalize: str = 'gaussian') -> np.array:
    """ Determines outlier scores for points by considering ratios of cardinalities of parent-child clusters.

    The ratios are weighted by the child's depth in the tree, and are then accumulated for each point in each cluster in the graph.
    Points with relatively low accumulated ration are the outliers.

    :param graph: Graph on which to calculate outlier scores.
    :param normalize: which normalization mode to use to get scores in a [0, 1] range.
                      Must be one of 'linear', 'gaussian', or 'sigmoid'.
    :return: A numpy array of outlier scores for each point in the manifold that the graph belongs to.
    """
    if normalize not in _NORMALIZATION_MODES:
        raise ValueError(f'Normalization method {normalize} is undefined. Must by one of {_NORMALIZATION_MODES}.')

    results: np.array = np.zeros(shape=graph.manifold.data.shape[0], dtype=float)
    for cluster in graph:
        ancestry = graph.manifold.ancestry(cluster)
        for i in range(1, len(ancestry)):
            score = float(ancestry[i-1].cardinality) / (ancestry[i].cardinality * np.sqrt(i))
            for p in cluster.argpoints:
                results[p] += score
    return _normalize(results, normalize)


def graph_neighborhood(graph: Graph, normalize: str = 'gaussian') -> np.array:
    """ Determines outlier scores by the considering the relative graph-neighborhood of clusters.

    Points in clusters with relatively small neighborhoods are the outliers.

    :param  graph: Graph on which to calculate outlier scores.
    :param normalize: which normalization mode to use to get scores in a [0, 1] range.
               Must be one of 'linear', 'gaussian', or 'sigmoid'.

    :return: A numpy array of outlier scores for each point in the manifold that the graph belongs to.
    """
    if normalize not in _NORMALIZATION_MODES:
        raise ValueError(f'Normalization method {normalize} is undefined. Must by one of {_NORMALIZATION_MODES}.')

    def _bft(start: Cluster) -> List[int]:
        path_length: List[int] = list()
        visited: Set[Cluster] = set()
        frontier: Set[Cluster] = {start}
        while frontier:
            visited += frontier
            path_length.append(len(visited))
            frontier = {neighbor for cluster in frontier for neighbor in graph.neighbors(cluster)}
            frontier -= visited
        return path_length

    path_lengths: Dict[Cluster, List[int]] = {cluster: _bft(cluster) for cluster in graph.clusters}
    scores: Dict[Cluster, int] = {cluster: path_length[len(path_lengths) // 4] if len(path_lengths) > 0 else 0 for cluster, path_length in path_lengths.items()}
    scores: Dict[int, int] = {point: -score for cluster, score in scores.items() for point in cluster.argpoints}
    return _normalize(np.asarray([scores[i] for i in range(len(scores))], dtype=float), normalize)


def subgraph_cardinality(graph: Graph, normalize: str = 'gaussian') -> np.array:
    """ Determines outlier scores by considering the relative cardinalities of the disconnected components of the graph.

    Points in subgraphs of relatively low cardinalities are the outliers

    :param  graph: Graph on which to calculate outlier scores.
    :param  normalize: which normalization mode to use to get scores in a [0, 1] range.
               Must be one of 'linear', 'gaussian', or 'sigmoid'.
    :return: A numpy array of outlier scores for each point in the manifold that the graph belongs to.
    """
    if normalize not in _NORMALIZATION_MODES:
        raise ValueError(f'Normalization method {normalize} is undefined. Must by one of {_NORMALIZATION_MODES}.')

    scores: Dict[int, int] = {
        p: -subgraph.cardinality
        for subgraph in graph.subgraphs
        for cluster in subgraph.clusters
        for p in cluster.argpoints
    }
    return _normalize(np.asarray([scores[i] for i in range(len(scores))], dtype=float), normalize)


# TODO: Add Random Walks
