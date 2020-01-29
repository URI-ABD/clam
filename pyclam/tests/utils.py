""" Utilities for Testing.
"""
import numpy as np
from scipy.spatial.distance import cdist

from chess.manifold import BATCH_SIZE, Cluster
from chess.types import Data, Radius


def linear_search(point: Data, radius: Radius, data: Data, metric: str):
    point = np.expand_dims(point, 0)
    results = []
    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i: i + BATCH_SIZE]
        distances = cdist(point, batch, metric)[0]
        results.extend([p for p, d in zip(batch, distances) if d <= radius])
    return results


def trace_lineage(cluster: Cluster, other: Cluster):  # TODO: Cover
    assert cluster.depth == other.depth
    assert cluster.overlaps(other.medoid, other.radius)
    lineage = [other.name[:i] for i in range(other.depth) if cluster.name[:i] != other.name[:i]]
    ancestors = [other.manifold.select(n) for n in reversed(lineage)]
    for ancestor in ancestors:
        print(f'checking {ancestor.name}...')
        if not cluster.overlaps(ancestor.medoid, 2 * ancestor.radius):
            print(f'{cluster.name} did not overlap with {ancestor.name}')
            distance = cluster.distance(np.asarray([ancestor.medoid], dtype=np.float64))[0]
            print(f'cluster.radius: {cluster.radius} vs ancestor.radius: {ancestor.radius}')
            print(f'distance: {distance} vs cluster.radius + 2 * ancestor.radius: {cluster.radius + 2 * ancestor.radius}')
            print(f'off by {(distance - (cluster.radius + 2 * ancestor.radius)) / distance} percent')
            print(f'cluster.depth: {cluster.depth} vs ancestor.depth: {ancestor.depth}')
            print(f'cluster_population: {len(cluster.argpoints)} vs ancestor_population: {len(ancestor.argpoints)}')
            print('\n\n\n')
            return
    else:
        raise ValueError(f'all divergent ancestors had overlap')
