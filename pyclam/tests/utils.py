""" Utilities for Testing.
"""
import numpy as np
from scipy.spatial.distance import cdist

from pyclam.types import Data, Radius
from pyclam.utils import BATCH_SIZE


def linear_search(point: Data, radius: Radius, data: Data, metric: str):
    point = np.expand_dims(point, 0)
    results = []
    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i: i + BATCH_SIZE]
        distances = cdist(point, batch, metric)[0]
        results.extend([p for p, d in zip(batch, distances) if d <= radius])
    return results
