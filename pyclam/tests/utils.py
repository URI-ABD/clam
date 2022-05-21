""" Utilities for Testing.
"""
import numpy as np
from scipy.spatial.distance import cdist

from pyclam import types
from pyclam.utils import constants


def linear_search(point: types.Dataset, radius: types.Distance, data: types.Dataset, metric: str) -> dict[int, float]:
    """ Performs naive linear search over the data and returns hits within 'radius' of 'point'. """
    point = np.expand_dims(point, 0)
    results: dict[int, float] = dict()
    for i in range(0, len(data), constants.BATCH_SIZE):
        batch = data[i: i + constants.BATCH_SIZE]
        distances = cdist(point, batch, metric)[0]
        results.update({p: d for p, d in zip(batch, distances) if d <= radius})
    return results
