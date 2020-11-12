""" Some common functions and constants for all of CLAM.
"""
from typing import List

import numpy as np
from scipy.special import erf

SUBSAMPLE_LIMIT = 100
BATCH_SIZE = 10_000
EPSILON = 1e-8


def catch_normalization_mode(mode: str) -> None:
    """ Make sure that the normalization mode is allowed. """
    modes: List[str] = ['linear', 'gaussian', 'sigmoid']
    if mode not in modes:
        raise ValueError(f'Normalization method {mode} is undefined. Must by one of {modes}.')
    else:
        return


def normalize(values: np.array, mode: str) -> np.array:
    """ Normalizes values into a [0, 1] range.

    :param values: 1-d array of values to normalize.
    :param mode: Normalization mode to use. Must be one of 'linear', 'gaussian', or 'sigmoid'.
    :return: 1-d array of normalized values.
    """
    if mode == 'linear':
        min_v, max_v, = float(np.min(values)), float(np.max(values))
        if min_v == max_v:
            max_v += 1.
        values = (values - min_v) / (max_v - min_v)
    else:
        mu: float = float(np.mean(values))
        sigma: float = max(float(np.std(values)), 1e-3)

        if mode == 'gaussian':
            values = erf((values - mu) / (sigma * np.sqrt(2)))
        else:
            values = 1 / (1 + np.exp(-(values - mu) / sigma))

    return values.ravel().clip(0, 1)
