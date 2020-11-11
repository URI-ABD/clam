""" Some common functions and constants for all of CLAM.
"""
from typing import List

import numpy as np
from scipy.special import erf

SUBSAMPLE_LIMIT = 100
BATCH_SIZE = 10_000


def catch_normalization_mode(mode: str) -> None:
    """ Make sure that the normalization mode is allowed. """
    modes: List[str] = ['linear', 'gaussian', 'sigmoid']
    if mode not in modes:
        raise ValueError(f'Normalization method {mode} is undefined. Must by one of {modes}.')
    else:
        return


def normalize(scores: np.array, mode: str) -> np.array:
    """ Normalize a 1-d array of values into a [0, 1] range. """
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
        else:
            scores = 1 / (1 + np.exp(-(scores - mu) / sigma))

    return scores.ravel().clip(0, 1)
