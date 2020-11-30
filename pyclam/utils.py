""" Some common functions and constants for all of CLAM.
"""
SUBSAMPLE_LIMIT = 100
BATCH_SIZE = 10_000
EPSILON = 1e-8


def catch_normalization_mode(mode: str) -> None:
    from typing import List

    """ Make sure that the normalization mode is allowed. """
    modes: List[str] = ['linear', 'gaussian', 'sigmoid']
    if mode not in modes:
        raise ValueError(f'Normalization method {mode} is undefined. Must by one of {modes}.')
    else:
        return


def normalize(values, mode: str):
    """ Normalizes each column in values into a [0, 1] range.

    :param values: A 1-d or 2-d array of values to normalize.
    :param mode: Normalization mode to use. Must be one of 'linear', 'gaussian', or 'sigmoid'.
    :return: array of normalized values.
    """
    import numpy as np

    squeeze = False
    if len(values.shape) == 1:
        squeeze = True
        values = np.expand_dims(values, axis=1)

    if mode == 'linear':
        min_v, max_v = np.min(values, axis=0), np.max(values, axis=0)
        for i in range(values.shape[1]):
            if min_v[i] == max_v[i]:
                max_v[i] += 1
                values[:, i] = min_v[i] + 0.5
        values = (values - min_v) / (max_v - min_v)
    else:
        mu = np.mean(values, axis=0)
        sigma = np.std(values, axis=0)

        for i in range(values.shape[1]):
            if sigma[i] < EPSILON:
                values[:, i] = 0.5
            else:
                if mode == 'gaussian':
                    from scipy.special import erf

                    values[:, i] = (1 + erf((values[:, i] - mu[i]) / (sigma[i] * np.sqrt(2)))) / 2
                else:
                    values[:, i] = 1 / (1 + np.exp(-(values[:, i] - mu[i]) / sigma[i]))

    values = values.clip(EPSILON, 1)
    if squeeze:
        values = np.squeeze(values)
    return values
