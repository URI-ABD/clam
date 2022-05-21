import numpy
from scipy.special import erf

from . import constants


def catch_normalization_mode(mode: str) -> None:
    """ Make sure that the normalization mode is allowed. """
    modes: list[str] = ['linear', 'gaussian', 'sigmoid']
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
    squeeze = False
    if len(values.shape) == 1:
        squeeze = True
        values = numpy.expand_dims(values, axis=1)

    if mode == 'linear':
        min_v, max_v = numpy.min(values, axis=0), numpy.max(values, axis=0)
        for i in range(values.shape[1]):
            if min_v[i] == max_v[i]:
                max_v[i] += 1
                values[:, i] = min_v[i] + 0.5
        values = (values - min_v) / (max_v - min_v)
    else:
        mu = numpy.mean(values, axis=0)
        sigma = numpy.std(values, axis=0)

        for i in range(values.shape[1]):
            if sigma[i] < constants.EPSILON:
                values[:, i] = 0.5
            else:
                if mode == 'gaussian':
                    values[:, i] = (1 + erf((values[:, i] - mu[i]) / (sigma[i] * numpy.sqrt(2)))) / 2
                else:
                    values[:, i] = 1 / (1 + numpy.exp(-(values[:, i] - mu[i]) / sigma[i]))

    values = values.clip(constants.EPSILON, 1)
    if squeeze:
        values = numpy.squeeze(values)
    return values
