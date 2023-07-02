"""Helper functions for the package."""

import logging
import typing

import numpy
from scipy.special import erf

from . import constants

NormalizationMode = typing.Literal[
    "linear",
    "gaussian",
    "sigmoid",
]


def make_logger(name: str) -> logging.Logger:
    """Create a logger with the given name."""
    logger = logging.getLogger(name)
    logger.setLevel(constants.LOG_LEVEL)
    return logger


def next_ema(ratio: float, ema: float) -> float:
    """Computes the next exponential moving average."""
    return constants.EMA_ALPHA * ratio + (1 - constants.EMA_ALPHA) * ema


def normalize(values: numpy.ndarray, mode: NormalizationMode) -> numpy.ndarray:
    """Normalizes each column in values into a [0, 1] range.

    Args:
        values: A 1-d or 2-d array of values to normalize.
        mode: Normalization mode to use. Must be one of:
         - 'linear',
         - 'gaussian', or
         - 'sigmoid'.

    Returns:
        array of normalized values.
    """
    squeeze = False
    if len(values.shape) == 1:
        squeeze = True
        values = numpy.expand_dims(values, axis=1)

    if mode == "linear":
        min_v, max_v = numpy.min(values, axis=0), numpy.max(values, axis=0)
        values = (values - min_v) / (max_v - min_v + constants.EPSILON)
    else:
        means = numpy.mean(values, axis=0)
        sds = numpy.std(values, axis=0) + constants.EPSILON
        if mode == "gaussian":
            values = (1 + erf((values - means) / (sds * numpy.sqrt(2)))) / 2
        else:
            values = 1 / (1 + numpy.exp(-(values - means) / sds))

    values = values.clip(constants.EPSILON, 1)
    if squeeze:
        values = numpy.squeeze(values)
    return values
