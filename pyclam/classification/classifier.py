import operator
import typing

import numpy

from .. import core
from ..anomaly_detection import CHAODA
from ..utils import helpers

logger = helpers.make_logger(__name__)


class Classifier:

    def __init__(
            self,
            labels: numpy.ndarray,
            metric_spaces: list[core.Space],
            **kwargs,
    ):
        """ Creates and initializes a CLAM Classifier.

        Args:
            metric_spaces: See `CHAODA`.
            labels: 1d array of labels. dtype must be numpy.uint.
            kwargs: These are the same as the kwargs for `CHAODA`.
        """
        if not labels.dtype == numpy.uint:
            raise ValueError(f'labels must have dtype {numpy.uint}. Got {labels.dtype} instead.')

        self.__metric_spaces = metric_spaces
        self.__labels = list(map(int, labels))
        self.__unique_labels = list(set(self.__labels))
        self.__kwargs = kwargs
        self.__bowls: typing.Dict[int, CHAODA] = dict()

    @property
    def labels(self) -> list[int]:
        return self.__labels

    @property
    def unique_labels(self) -> list[int]:
        return self.__unique_labels

    def build(self) -> 'Classifier':
        """ Fits the Classifier to the data and returns the fitted object.
        """
        for label in self.__unique_labels:
            logger.info(f'Fitting CHAODA object for label {label} ...')

            indices = [i for i, l in enumerate(self.__labels) if l == label]
            metric_spaces = [s.subset(indices, f'{s.data.name}__{label}') for s in self.__metric_spaces]

            self.__bowls[label] = CHAODA(metric_spaces, **self.__kwargs).build()

        return self

    def rank_single(self, query) -> list[tuple[int, float]]:
        """ Predicts the class rankings for a single query. Lower scores are
        better.
        """
        label_scores = list()
        for label, bowl in self.__bowls.items():
            score = bowl.predict_single(query)
            label_scores.append((label, score))

        return label_scores

    def rank(self, queries: core.Dataset) -> list[list[tuple[int, float]]]:
        """ Predicts the class rankings for a set of queries. Lower scores are
        better.
        """
        label_scores = list()
        for i in range(queries.cardinality):
            logger.info(f'Predicting class for query {i} ...')
            label_scores.append(self.rank_single(queries[i]))
        return label_scores

    def predict_single(self, query) -> tuple[int, float]:
        """ Predicts the label for a single query.
        """
        label_scores = self.rank_single(query)
        best_label, best_score = min(label_scores, key=operator.itemgetter(1))
        return best_label, best_score

    def predict(self, queries: core.Dataset) -> tuple[list[int], list[float]]:
        """ Predicts the label for a set of queries.
        """
        label_scores = list()
        for i in range(queries.cardinality):
            logger.info(f'Predicting class for query {i + 1}/{queries.cardinality} ...')
            label_scores.append(self.predict_single(queries[i]))
        [labels, scores] = list(zip(*label_scores))
        return labels, scores


__all__ = [
    'Classifier',
]
