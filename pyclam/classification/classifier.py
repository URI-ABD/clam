import logging
import typing

import numpy

from ..anomaly_detection import CHAODA
from ..utils import constants

logger = logging.getLogger(__name__)
logger.setLevel(constants.LOG_LEVEL)


class Classifier:

    def __init__(self, **kwargs):
        """ Creates and initializes a CLAM Classifier.

        See CHAODA for the list of input arguments.
        """
        self.kwargs = kwargs

        self._bowls: typing.Dict[int, CHAODA] = dict()

    def fit(self, data: numpy.ndarray, labels: typing.List[int], *, voting: str = 'mean') -> 'Classifier':
        """ Fits the Classifier to the data.

        Args:
            data: 2d array where the rows are instances and the columns are features.
            labels: List of enumerated labels for each row of data.
            voting: How to vote among scores.

        Returns:
            the fitted Classifier.
        """
        labels = numpy.array(labels)
        unique_labels = numpy.unique(labels)
        for label in unique_labels:
            indices = list(numpy.argwhere(labels == label))

            logger.info(f'Fitting CHAODA object for label {label} ...')
            bowl = CHAODA(**self.kwargs)
            bowl = bowl.fit(data, indices=indices, voting=voting)
            self._bowls[label] = bowl

        return self

    def predict_single(self, query: numpy.ndarray) -> int:
        """ Predicts the label for a single query.
        """
        labels = list(self._bowls.keys())
        scores = numpy.array(list(bowl.predict_single(query) for bowl in self._bowls.values()), dtype=numpy.float32)
        min_index = numpy.argmin(scores)
        return labels[min_index]

    def predict(self, queries: numpy.ndarray) -> typing.List[int]:
        """ Predicts the label for a set of queries.
        """
        labels = list()
        for i in range(queries.shape[0]):
            logger.info(f'Predicting class for query {i} ...')
            labels.append(self.predict_single(queries[i]))
        return labels
