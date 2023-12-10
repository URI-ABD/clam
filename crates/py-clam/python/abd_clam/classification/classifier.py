"""This module contains the (experimental) Classifier class."""

import operator
import typing

import numpy

from .. import core
from ..anomaly_detection import CHAODA
from ..utils import helpers

logger = helpers.make_logger(__name__)


class Classifier:
    """A CLAM Classifier.

    This is very experimental.
    """

    def __init__(
        self,
        labels: numpy.ndarray,
        metric_spaces: typing.Sequence[core.Space],
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Creates and initializes a CLAM Classifier.

        Lower scores are better.

        Args:
            metric_spaces: See `CHAODA`.
            labels: 1d array of labels. dtype must be numpy.uint.
            kwargs: These are the same as the kwargs for `CHAODA`.
        """
        if labels.dtype != numpy.uint:
            msg = f"labels must have dtype {numpy.uint}. Got {labels.dtype} instead."
            raise ValueError(
                msg,
            )

        self.__metric_spaces = metric_spaces
        self.__labels = list(map(int, labels))
        self.__unique_labels = set(self.__labels)
        self.__kwargs = kwargs
        self.__bowls: dict[int, CHAODA] = {}

    @property
    def labels(self) -> list[int]:
        """Returns the labels in the dataset."""
        return self.__labels

    @property
    def unique_labels(self) -> set[int]:
        """Returns the unique labels in the dataset."""
        return self.__unique_labels

    def build(self) -> "Classifier":
        """Fits the Classifier to the data and returns the fitted object."""
        for label in self.__unique_labels:
            logger.info(f"Fitting CHAODA object for label {label} ...")

            indices = [i for i, _l in enumerate(self.__labels) if _l == label]
            metric_spaces = [
                s.subspace(indices, f"{s.data.name}__{label}")
                for s in self.__metric_spaces
            ]

            self.__bowls[label] = CHAODA(metric_spaces, **self.__kwargs).build()

        return self

    def rank_single(self, query: typing.Any) -> list[tuple[int, float]]:  # noqa: ANN401
        """Predicts the class rankings for a single query."""
        label_scores = []
        for label, bowl in self.__bowls.items():
            score = bowl.predict_single(query)
            label_scores.append((label, score))

        return label_scores

    def rank(self, queries: core.Dataset) -> list[list[tuple[int, float]]]:
        """Predicts the class rankings for a set of queries."""
        label_scores = []
        for i in range(queries.cardinality):
            logger.info(f"Predicting class for query {i} ...")
            label_scores.append(self.rank_single(queries[i]))
        return label_scores

    def predict_single(self, query: typing.Any) -> tuple[int, float]:  # noqa: ANN401
        """Predicts the label for a single query."""
        label_scores = self.rank_single(query)
        best_label, best_score = min(label_scores, key=operator.itemgetter(1))
        return best_label, best_score

    def predict(self, queries: core.Dataset) -> tuple[list[int], list[float]]:
        """Predicts the label for a set of queries."""
        label_scores = []
        for i in range(queries.cardinality):
            logger.info(f"Predicting class for query {i + 1}/{queries.cardinality} ...")
            label_scores.append(self.predict_single(queries[i]))

        [labels, scores] = list(zip(*label_scores))
        return labels, scores  # type: ignore[return-value]


__all__ = [
    "Classifier",
]
