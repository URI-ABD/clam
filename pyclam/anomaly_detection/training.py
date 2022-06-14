import pathlib
import typing

import numpy

from . import anomaly_space
from . import graph_scorers
from . import meta_ml


def default_models() -> list[typing.Type[meta_ml.MetaMLModel]]:
    return [meta_ml.MetaLR, meta_ml.MetaDT]


def data_from_graph(
        cluster_scores: graph_scorers.ClusterScores,
        labels: numpy.ndarray
) -> tuple[numpy.ndarray, numpy.ndarray]:

    clusters = list(cluster_scores.keys())
    scores = list(cluster_scores.values())

    train_x = numpy.zeros(shape=(len(clusters), 6), dtype=numpy.float32)
    train_y = numpy.zeros(shape=(len(clusters),))

    for i, (cluster, score) in enumerate(zip(clusters, scores)):
        train_x[i] = cluster.ratios

        y_true = numpy.asarray(labels[cluster.indices], dtype=numpy.float32)

        loss = float(numpy.mean(numpy.square(score - y_true))) / cluster.cardinality

        train_y[i] = 1. - loss

    return train_x, train_y


def train_meta_ml(
        metric_spaces: list[anomaly_space.AnomalySpace],
        models: list[typing.Type[meta_ml.MetaMLModel]],
        scorers: list[graph_scorers.GraphScorer],
        out_path: pathlib.Path,
):

    metric_names = [s.distance_metric.name for s in metric_spaces]

    scorer_names = [s.name for s in scorers]

    return
