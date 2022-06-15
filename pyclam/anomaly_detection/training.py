import pathlib
import typing

import numpy

from . import anomaly_space
from . import graph_scorers
from . import meta_ml
from .. import core
from ..core import graph_criteria
from ..utils import helpers

logger = helpers.make_logger(__name__)


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


def save_models(
        path: pathlib.Path,
        meta_models: dict[str, dict[graph_scorers.GraphScorer, list[meta_ml.MetaMLModel]]],
):
    # list of tuples whose elements are:
    # - list of import lines
    # - code for function
    model_codes: list[tuple[list[str], str]] = [
        model.extract_python(metric_name, scorer.name)
        for metric_name, scorer_models in meta_models.items()
        for scorer, models in scorer_models.items()
        for model in models
    ]

    [import_lines_list, function_codes] = list(zip(*model_codes))
    import_lines: list[str] = list(set(lines for import_lines in import_lines_list for lines in import_lines))

    with open(path, 'w') as writer:
        writer.writelines(import_lines)
        writer.write('\n')

        for code in function_codes:
            writer.write(f'{code}\n\n')

    return


def train_meta_ml(
        spaces_criteria: list[tuple[anomaly_space.AnomalySpace, list[core.ClusterCriterion]]],
        models_kwargs: list[tuple[typing.Type[meta_ml.MetaMLModel], dict[str, typing.Any]]],
        scorers: list[graph_scorers.GraphScorer],  # Should all have unique names.
        out_dir: pathlib.Path,
        num_epochs: int = 10,
        save_frequency: int = 1,
        only_train_fast_scorers: bool = False,
) -> None:
    """ Trains meta-ml models for CHAODA. See examples/chaoda_training.py for
     usage.

    TODO: Fill out details of what this does

    Args:
        spaces_criteria: list of 2-tuples whose items are:
            - an AnomalySpace
            - a list of partition criteria to use for building the root cluster
              for that space.
        models_kwargs: A list of 2-tuples whose items are:
            - The meta-ml model class
            - A dictionary of key-word arguments to use for initializing an
              object of that that class.
        scorers: A list of unique graph scorers (the individual algorithms).
        out_dir: Where the training meta-ml models will be scores.
        num_epochs: The number of epochs to train the models.
        save_frequency: Save the trained models after each of this many epochs.
        only_train_fast_scorers: Whether to use the `should_be_fast` method on
            `GraphScorer`s to avoid generating training data from slow
            algorithms.
    """

    roots = list()
    for space, criteria in spaces_criteria:
        logger.info(f'Building root cluster for {space.name} ...')

        root = core.Cluster.new_root(space).build().iterative_partition(criteria)
        roots.append(root)

    metric_names = list(set(s.distance_metric.name for s, _ in spaces_criteria))

    # metric -> scorer -> list of meta-ml models
    meta_models: dict[str, dict[graph_scorers.GraphScorer, list[meta_ml.MetaMLModel]]] = {
        metric_name: {
            scorer: [model(**kwargs) for model, kwargs in models_kwargs]
            for scorer in scorers
        }
        for metric_name in metric_names
    }

    # metric -> scorer -> data for meta-ml models
    full_train_x: dict[str, dict[graph_scorers.GraphScorer, numpy.ndarray]] = {
        metric_name: {
            scorer: None
            for scorer in scorers
        }
        for metric_name in metric_names
    }
    full_train_y: dict[str, dict[graph_scorers.GraphScorer, numpy.ndarray]] = {
        metric_name: {
            scorer: None
            for scorer in scorers
        }
        for metric_name in metric_names
    }

    for epoch in range(1, num_epochs + 1):
        logger.info(f'Starting Epoch {epoch}/{num_epochs} ...')

        for root, (space, _) in zip(roots, spaces_criteria):
            logger.info(f'Epoch {epoch}/{num_epochs}: Using root {root.metric_space.name} ...')

            metric_name = space.distance_metric.name
            labels = space.data.labels

            selectors: list[graph_criteria.GraphCriterion]
            if epoch == 1:
                selectors = [graph_criteria.Layer(d) for d in range(5, root.max_leaf_depth, 5)]
            else:
                selectors = [
                    graph_criteria.MetaMLSelect(lambda ratios: model.predict(ratios), name=model.name)
                    for models in meta_models[metric_name].values()
                    for model in models
                ]

            for scorer in scorers:

                for selector in selectors:
                    graph = core.Graph(selector(root))

                    if only_train_fast_scorers and (not scorer.should_be_fast(graph)):
                        continue

                    (cluster_scores, _) = scorer(graph)
                    train_x, train_y = data_from_graph(cluster_scores, labels)

                    if epoch == 1:
                        full_train_x[metric_name][scorer] = train_x
                        full_train_y[metric_name][scorer] = train_y
                    else:
                        full_train_x[metric_name][scorer] = numpy.concatenate(full_train_x[metric_name][scorer], train_x, axis=0)
                        full_train_y[metric_name][scorer] = numpy.concatenate(full_train_y[metric_name][scorer], train_y, axis=0)

                for model in meta_models[metric_name][scorer]:
                    logger.info(f'Fitting model {model.name} with scorer {scorer.name} ...')
                    model.fit(
                        full_train_x[metric_name][scorer],
                        full_train_y[metric_name][scorer],
                    )

        if epoch % save_frequency == 0:
            save_models(out_dir.joinpath(f'models_epoch_{epoch}.py'), meta_models)

    save_models(out_dir.joinpath(f'models_final.py'), meta_models)

    return


__all__ = [
    'train_meta_ml',
]
