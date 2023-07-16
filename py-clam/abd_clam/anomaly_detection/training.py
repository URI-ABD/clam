"""Utilities for training CHAODA."""

import pathlib
import typing

import numpy

from .. import core
from ..core import graph_criteria
from ..utils import helpers
from . import anomaly_space
from . import graph_scorers
from . import meta_ml

logger = helpers.make_logger(__name__)

# metric -> scorer -> list of meta-ml models
ModelsDict = dict[str, dict[graph_scorers.GraphScorer, list[meta_ml.MetaMLModel]]]

# metric -> scorer -> data for meta-ml models
ScorerData = dict[str, dict[graph_scorers.GraphScorer, numpy.ndarray]]


def data_from_graph(
    cluster_scores: graph_scorers.ClusterScores,
    labels: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Extracts data from scored clusters to use for training meta-ml models.

    Args:
        cluster_scores: The dict of cluster scores obtained by running a scorer
        on a graph.
        labels: Ground-truth anomaly scores for the points in the graph.

    Returns:
        A 2-tuple whose elements are:
            - a 2d array of feature vectors, i.e. the cluster ratios.
            - a 1d array of (1 - loss) for each cluster.
    """
    clusters = list(cluster_scores.keys())
    scores = list(cluster_scores.values())

    train_x = numpy.zeros(shape=(len(clusters), 6), dtype=numpy.float32)
    train_y = numpy.zeros(shape=(len(clusters),))

    for i, (cluster, score) in enumerate(zip(clusters, scores)):
        train_x[i] = cluster.ratios

        y_true = numpy.asarray(labels[cluster.indices], dtype=numpy.float32)

        loss = (
            float(numpy.sqrt(numpy.mean(numpy.square(score - y_true))))
            / cluster.cardinality
        )

        train_y[i] = 1.0 - loss

    return train_x, train_y


def save_models(path: pathlib.Path, meta_models: ModelsDict) -> None:
    """Saves the models in the given file."""
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
    import_lines: list[str] = list(
        {lines for import_lines in import_lines_list for lines in import_lines},
    )

    with path.open("w") as writer:
        writer.writelines(import_lines)
        writer.write("\n")

        for code in function_codes:
            writer.write(f"\n\n{code}\n")


def train_meta_ml(  # noqa: PLR0913
    *,
    spaces_criteria: typing.Sequence[
        tuple[anomaly_space.AnomalySpace, typing.Sequence[core.ClusterCriterion]]
    ],
    models_kwargs: typing.Sequence[
        tuple[type[meta_ml.MetaMLModel], dict[str, typing.Any]]
    ],
    scorers: typing.Sequence[graph_scorers.GraphScorer],
    out_dir: pathlib.Path,
    num_epochs: int = 10,
    save_frequency: int = 1,
    only_train_fast_scorers: bool = False,
) -> pathlib.Path:
    """Trains meta-ml models for CHAODA. See examples/chaoda_training.py for usage.

    Training for CHAODA takes the following steps:

    1. We build a cluster tree for each given metric space using the respective
    partition criteria. These trees are only built once. They are used to select
    graphs during training.

    2. We start the first training epoch. In this epoch, since we do not have
    pretrained meta-ml models for selecting graphs, we will select every fifth
    layer from the trees to use as graphs for training the models. For each
    subsequent epoch, we will use the meta-ml models from the previous epoch to
    select new graphs.

    3. During an epoch, from each tree, we use each graph selection criterion,
    i.e. fixed-depth layers for the first epoch and meta-ml models for other
    epochs, to select a graph.

    4. For each individual algorithm for graph scoring, we score the clusters in
    the graph. For each cluster, we compute the difference, i.e. the loss
    between the mean ground-truth scores of the points in that cluster and the
    predicted anomaly score from the graph. The training data for the models are
    six ratios for each cluster as the feature vector and (1 - loss) as the
    target.

    5. For each individual algorithm, we merge the training data from the
    current epoch with those data from previous epochs. We use these data to
    train the meta-ml models. Using data from previous epochs in this way
    ensures that models learn what cluster ratios make for good performance and
    what ratios make for poor performance on the anomaly detection task.

    6. After finishing training, and during training at the user specified
    frequency, we extract the decision functions from the trained meta-ml models
    and write them out as python functions. These exported functions can be used
    in `graph_criteria.MetaMLSelect` to select graphs for inference on new
    datasets.

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

    Returns:
        Path to file where the final models were saved.
    """
    roots = []
    for space, criteria in spaces_criteria:
        logger.info(f"Building root cluster for {space.name} ...")

        root = (
            core.Cluster.new_root(space)
            .build()
            .iterative_partition(criteria)
            .normalize_ratios("gaussian")
        )
        roots.append(root)

    metric_names = list({s.distance_metric.name for s, _ in spaces_criteria})

    meta_models: ModelsDict = {
        metric_name: {
            scorer: [model(**kwargs) for model, kwargs in models_kwargs]
            for scorer in scorers
        }
        for metric_name in metric_names
    }

    full_train_x: ScorerData = {
        metric_name: {scorer: None for scorer in scorers}
        for metric_name in metric_names
    }
    full_train_y: ScorerData = {
        metric_name: {scorer: None for scorer in scorers}
        for metric_name in metric_names
    }

    for epoch in range(1, num_epochs + 1):
        logger.info(f"Starting Epoch {epoch}/{num_epochs} ...")

        for root, (space, _) in zip(roots, spaces_criteria):
            root_name = root.metric_space.name
            logger.info(f"Epoch {epoch}/{num_epochs}: Using root {root_name} ...")

            metric_name = space.distance_metric.name
            labels = space.data.labels

            graphs: dict[graph_scorers.GraphScorer, list[core.Graph]]
            if epoch == 1:
                graphs_list = [
                    core.Graph(graph_criteria.Layer(d)(root)).build()
                    for d in range(5, root.max_leaf_depth, 5)
                ]
                graphs = {scorer: graphs_list for scorer in scorers}
            else:
                graphs = {
                    scorer: [
                        core.Graph(
                            graph_criteria.MetaMLSelect(
                                lambda ratios: model.predict(  # noqa: B023
                                    ratios[None, :],
                                ),
                                name=model.name,
                            )(root),
                        ).build()
                        for model in models
                    ]
                    for scorer, models in meta_models[metric_name].items()
                }

            for scorer in scorers:
                logger.info(
                    f"Epoch {epoch}/{num_epochs}: Using root "
                    f"{root_name} and scorer {scorer.name} ...",
                )

                for graph in graphs[scorer]:
                    if only_train_fast_scorers and (not scorer.should_be_fast(graph)):
                        continue

                    (cluster_scores, _) = scorer(graph)
                    train_x, train_y = data_from_graph(cluster_scores, labels)

                    if epoch == 1:
                        full_train_x[metric_name][scorer] = train_x
                        full_train_y[metric_name][scorer] = train_y
                    else:
                        full_train_x[metric_name][scorer] = numpy.concatenate(
                            [full_train_x[metric_name][scorer], train_x],
                            axis=0,
                        )
                        full_train_y[metric_name][scorer] = numpy.concatenate(
                            [full_train_y[metric_name][scorer], train_y],
                            axis=0,
                        )

                new_models = []
                for model in meta_models[metric_name][scorer]:
                    logger.info(
                        f"Epoch {epoch}/{num_epochs}: Fitting model "
                        f"{model.name}, scorer {scorer.name} with root "
                        f"{root_name} and scorer {scorer.name} ...",
                    )
                    new_models.append(
                        model.fit(
                            full_train_x[metric_name][scorer],
                            full_train_y[metric_name][scorer],
                        ),
                    )
                meta_models[metric_name][scorer] = new_models

        if epoch % save_frequency == 0:
            logger.info(f"Saving models after epoch {epoch}/{num_epochs} ...")
            save_models(out_dir.joinpath(f"models_epoch_{epoch}.py"), meta_models)

    final_path = out_dir.joinpath("models_final.py")
    save_models(final_path, meta_models)

    return final_path


__all__ = [
    "train_meta_ml",
]
