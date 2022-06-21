import math
import pathlib

from . import anomaly_data
from pyclam import anomaly_detection
from pyclam import metric
from pyclam.anomaly_detection import graph_scorers
from pyclam.core import cluster_criteria


def default_training(data_dir: pathlib.Path, output_dir: pathlib.Path):
    raw_datasets = [
        anomaly_data.AnomalyData.load(data_dir, name)
        for name in anomaly_data.TRAINING_SET
    ]
    datasets = [
        anomaly_detection.anomaly_dataset.AnomalyTabular(
            data=data.normalized_features,
            scores=data.scores,
            name=name
        )
        for name, data in zip(anomaly_data.TRAINING_SET, raw_datasets)
    ]

    metrics = [
        metric.ScipyMetric('euclidean'),
        metric.ScipyMetric('cityblock'),
    ]

    spaces = [
        anomaly_detection.anomaly_space.AnomalyTabularSpace(d, m, use_cache=False)
        for d in datasets
        for m in metrics
    ]
    spaces_criteria = [
        (s, [cluster_criteria.MinPoints(1 + int(math.log2(s.data.cardinality)))])
        for s in spaces
    ]

    models_kwargs = [
        (anomaly_detection.meta_ml.MetaDT, dict()),
        (anomaly_detection.meta_ml.MetaLR, dict()),
    ]

    scorers = [
        graph_scorers.ClusterCardinality(),
        graph_scorers.ComponentCardinality(),
        graph_scorers.VertexDegree(),
        graph_scorers.ParentCardinality(weight=lambda d: 1 / (d ** 0.5)),
        graph_scorers.GraphNeighborhood(eccentricity_fraction=0.25),
        graph_scorers.StationaryProbabilities(steps=16),
    ]

    final_path = anomaly_detection.training.train_meta_ml(
        spaces_criteria=spaces_criteria,
        models_kwargs=models_kwargs,
        scorers=scorers,
        out_dir=output_dir,
        num_epochs=10,
        save_frequency=1,
        only_train_fast_scorers=True,
    )

    return final_path
