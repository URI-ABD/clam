import logging
import math
import pathlib

import anomaly_data
from pyclam import anomaly_detection
from pyclam import metric
from pyclam.anomaly_detection import graph_scorers
from pyclam.core import cluster_criteria
from utils import paths

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s",
    datefmt='%d-%b-%y %H:%M:%S',
)

DATA_DIR = paths.DATA_ROOT.joinpath('anomaly_data')
OUTPUT_DIR = paths.DATA_ROOT.joinpath('trained_models')


def download_and_save(data_dir: pathlib.Path):

    for name, url in sorted(anomaly_data.DATASET_URLS.items()):
        data = anomaly_data.AnomalyData(data_dir, name, url).download().preprocess()
        save_path = data.save()

        print(f'Saved {data.name} to {save_path}')

    return


def load(data_dir: pathlib.Path):

    for name in anomaly_data.DATASET_URLS:
        data = anomaly_data.AnomalyData.load(data_dir, name)

        print(f'loaded {data.name} data with features of shape {data.features.shape}.')

    return


def default_training(data_dir: pathlib.Path, output_dir: pathlib.Path):
    totally_random_dataset_names = [
        'annthyroid',
        'mnist',
        'pendigits',
        'satellite',
        'shuttle',
        'thyroid',
    ]

    raw_datasets = [
        anomaly_data.AnomalyData.load(data_dir, name)
        for name in totally_random_dataset_names
    ]
    datasets = [
        anomaly_detection.anomaly_dataset.AnomalyTabular(
            data=data.normalized_features,
            scores=data.scores,
            name=name
        )
        for name, data in zip(totally_random_dataset_names, raw_datasets)
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


if __name__ == '__main__':
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    # download_and_save(DATA_DIR)
    # load(DATA_DIR)
    default_training(DATA_DIR, OUTPUT_DIR)
