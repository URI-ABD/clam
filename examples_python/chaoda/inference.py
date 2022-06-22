import json
import math
import pathlib
import time

from . import anomaly_data
import pyclam
import sklearn

logger = pyclam.utils.helpers.make_logger(__name__)


def run_one_dataset(
        data_dir: pathlib.Path,
        name: str,
        metrics: list[pyclam.Metric],
        output_dir: pathlib.Path,
):
    raw_data = anomaly_data.AnomalyData.load(data_dir, name)
    dataset = pyclam.dataset.TabularDataset(
        data=raw_data.normalized_features,
        name=name,
    )

    spaces = [
        pyclam.space.TabularSpace(dataset, metric, False)
        for metric in metrics
    ]

    min_cardinality: int
    if dataset.cardinality < 10_000:
        min_cardinality = 1
    elif dataset.cardinality < 100_000:
        min_cardinality = 1 + int(math.log2(dataset.cardinality))
    else:
        min_cardinality = 1 + int(math.sqrt(dataset.cardinality))

    criteria = [pyclam.cluster_criteria.MinPoints(min_cardinality)]

    start = time.perf_counter()
    chaoda = pyclam.anomaly_detection.CHAODA(
        spaces,
        partition_criteria=criteria,
    )
    predicted_scores = chaoda.fit_predict()
    time_taken = time.perf_counter() - start

    roc_score = sklearn.metrics.roc_auc_score(raw_data.scores, predicted_scores)

    logger.info(f'Dataset {name} scored {roc_score:.3f} in {time_taken:.2e} seconds.')

    results = {
        'roc_score': f'{roc_score:.6f}',
        'time_taken': f'{time_taken:.2e} seconds',
        'predicted_scores': [f'{s:.6f}' for s in predicted_scores],
    }
    results_path = output_dir.joinpath(name)
    results_path.mkdir(exist_ok=True)

    with open(results_path.joinpath(f'results.json'), 'w') as writer:
        json.dump(results, writer, indent=4)

    return


def compile_results(output_dir: pathlib.Path):
    full_results = dict()
    for name in anomaly_data.INFERENCE_SET:
        full_results[name] = dict()

        with open(output_dir.joinpath(name).joinpath(f'results.json'), 'r') as reader:
            results = json.load(reader)
        full_results[name]['roc_score'] = results['roc_score']
        full_results[name]['time_taken'] = results['time_taken']

    with open(output_dir.joinpath('full_results.json'), 'w') as writer:
        json.dump(full_results, writer, indent=4)

    return


def run_inference(data_dir: pathlib.Path, output_dir: pathlib.Path):

    metrics = [
        pyclam.metric.ScipyMetric('euclidean'),
        pyclam.metric.ScipyMetric('cityblock'),
    ]

    for name in anomaly_data.INFERENCE_SET:
        logger.info(f'Staring CHAODA inference on {name} ...')
        run_one_dataset(data_dir, name, metrics, output_dir)

    compile_results(output_dir)
    return
