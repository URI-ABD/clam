import logging
import pathlib
import time

import datatable
import numpy
from sklearn.metrics import accuracy_score

import csv_space
from pyclam import metric
from pyclam.classification import classifier
from pyclam.tests import synthetic_datasets
from utils import paths

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s",
    datefmt='%d-%b-%y %H:%M:%S',
)

SYNTHETIC_DATA_DIR = paths.DATA_ROOT.joinpath('synthetic_data')
SYNTHETIC_DATA_DIR.mkdir(exist_ok=True)

BULLSEYE_TRAIN_PATH = SYNTHETIC_DATA_DIR.joinpath('bullseye_train.csv')
BULLSEYE_TEST_PATH = SYNTHETIC_DATA_DIR.joinpath('bullseye_test.csv')
FEATURE_COLUMNS = ['x', 'y']
LABEL_COLUMN = 'label'


def make_bullseye(path: pathlib.Path, n: int, force: bool = False):
    if not force and path.exists():
        return

    data, labels = synthetic_datasets.bullseye(n=n, num_rings=3, noise=0.10)
    x = data[:, 0].astype(numpy.float32)
    y = data[:, 1].astype(numpy.float32)
    labels = numpy.asarray(labels, dtype=numpy.int8)

    full = datatable.Frame({'x': x, 'y': y, 'label': labels})
    full.to_csv(str(path))

    return


# noinspection DuplicatedCode
def main():
    bullseye_train = csv_space.CsvDataset(
        BULLSEYE_TRAIN_PATH,
        'bullseye_train',
        labels=LABEL_COLUMN,
    )
    bullseye_spaces = [
        csv_space.CsvSpace(bullseye_train, metric.ScipyMetric('euclidean'), False),
        csv_space.CsvSpace(bullseye_train, metric.ScipyMetric('cityblock'), False),
    ]

    start = time.perf_counter()
    bullseye_classifier = classifier.Classifier(
        bullseye_train.labels,
        bullseye_spaces,
    ).build()
    end = time.perf_counter()
    build_time = end - start

    bullseye_test = csv_space.CsvDataset(
        BULLSEYE_TEST_PATH,
        'bullseye_test',
        labels=LABEL_COLUMN,
    )

    start = time.perf_counter()
    predicted_labels, _ = bullseye_classifier.predict(bullseye_test)
    end = time.perf_counter()
    prediction_time = end - start

    score = accuracy_score(bullseye_test.labels, predicted_labels)

    print(f'Building the classifier for:')
    print(f'\t{bullseye_train.cardinality} instances and')
    print(f'\t{bullseye_classifier.unique_labels} unique labels')
    print(f'\ttook {build_time:.2e} seconds.')

    print(f'Predicting from the classifier for:')
    print(f'\t{bullseye_test.cardinality} instances took')
    print(f'\ttook {prediction_time:.2e} seconds.')

    print(f'The accuracy score was {score:.3f}')

    # Desktop   non-cached, cached
    # build,    152,        154
    # search,   105,        106
    # accuracy, 0.999,      1.000

    # M1Pro     non-cached, cached
    # build,    95.7,       96.1
    # search,   48.4,       48.7
    # accuracy, 0.999,      0.999

    return


if __name__ == '__main__':
    make_bullseye(BULLSEYE_TRAIN_PATH, n=1000, force=True)
    make_bullseye(BULLSEYE_TEST_PATH, n=200, force=True)
    main()
