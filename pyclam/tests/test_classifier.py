import unittest
import tempfile

import numpy
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from pyclam import dataset
from pyclam import metric
from pyclam import space
from pyclam.classification import Classifier
from pyclam.utils import helpers
from . import synthetic_datasets


class TestSearch(unittest.TestCase):

    @staticmethod
    def get_score(train_x, test_x, train_y, test_y):
        data = dataset.TabularDataset(train_x, name='temp_train')
        labels = numpy.asarray(train_y).astype(numpy.uint)

        metric_spaces = [
            space.TabularSpace(data, metric.ScipyMetric('euclidean'), False),
            space.TabularSpace(data, metric.ScipyMetric('cityblock'), False),
        ]
        classifier = Classifier(labels, metric_spaces).build()

        test_labels = list(map(int, test_y))
        predicted_labels, scores = classifier.predict(dataset.TabularDataset(test_x, name='temp_test'))
        score = accuracy_score(test_labels, predicted_labels)

        return score
    
    def test_digits(self):
        digits = load_digits()

        full_x = numpy.asarray(digits['data']).astype(numpy.float32)
        full_x = helpers.normalize(full_x, mode='gaussian')
        full_y = digits['target']

        score = self.get_score(*train_test_split(full_x, full_y, test_size=100, random_state=42))
        self.assertGreaterEqual(score, 0.75, f'score on digits dataset was too low.')

        return

    def test_bullseye(self):

        full_x, full_y = synthetic_datasets.bullseye(n=256, num_rings=3)
        full_x = helpers.normalize(full_x, mode='gaussian')
        full_y = numpy.asarray(full_y).astype(numpy.uint)

        score = self.get_score(*train_test_split(full_x, full_y, test_size=100, random_state=42))
        self.assertGreaterEqual(score, 0.75, f'score on bullseye dataset was too low.')

        return

    def test_cached(self):

        full_x, full_y = synthetic_datasets.bullseye(n=200, num_rings=3)
        full_x = helpers.normalize(full_x, mode='gaussian')
        full_y = numpy.asarray(full_y).astype(numpy.uint)

        train_x_, test_x_, train_y, test_y = train_test_split(full_x, full_y, test_size=100, random_state=42)

        with tempfile.NamedTemporaryFile(suffix='.npy') as train_file, tempfile.NamedTemporaryFile(suffix='.npy') as test_file:
            numpy.save(train_file.name, train_x_, allow_pickle=False, fix_imports=False)
            numpy.save(test_file.name, test_x_, allow_pickle=False, fix_imports=False)

            train_data = dataset.TabularMMap(train_file.name, name='mmap_bullseye_train')
            train_labels = numpy.asarray(train_y).astype(numpy.uint)

            metric_spaces = [
                space.TabularSpace(train_data, metric.ScipyMetric('euclidean'), True),
                space.TabularSpace(train_data, metric.ScipyMetric('cityblock'), True),
            ]
            classifier = Classifier(train_labels, metric_spaces).build()

            test_data = dataset.TabularMMap(test_file.name, name='mmap_bullseye_test')
            predicted_labels, scores = classifier.predict(test_data)

        test_labels = list(map(int, test_y))
        score = accuracy_score(test_labels, predicted_labels)

        self.assertGreaterEqual(score, 0.75, f'score on cached bullseye dataset was too low.')

        return
