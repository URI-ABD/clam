import os
import tempfile
import unittest

import numpy
from abd_clam import dataset
from abd_clam import metric
from abd_clam import space
from abd_clam.classification import Classifier
from abd_clam.utils import helpers
from abd_clam.utils import synthetic_data
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


class TestSearch(unittest.TestCase):
    @staticmethod
    def get_score(train_x, test_x, train_y, test_y):
        data = dataset.TabularDataset(train_x, name="temp_train")
        labels = numpy.asarray(train_y).astype(numpy.uint)

        metric_spaces = [
            space.TabularSpace(data, metric.ScipyMetric("euclidean"), False),
            space.TabularSpace(data, metric.ScipyMetric("cityblock"), False),
        ]
        classifier = Classifier(labels, metric_spaces).build()

        test_labels = list(map(int, test_y))
        predicted_labels, scores = classifier.predict(
            dataset.TabularDataset(test_x, name="temp_test"),
        )
        score = accuracy_score(test_labels, predicted_labels)

        return score

    @unittest.skipIf(
        IN_GITHUB_ACTIONS, "Classification tests have high variance on github actions."
    )
    def test_digits(self):
        digits = load_digits()

        full_x = numpy.asarray(digits["data"]).astype(numpy.float32)
        full_x = helpers.normalize(full_x, mode="gaussian")
        full_y = digits["target"]

        score = self.get_score(
            *train_test_split(full_x, full_y, test_size=100, random_state=42),
        )
        assert score >= 0.75, "score on digits dataset was too low."

    @unittest.skipIf(
        IN_GITHUB_ACTIONS, "Classification tests have high variance on github actions."
    )
    def test_bullseye(self):
        full_x, full_y = synthetic_data.bullseye(n=256, num_rings=3)
        full_x = helpers.normalize(full_x, mode="gaussian")
        full_y = numpy.asarray(full_y).astype(numpy.uint)

        score = self.get_score(
            *train_test_split(full_x, full_y, test_size=100, random_state=42),
        )
        assert score >= 0.75, "score on bullseye dataset was too low."

    @unittest.skipIf(IN_GITHUB_ACTIONS, "Requires disk write access.")
    def test_cached(self):
        full_x, full_y = synthetic_data.bullseye(n=200, num_rings=3)
        full_x = helpers.normalize(full_x, mode="gaussian")
        full_y = numpy.asarray(full_y).astype(numpy.uint)

        train_x_, test_x_, train_y, test_y = train_test_split(
            full_x,
            full_y,
            test_size=100,
            random_state=42,
        )

        with tempfile.NamedTemporaryFile(
            suffix=".npy",
        ) as train_file, tempfile.NamedTemporaryFile(suffix=".npy") as test_file:
            numpy.save(train_file.name, train_x_, allow_pickle=False, fix_imports=False)
            numpy.save(test_file.name, test_x_, allow_pickle=False, fix_imports=False)

            train_data = dataset.TabularMMap(
                train_file.name,
                name="mmap_bullseye_train",
            )
            train_labels = numpy.asarray(train_y).astype(numpy.uint)

            metric_spaces = [
                space.TabularSpace(train_data, metric.ScipyMetric("euclidean"), True),
                space.TabularSpace(train_data, metric.ScipyMetric("cityblock"), True),
            ]
            classifier = Classifier(train_labels, metric_spaces).build()

            test_data = dataset.TabularMMap(test_file.name, name="mmap_bullseye_test")
            predicted_labels, scores = classifier.predict(test_data)

        test_labels = list(map(int, test_y))
        score = accuracy_score(test_labels, predicted_labels)

        assert score >= 0.75, "score on cached bullseye dataset was too low."
