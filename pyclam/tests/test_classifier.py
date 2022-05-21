import unittest

import numpy
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from pyclam import Classifier
from pyclam.utils import helpers


class TestSearch(unittest.TestCase):
    
    def test_digits(self):
        digits = load_digits()
        labels = list(map(int, digits['target']))

        data: numpy.ndarray = digits['data']
        data = helpers.normalize(data, mode='gaussian')

        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2)
        classifier = Classifier().fit(train_data, train_labels)
        predicted_labels = classifier.predict(test_data)
        score = accuracy_score(test_labels, predicted_labels)

        self.assertGreaterEqual(score, 0.5, f'score on digits dataset was too low.')
        return
