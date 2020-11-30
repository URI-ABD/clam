import random
import unittest

import numpy as np

from pyclam import datasets

np.random.seed(42), random.seed(42)


class TestSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        data, cls.labels = datasets.bullseye(n=1000, num_rings=3)
        anomalies = np.random.random(size=(data.shape[0] // 20, 2)) * (np.max(data, axis=0) - np.min(data, axis=0)) + np.min(data, axis=0)
        cls.data = np.concatenate([data, anomalies])
        cls.labels.extend([0 for _ in range(anomalies.shape[0])])
        return

    @unittest.skip
    def test_init(self):
        pass

    @unittest.skip
    def test_fit(self):
        pass

    @unittest.skip
    def test_predict(self):
        pass

    @unittest.skip
    def test_ensemble(self):
        pass

    @unittest.skip
    def test_vote(self):
        pass

    @unittest.skip
    def test_score_points(self):
        pass

    @unittest.skip
    def test_normalize_scores(self):
        pass

    @unittest.skip
    def test_cc(self):
        pass

    @unittest.skip
    def test_sc(self):
        pass

    @unittest.skip
    def test_gn(self):
        pass

    @unittest.skip
    def test_pc(self):
        pass

    @unittest.skip
    def test_rw(self):
        pass

    @unittest.skip
    def test_sp(self):
        pass
