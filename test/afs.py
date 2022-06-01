# pylint: disable=C,unbalanced-tuple-unpacking

import unittest
import pandas as pd
from sklearn import datasets
from src.afs import average_feature_significance


class AFSTest(unittest.TestCase):

    def test_basic(self):
        data, labels = datasets.make_blobs(n_samples=500, random_state=8)
        X = pd.DataFrame(data, columns=['Column_A', 'Column_B'])
        results = average_feature_significance(X, labels)
        self.assertAlmostEqual(1.833, results, 3)

    def test_basic_with_signficance_map(self):
        # All features should be significant
        data, labels = datasets.make_blobs(n_samples=100, random_state=7, n_features=3)
        X = pd.DataFrame(data, columns=['Column_A', 'Column_B', 'Column_C'])
        expected_sig_map = {'pop_significance': {0: ['Column_A', 'Column_B', 'Column_C'],
                                                 1: ['Column_A', 'Column_B', 'Column_C'],
                                                 2: ['Column_A', 'Column_B', 'Column_C']},
                            'vs_significance': {'0 vs 1': ['Column_A', 'Column_B', 'Column_C'],
                                                '0 vs 2': ['Column_A', 'Column_B', 'Column_C'],
                                                '1 vs 2': ['Column_A', 'Column_B', 'Column_C']}}

        results, sig_map = average_feature_significance(X, labels, return_significance_map=True)
        self.assertAlmostEqual(3, results, 3)
        self.assertEqual(expected_sig_map, sig_map)

    def test_when_no_significance(self):
        # circles will have the same centroids so not significant
        data, labels = datasets.make_circles(n_samples=100, factor=0.5, noise=0.05)
        X = pd.DataFrame(data, columns=['A', 'B'])
        expected_sig_map = {'pop_significance': {0: [], 1: []}, 'vs_significance': {'0 vs 1': []}}

        results, sig_map = average_feature_significance(X, labels, return_significance_map=True)
        self.assertEqual(0, results)
        self.assertEqual(expected_sig_map, sig_map)


if __name__ == '__main__':
    unittest.main()
