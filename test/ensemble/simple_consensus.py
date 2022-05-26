import unittest
from src.ensemble.simple_consensus import simple_voting_consensus


class TestSimpleVoting(unittest.TestCase):

    def test_when_negative_label(self):
        test_labels = [[0, 1, 1, 0, 0, -1],
                       [1, 0, 0, 1, 1, 0],
                       [0, 1, 1, 0, 0, 1]]

        with self.assertRaises(ValueError) as e:
            simple_voting_consensus(test_labels)
        self.assertEqual(e.exception.args[0],
                         "Negative cluster label (often used for noise)- "
                         "cluster labels must be >= 0")

    def test_when_mismatch_label_count(self):
        test_labels = [[0, 1, 1, 0, 0, 1],
                       [1, 0, 0, 1, 1, 0],
                       [0, 1, 1, 0, 0]]

        with self.assertRaises(ValueError) as e:
            simple_voting_consensus(test_labels)
        self.assertEqual(e.exception.args[0],
                         "Each set of labels in the label_matrix must be the same length")

    def test_when_labels_agree(self):
        test_labels = [[0, 1, 1, 0, 0, 1],
                       [1, 0, 0, 1, 1, 0],
                       [0, 1, 1, 0, 0, 1]]
        expected_results = [0, 1, 1, 0, 0, 1]

        result = simple_voting_consensus(test_labels)
        self.assertEqual(result, expected_results)

    def test_when_labels_agree_with_diff_labels(self):
        test_labels = [[0, 1, 1, 0, 0, 1],
                       [1, 2, 2, 1, 1, 2],
                       [0, 1, 1, 0, 0, 1]]
        expected_results = [0, 1, 1, 0, 0, 1]

        result = simple_voting_consensus(test_labels)
        self.assertEqual(result, expected_results)

    def test_with_slight_disagreement(self):
        test_labels = [[0, 1, 1, 0, 1, 1],
                       [1, 0, 0, 1, 1, 0],
                       [0, 1, 1, 0, 0, 1]]
        expected_results = [0, 1, 1, 0, 0, 1]

        result = simple_voting_consensus(test_labels)
        self.assertEqual(result, expected_results)

    def test_with_many_clusters(self):
        test_labels = [[1, 2, 4, 0, 3, 3, 2, 4, 0, 1],
                    [2, 3, 4, 0, 1, 1, 3, 4, 0, 2]]
        result = simple_voting_consensus(test_labels)
        self.assertEqual(result, test_labels[0])


if __name__ == '__main__':
    unittest.main()
