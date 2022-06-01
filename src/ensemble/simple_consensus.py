"""
Simple voting consensus function

Multiple refences were used to develop the consensus function code

Notes
-----

Two papers were referenced during the development of the simple voting consensus function:

1. Boongoen, T. and Iam-On, N., “Cluster ensembles: A survey of approaches with recent extensions
   and applications,” Comput. Sci. Rev., vol. 28, pp. 1–25, May 2018,
   doi: 10.1016/j.cosrev.2018.01.003.
2. Topchy, A. P., Law, M. H. C., Jain, A. K. and Fred, A. L., “Analysis of consensus partition in
   cluster ensemble,” in Fourth IEEE International Conference on Data Mining (ICDM’04), Nov. 2004,
   pp. 225–232. doi: 10.1109/ICDM.2004.10100.

Two code bases were also referenced during implementation. They are:

1. hungarian-algorithm by tdedecko
   https://github.com/tdedecko/hungarian-algorithm/blob/master/hungarian.py
2. OpenEnsembles by NaegleLab https://github.com/NaegleLab/OpenEnsembles

"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def simple_voting_consensus(label_matrix):
    """
    Uses Simple Voting to the consensus between multiple sets of labels

    Parameters
    ----------
    label_matrix : matrix of ints
        The matrix of labels [[partition 1 labels], [partition 2 labels], etc.]
        Cannot have negative values for the labels (often used for noise)

    Returns
    -------
    array :
        the list of labels according to the consensus
    """

    par_r = label_matrix[0]

    k = -1  # max number of clusters
    for labels in label_matrix:
        if -1 in labels:
            raise ValueError("Negative cluster label (often used for noise) - "
                             "cluster labels must be >= 0")
        if len(labels) != len(par_r):
            raise ValueError("Each set of labels in the label_matrix must be the same length")

        k = max(k, max(labels) + 1)

    adjusted_label_matrix = [par_r]
    remaining_par = label_matrix[1:]

    for par_g in remaining_par:
        co_occ = calculate_co_occurrence_matrix(k, par_g, par_r)

        # invert values since the Hungarian algorithm is implemented as a min fn,
        # when we want a max fn
        cost = (np.ones((k, k)) * co_occ.max()) - co_occ
        relabelling = linear_sum_assignment(cost)
        new_par_g = [relabelling[1][x] for x in par_g]

        adjusted_label_matrix.append(new_par_g)

    transposed_labels = np.transpose(adjusted_label_matrix)
    labels = [np.bincount(x).argmax() for x in transposed_labels]
    return labels


def calculate_co_occurrence_matrix(k, par_g, par_r):
    """
    Calculate the co-occurrence matrix of two sets of labels

    Parameters
    ----------
    k : int
        The number of clusters/classes/unique labels
    par_g, par_r : array_like of ints
        the lists of labels to be compared

    Returns
    -------
    the co-occurrence matrix of the two sets of labels
    """
    co_occ = np.zeros((k, k))  # co-occurrence matrix
    for i in range(k):
        for j in range(k):
            count = 0
            for x, value in enumerate(par_r):
                if par_g[x] == i and value == j:
                    count += 1
            co_occ[i][j] = count
    return co_occ
