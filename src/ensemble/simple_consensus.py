# TODO: proper references in the proper spot
# References:
# https://www.sciencedirect.com/science/article/pii/S1574013717300692
# https://ieeexplore.ieee.org/abstract/document/1410288

# minor reference
# https://github.com/tdedecko/hungarian-algorithm/blob/master/hungarian.py
# https://github.com/NaegleLab/OpenEnsembles

import numpy as np
from scipy.optimize import linear_sum_assignment


def simple_voting_consensus(label_matrix):
    """
    Uses Simple Voting to the consensus between multiple sets of labels

    Parameters
    ----------
    label_matrix: the matrix of labels [[partition 1 labels], [partition 2 labels], etc.]
        Currently cannot have negative values for the labels (often used for noise)

    Returns
    -------
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
        co_occ = np.zeros((k, k))  # co-occurrence matrix
        for i in range(k):
            for j in range(k):
                count = 0
                for x in range(len(par_r)):
                    if par_g[x] == i and par_r[x] == j:
                        count += 1
                co_occ[i][j] = count

        # invert values since the Hungarian algorithm is implemented as a min fn,
        # when we want a max fn
        cost = (np.ones((k, k)) * co_occ.max()) - co_occ
        relabelling = linear_sum_assignment(cost)
        new_par_g = [relabelling[1][x] for x in par_g]

        adjusted_label_matrix.append(new_par_g)

    transposed_labels = np.transpose(adjusted_label_matrix)
    labels = [np.bincount(x).argmax() for x in transposed_labels]
    return labels
