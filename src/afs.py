# pylint: disable=too-many-locals,anomalous-backslash-in-string
"""
Average Feature Significance

Calculates the average feature significance of a cluster set.
"""

from itertools import combinations
import numpy as np
from scipy.stats import ttest_ind, ttest_1samp


# TODO: docstring
def average_feature_significance(data, labels, return_significance_map=False):
    """
    Calculate the average feature significance (AFS) of a cluster set. AFS gives the average number
    of features in a cluster that significantly differ from either the population mean or the other
    clusters. See notes for details.

    Parameters
    ----------
    data : Pandas DataFrame
        The data that was clustered.
    labels : int array
        The labels assigned to the data.
    return_significance_map : bool, default=False
        Whether to also return the map with all the calculated significance for the clusters

    Returns
    -------
    afs : float
        The Average Feature Significance
    significance_map : dict, optional
        returned when `return_significance_map` is true.
        Gives the significant features compared of each cluster following the format:
        {"pop_significance":
            {0: [list of significant features compared to the population for cluster 0],
             1: [], etc...},
         "vs_significance":
            {"0 vs 1": [list of features that are significantly different between cluster 0 and
                        cluster 1],
             "0 vs 2": [], etc...}
        }

    Notes
    ------
    Average Feature Significance as defined in [1]_

    When the list of clusters is given as :math:`c = \{c_{1}, ... , c_{n}\}` and the distinct pairs
    of clusters, :math:`_{n}C_{2}`, are given as :math:`p = \{p_{1} ... , p_{m}\}`.
    Let :math:`t_{1}(c_{i}, \mu)` return the number of features in the cluster, :math:`c_{i}`,
    that are significantly different compared to the mean :math:`\mu` using a one-sample t-test.
    Let :math:`t_{2}(p_{i})` return the number of features that are significantly different between
    a pair of clusters, :math:`p_{i}`, using a two-sample t-test.

    Then, AFS can be defined as:

    .. math::

        AFS = \frac{ \sum_{i+1}^n t_{1}(c_{i}, \mu)+ \sum_{j+1}^m t_{2}(p_{i}) }/{n+m}\\

    A feature is considered statistically significant if it has a p-value less than 0.05.
    The AFS is not bounded but will always be greater than 0, with higher values meaning that,
    on average, the features of the clusters are more significantly different.
    """

    unique_labels = np.unique(labels)
    data = data.copy()
    pop_means = data.mean()

    columns = data.columns
    data["labels"] = labels

    # get all significant features compared to the population mean
    pop_sig = {}
    for label in unique_labels:
        cluster = data[data.labels == label]
        pop_sig[label] = []

        for column in columns:
            pop_mean = pop_means[column]
            _, pval = ttest_1samp(cluster[[column]], pop_mean)
            if pval[0] < 0.05:
                pop_sig[label].append(column)

    # get all significant features between clusters
    label_combinations = list(combinations(unique_labels, 2))
    vs_sig = {}
    for label_1, label_2 in label_combinations:
        cluster_1 = data[data.labels == label_1]
        cluster_2 = data[data.labels == label_2]
        vs_str = f"{label_1} vs {label_2}"
        vs_sig[vs_str] = []

        for column in columns:
            _, pval = ttest_ind(cluster_1[[column]], cluster_2[[column]])
            if pval[0] < 0.05:
                vs_sig[vs_str].append(column)

    pop_sig_counts = [len(x) for x in pop_sig.values()]
    vs_sig_counts = [len(y) for y in vs_sig.values()]
    afs = (sum(pop_sig_counts) + sum(vs_sig_counts)) / (len(pop_sig_counts) + len(vs_sig_counts))

    if return_significance_map:
        return afs, {"pop_significance": pop_sig, "vs_significance": vs_sig}

    return afs
