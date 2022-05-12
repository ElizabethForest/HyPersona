from itertools import combinations
from scipy.stats import ttest_ind, ttest_1samp
import numpy as np


def average_feature_significance(data, labels, return_significance_map=False):
    unique_labels, counts = np.unique(labels, return_counts=True)

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
            tstat, pval = ttest_1samp(cluster[[column]], pop_mean)
            if pval[0] < 0.05:
                pop_sig[label].append(column)

    # get all significant features between clusters
    label_combinations = list(combinations(unique_labels, 2))
    vs_sig = {}
    for label_1, label_2 in label_combinations:
        cluster_1 = data[data.labels == label_1]
        cluster_2 = data[data.labels == label_2]
        vs = f"{label_1} vs {label_2}"
        vs_sig[vs] = []

        for column in columns:
            tstat, pval = ttest_ind(cluster_1[[column]], cluster_2[[column]])
            if pval[0] < 0.05:
                vs_sig[vs].append(column)

    pop_sig_counts = [len(x) for x in pop_sig.values()]
    vs_sig_counts = [len(y) for y in vs_sig.values()]
    afs = (sum(pop_sig_counts) + sum(vs_sig_counts)) / (len(pop_sig_counts) + len(vs_sig_counts))

    if return_significance_map:
        return afs, {"pop_significance": pop_sig, "vs_significance": vs_sig}
    else:
        return afs
