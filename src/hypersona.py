import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from .afs import average_feature_significance
from .ensemble.nmf_consensus import nmf_consensus
from .ensemble.simple_consensus import simple_voting_consensus

DROPPED_LOCATION = 'dropped.csv'
ERRORS_LOCATION = 'error.csv'
METRICS_LOCATION = 'all_metrics.csv'
COLOURS = ['lightblue', 'cornflowerblue', 'steelblue', 'royalblue', 'mediumblue', 'navy']


# TODO: add tests and comments


def _explode_params(params):
    pairs = [[(k, v) for v in params[k]] for k in params]
    x = list(itertools.product(*pairs))
    maps = [dict(i) for i in x]
    return maps


# Based on sklearn classes
def _run_class_algorithm(alg, params, data):
    kwargs = params.pop('kwargs', None)
    if kwargs:
        model = alg(**params, **kwargs)
    else:
        model = alg(**params)

    labels = model.fit_predict(data)
    if kwargs:
        params['kwargs'] = kwargs

    return labels


# based on pyclustering functions
def _run_function_algorithm(alg, params, data):
    if type(data) != np.ndarray:
        data = data.to_numpy()

    instance = alg(data, **params)
    instance.process()
    clusters = instance.get_clusters()

    label_list = []
    for i in range(len(clusters)):
        for j in clusters[i]:
            label_list.append([j, i])

    label_list.sort(key=lambda x: x[0])
    labels = [x[1] for x in label_list]

    return labels


# to run ensembles
# alg should be a list of algorithms and params should be the consensus function
def _run_ensemble_algorithm(algorithms, params, data):
    consensus = params.get("consensus", "basic")  # default to simple voting consensus
    if consensus == "basic":
        consensus_method = simple_voting_consensus
    elif consensus == "nmf":
        consensus_method = nmf_consensus
    elif callable(consensus):
        consensus_method = consensus
    else:
        raise ValueError(f"Consensus {consensus} is not a valid option."
                         f" Valid options are 'basic', 'nmf', or you may pass in a custom function.")

    label_matrix = []
    counter = 0

    for details in algorithms:
        type_ = details.get("type", "class")  # default to class (sklearn style) implementation
        if type_ == "class":
            run_method = _run_class_algorithm
        elif type_ == "function":
            run_method = _run_function_algorithm
        elif type_ == "ensemble":
            raise ValueError("Cannot nest ensembles")
        else:
            raise ValueError("Unknown algorithm type within ensemble: {type_} - expecting 'class' or 'function'")

        alg = details['algorithm']
        inner_params = details.get("params", None)
        param_maps = _explode_params(inner_params) if inner_params else [{}]

        for param_map in param_maps:
            print(f"Running algorithm {counter} with {param_map} for ensemble")
            counter += 1

            labels = run_method(alg, param_map, data)
            label_matrix.append(labels)

    return consensus_method(label_matrix)


def _calculate_metrics_and_thresholds(data, labels, thresholds):
    threshold_keys = thresholds.keys()
    errors = []
    metrics = {}

    unique_labels, counts = np.unique(labels, return_counts=True)
    k = len(unique_labels)
    if ("min_clusters" in threshold_keys) and k < thresholds["min_clusters"]:
        errors.append(f"Created too few clusters: {k} - sizes {counts}")

    elif ("max_clusters" in threshold_keys) and k > thresholds["max_clusters"]:
        errors.append(f"Created too many clusters: {k} - sizes {counts}")

    # if there are too few clusters metrics cannot be calculated
    if not errors:

        min_cluster_size = thresholds["min_cluster_size"]
        if any(size < min_cluster_size for size in counts):
            errors.append(f"Created clusters that are too small (below {min_cluster_size}): {counts}")

        metrics['SC'] = silhouette_score(data, labels, metric="euclidean")
        if ("SC" in threshold_keys) and metrics['SC'] < thresholds['SC']:
            errors.append(f"SC value - {metrics['SC']} - does not meet internal threshold of {thresholds['SC']}")

        metrics['CHI'] = calinski_harabasz_score(data, labels)
        if ("CHI" in threshold_keys) and metrics['CHI'] < thresholds['CHI']:
            errors.append(f"CHI value - {metrics['CHI']} - does not meet internal threshold of {thresholds['CHI']}")

        metrics['DBI'] = davies_bouldin_score(data, labels)
        if ("DBI" in threshold_keys) and metrics['DBI'] > thresholds['DBI']:
            errors.append(f"DBI value - {metrics['DBI']} - does not meet internal threshold of {thresholds['DBI']}")

        metrics['AFS'], significance_map = average_feature_significance(data, labels, True)
        if ("AFS" in threshold_keys) and metrics['AFS'] < thresholds['AFS']:
            errors.append(f"AFS value - {metrics['AFS']} - does not meet internal threshold of {thresholds['AFS']}")

        return errors, metrics, significance_map

    else:
        return errors, metrics, None


def _create_graph(key_diff_in_std, run_id, graph_output_location):
    cluster_count, feature_count = key_diff_in_std.shape
    label_locations = np.arange(feature_count)
    labels = key_diff_in_std.columns.tolist()

    fig_width = (3 * cluster_count) + 1
    fig_height = (feature_count * .5) + 2
    fig, axes = plt.subplots(nrows=cluster_count, figsize=(fig_height, fig_width), dpi=300)

    max_diff = np.max(key_diff_in_std.values)
    min_diff = np.min(key_diff_in_std.values)

    for i in range(cluster_count):
        ax = axes[i]
        x = key_diff_in_std.loc[i]  # get data for the ith cluster
        ax.bar(label_locations, x, tick_label=labels, color=COLOURS, zorder=2)

        # Make the plot look nice
        ax.set_ylim(top=max_diff, bottom=min_diff)
        ax.grid(axis='y', zorder=0)
        ax.set_ylabel('Standard Deviations from\nPopluation Mean')
        ax.set_xlabel('Feature')
        ax.set_title(f"Cluster {i}")

    plt.suptitle(f"Clusters for {run_id}")
    fig.tight_layout()  # stops graphs overlapping
    fig.show()
    fig.savefig(f"{graph_output_location}{run_id}.svg", format='svg')


def _output_dropped(run_id, params, threshold_errors, output_location):
    error_string = [f""""{error},""" for error in threshold_errors]
    _append_string(output_location, DROPPED_LOCATION, f"""{run_id},"{params}",{error_string}\n""")


def _create_personas(always_in_personas, aggregate_features, significance_map, centroids, diffs_in_std, pop_means):
    personas = "Personas\n\nPlease note: significance is not calculated for aggregate features\n"
    for index, values in centroids.iterrows():
        personas += f"\nPersona {index}\n"
        vs_pop_sig = significance_map['pop_significance'][index]

        if always_in_personas:
            for feature in always_in_personas:
                personas += f"{feature:>19}{'*' if feature in vs_pop_sig else ' '}: " \
                            f"{round(values[feature], 2):>4.2} - " \
                            f"pop avg: {round(pop_means[feature], 2):>4.2}, " \
                            f"StD diff: {round(diffs_in_std[feature][index], 2):+.2}\n"

        if aggregate_features:
            personas += "Aggregate Features\n"
            for acr in aggregate_features:
                if acr not in always_in_personas:
                    personas += f"{acr:>20}: " \
                                f"{round(values[acr], 2):>4.2} - " \
                                f"pop avg: {round(pop_means[acr], 2):>4.2}, " \
                                f"StD diff: {round(diffs_in_std[acr][index], 2):+.2}\n"

        personas += "Features significantly different from the population: (not already mentioned)\n"
        for feature in vs_pop_sig:
            if not (feature in always_in_personas or feature in aggregate_features):
                personas += f"{feature:>20}: " \
                            f"{round(values[feature], 2):>4.2} - " \
                            f"pop avg: {round(pop_means[feature], 2):>4.2}, " \
                            f"StD diff: {round(diffs_in_std[feature][index], 2):+.2}\n"
    return personas


def _output_metrics(run_id, param_map, output_location, metrics, threshold_errors):
    metric_values = ",".join([f"{metrics[key]}" for key in metrics])
    metrics_string = f"""{run_id},"{param_map}",{metric_values},{bool(threshold_errors)}\n"""
    _append_string(output_location, METRICS_LOCATION, metrics_string)


def _build_metric_string(metrics):
    return "".join([f"{key},{metrics[key]}\n" for key in metrics])


def _build_significance_string(data, significance_map):
    output_str = "\nCluster Name, Cluster Size, vs Population Significance Count, Significant Features\n"
    output_str += "".join([f"""Cluster {label},{data[data.labels == label].shape[0]},{len(values)},"{values}"\n"""
                           for label, values in significance_map['pop_significance'].items()])

    output_str += "\nSignificant features between clusters\n"
    output_str += "".join([f"""Cluster {label},{len(values)},"{values}"\n"""
                           for label, values in significance_map["vs_significance"].items()])

    return output_str


def _write_string(output_location, suffix, string, run_id=''):
    file = open(f"{output_location}{run_id}{suffix}", 'w')
    file.write(string)
    file.close()


def _append_string(output_location, suffix, string):
    file = open(f"{output_location}{suffix}", 'a')
    file.write(string)
    file.close()


def _reset_files(output_location):
    _write_string(output_location, ERRORS_LOCATION, 'Run ID, Parameters, Error\n')
    _write_string(output_location, DROPPED_LOCATION, 'Run ID, Parameters, Dropped Reason\n')
    _write_string(output_location, METRICS_LOCATION, 'Run ID, Parameters, SC, CHI, DBI, AFS, Dropped\n')


# TODO: finish doc-string and add docstrings to other places
# expects data as a pandas data frame
def run_algorithms(data, algorithm_map, key_features=None, aggregate_features=None, thresholds=None, acronyms=None,
                   output_location="", graph_output_location=None, always_in_personas=None, reset_files=True):
    """
    Runs the HyPersona framework. See [docs] or [paper] for more details. TODO

    Parameters
    ----------
    data : Pandas DataFrame
        The data to be clustered.

    algorithm_map : dict
        The algorithms and parameters to be compared. See [xxx] for more details. TODO: write up explanation

    key_features : list, default=None
        The list of "key features" to be included in the graphs.
        When set to None, all features, including any aggregate features, are considered "key features".

    aggregate_features : dict, default=None

    thresholds : dict, default=None

    acronyms : dict, default=None

    output_location : String, default=""
        The location for all output (.csv files, personas, and graphs) to be saved to

    graph_output_location : String, default=""
        The location for the graphs to be saved to. Will override the output_location just for the graph output.

    always_in_personas : list, default=None

    reset_files : bool, default=True
        Whether the files that are appended to (metrics.csv, dropped.csv, and error.csv) should be reset, or whether
        additional information should just be appended to the existing files.

    Returns
    ----------
    Nothing. See [xxx] for details on output. TODO
    """
    # default graph output location to output location
    if not graph_output_location:
        graph_output_location = output_location

    # if no key features set, use all - includes agg features
    if not key_features:
        key_features = data.columns.tolist() + list(aggregate_features.keys())

    # use default thresholds if not set
    if not thresholds:
        row_count, feature_count = data.shape
        thresholds = {"min_clusters": 2,
                      "min_cluster_size": (row_count * 0.05),
                      "AFS": (feature_count * 0.3),
                      "SC": 0,
                      "CHI": 5,
                      "DBI": 5}

    if reset_files:
        _reset_files(output_location)

    for algorithm, details in algorithm_map.items():
        print(f"\nRunning {algorithm}")

        type_ = details.get("type", "class")  # default to class (sklearn style) implementation
        if type_ == "class":
            run_method = _run_class_algorithm
        elif type_ == "function":
            run_method = _run_function_algorithm
        elif type_ == "ensemble":
            run_method = _run_ensemble_algorithm
        else:
            print(f"Unknown algorithm type: {type_} - expecting 'class', 'function', or 'ensemble'")
            continue

        alg = details['algorithm']

        params = details.get("params", None)
        param_maps = _explode_params(params) if params else [{}]

        counter = 0
        for param_map in param_maps:
            run_id = f"{algorithm}_v{counter}"
            counter += 1
            print(f"\nRunning {run_id} with parameters: {param_map}")

            try:
                labels = run_method(alg, param_map, data)
                threshold_errors, metrics, significances = _calculate_metrics_and_thresholds(data, labels, thresholds)
                _output_metrics(run_id, param_map, output_location, metrics, threshold_errors)

                if threshold_errors:
                    # algorithm-parameter combination is dropped
                    _output_dropped(run_id, params, threshold_errors, output_location)

                else:
                    columns = data.columns.tolist()
                    data_clustered = data.copy()
                    data_clustered['labels'] = labels

                    # output clustered data
                    data_clustered.to_csv(f"{output_location}{run_id}_data.csv")

                    # Calculate aggregate features
                    if aggregate_features:
                        for acr, features in aggregate_features.items():
                            data_clustered[acr] = data_clustered[features].mean(axis=1)
                            columns.append(acr)

                    # begin building string for output
                    output_str = f"""{run_id},"{param_map}"\n\nMetrics\n"""
                    output_str += _build_metric_string(metrics)
                    output_str += _build_significance_string(data_clustered, significances)

                    cluster_headings = ','.join([f"Cluster {i} Mean,Cluster {i} StD from Population"
                                                 for i in significances['pop_significance'].keys()])
                    output_str += f"\nColumn,Population Mean,Population Std,{cluster_headings}\n"

                    diffs_in_std = pd.DataFrame()
                    centroids = data_clustered.groupby('labels').mean()
                    pop_means = data_clustered.mean()

                    # get the distance each feature of each centroid is from the population mean in standard deviations
                    for column in columns:
                        col_mean = pop_means[column]
                        col_std = data_clustered[column].std()
                        cluster_means = centroids[column]

                        diffs_in_std[column] = [(col_mean - clus_mean) / col_std for clus_mean in cluster_means]

                        cluster_values = ','.join([f"{x},{y}" for x, y in zip(cluster_means, diffs_in_std[column])])
                        output_str += f"{column}{' (aggregate)' if column in aggregate_features else ''}," \
                                      f"{col_mean},{col_std},{cluster_values}\n"

                    _write_string(output_location, '.csv', output_str, run_id)

                    # create data for graphs
                    graph_data = diffs_in_std[key_features].copy()
                    if acronyms:
                        graph_data.rename(columns=acronyms, inplace=True)
                    _create_graph(graph_data, run_id, graph_output_location)

                    personas = _create_personas(always_in_personas, aggregate_features, significances, centroids,
                                                diffs_in_std, pop_means)
                    _write_string(output_location, '_personas.txt', personas, run_id)

            # catch any/all exceptions to allow the framework to keep running
            except Exception as e:
                error_type = e.__class__.__name__
                print(f"An error occurred running {run_id}: \n{error_type}: {e}")
                _append_string(output_location, ERRORS_LOCATION, f"""{run_id},"{param_map}",{error_type}:{e}\n""")
