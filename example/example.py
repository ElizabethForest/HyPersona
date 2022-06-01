"""
A short example of HyPersona in use.
"""
import pandas as pd
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from pyclustering.cluster import agglomerative
from src.hypersona import run_algorithms

# create a data set
data = datasets.make_blobs(n_samples=500, random_state=8)[0]
X = pd.DataFrame(data, columns=['Column_A', 'Column_B'])

# the algorithm map
algorithm_map = {
    # The algorithm map for a sklearn (class) style clustering algorithm implementation
    "ahc_sklearn": {"algorithm": AgglomerativeClustering,
                    "type": "class",
                    "params": {"n_clusters": [3],
                               "linkage": ["ward", "complete", "average", "single"]}},

    # The algorithm map for a pyclustering (function) style clustering algorithm implementation
    "ahc_pycl": {"algorithm": agglomerative.agglomerative,
                 "type": "function",
                 "params": {"number_clusters": [3],
                            "link": [agglomerative.type_link.CENTROID_LINK,
                                     agglomerative.type_link.COMPLETE_LINK,
                                     agglomerative.type_link.AVERAGE_LINK,
                                     agglomerative.type_link.SINGLE_LINK]}},

    # The algorithm map for an ensemble clustering algorithm implementation
    # Please note - all the parameter variations will be used for the one ensemble.
    "ahc_ensemble": {"type": "ensemble",
                     "params": {"consensus": ["basic", "nmf"]},
                     "algorithm": [{"algorithm": agglomerative.agglomerative,
                                    "type": "function",
                                    "params": {"number_clusters": [3],
                                               "link": [agglomerative.type_link.CENTROID_LINK,
                                                        agglomerative.type_link.SINGLE_LINK]}},
                                   {"algorithm": AgglomerativeClustering,
                                    "type": "class",
                                    "params": {"n_clusters": [3],
                                               "linkage": ["ward", "complete"]}}]}
}

# Run HyPersona - mandatory inputs: the data (X) and the algorithm map
# See HyPersona documentation for full list of optional arguments.
run_algorithms(X, algorithm_map, acronyms={'Column_A': 'wow!'},
               aggregate_features={'a': ['Column_A', 'Column_B'], 'b': ['Column_A']},
               graph_output_location="graphs/", always_in_personas=['Column_A', 'b'])
