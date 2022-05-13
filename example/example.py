import pandas as pd
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from pyclustering.cluster import agglomerative
from src.hypersona import run_algorithms

data = datasets.make_blobs(n_samples=500, random_state=8)[0]
X = pd.DataFrame(data, columns=['Column_A', 'Column_B'])

algorithm_map = {
    "ahc_sklearn": {"algorithm": AgglomerativeClustering,
                    "type": "class",
                    "params": {"n_clusters": [3],
                               "linkage": ["ward", "complete", "average", "single"]}},

    "ahc_pycl": {"algorithm": agglomerative.agglomerative,
                 "type": "function",
                 "params": {"number_clusters": [3],
                            "link": [agglomerative.type_link.CENTROID_LINK,
                                     agglomerative.type_link.COMPLETE_LINK,
                                     agglomerative.type_link.AVERAGE_LINK,
                                     agglomerative.type_link.SINGLE_LINK]}},

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

run_algorithms(X, algorithm_map, aggregate_features={'a': ['Column_A', 'Column_B'], 'b': ['Column_A']},
               acronyms={'Column_A': 'wow!'}, graph_output_location="graphs/", always_in_personas=['Column_A', 'b'])
