import numpy as np
from sklearn.base import ClusterMixin
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize


# TODO: references, etc.
class NMFCluster(NMF, ClusterMixin):
    """
    n_components: should be the number of clusters you want
    """

    def __init__(self, n_components=None, init=None, solver='cd',
                 beta_loss='frobenius', tol=1e-4, max_iter=200,
                 random_state=None, alpha=0., l1_ratio=0., verbose=0,
                 shuffle=False):
        super().__init__(n_components=n_components, init=init, solver=solver, beta_loss=beta_loss,
                         tol=tol, max_iter=max_iter, random_state=random_state, alpha=alpha,
                         l1_ratio=l1_ratio, verbose=verbose, shuffle=shuffle)

    def fit_predict(self, X, **params):
        """
        Learns a NMF model for data X and returns a series of labels based on the normalized components

        Labels are assigned based on which component the sample has the strongest affinity for

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data matrix to be decomposed

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        W = self.fit_transform(X, **params)
        self.samples_ = W

        norm_W = normalize(W, norm='max', axis=0)
        self.labels_ = [np.argmax(w) for w in norm_W]
        return self.labels_
