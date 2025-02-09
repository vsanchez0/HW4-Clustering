import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        if not isinstance(X, np.ndarray) or X.ndim !=2:
            raise TypeError('Ruh roh! X should be a 2D NumPy array')
        if not isinstance(y, np.ndarray) or y.ndim !=1:
            raise TypeError('Ruh roh! y should be a 1D NumPy array')
        
        unique_labels=np.unique(y)
        scores=np.zeros(len(X))

        for i, xi in enumerate(X):
            same_cluster = X[(y==y[i]) & (np.arange(len(y))!=i)]
            other_clusters=[X[y==label] for label in unique_labels if label != y[i]]

            a_i = np.mean(np.linalg.norm(same_cluster - xi, axis=1)) if len(same_cluster)>1 else 0
            b_i = np.min([np.mean(np.linalg.norm(cluster - xi, axis=1)) for cluster in other_clusters]) if other_clusters else 0

            scores[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0

        return scores
