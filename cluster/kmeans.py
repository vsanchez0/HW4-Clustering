import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        if type(k)==int and k>0:
            self.K=k
        else:
            raise TypeError('Ruh roh! k should be a positive integer')
        self.tol = tol
        self.centroids = None
        self.max_iter = max_iter

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        if not isinstance(mat, np.ndarray) or mat.ndim !=2:
            raise TypeError('Ruh roh! Input should be a 2D NumPy array')
        if self.K > mat.shape[0]:
            raise ValueError(f'Matrix of size {mat.shape} cannot be fit into {self.K} clusters')
        
        np.random.seed(42)
        self.centroids=mat[np.random.choice(mat.shape[0], self.K, replace=False)]

        for _ in range(self.max_iter):
            distances = cdist(mat, self.centroids)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([mat[labels==j].mean(axis=0) if np.any(labels==j) else self.centroids[j] for j in range(self.K)])
            if np.linalg.norm(self.centroids - new_centroids) < self.tol:
                break
            self.centroids = new_centroids
        

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        if not isinstance(mat, np.ndarray) or mat.ndim !=2:
            raise TypeError('Ruh roh! Input should be a 2D NumPy array')
        if mat.shape[1] != self.centroids.shape[1]:
            raise ValueError('Input data must have same number of features as fitted data')
        
        distances=cdist(mat, self.centroids)
        return np.argmin(distances, axis=1)

    def get_error(self, mat: np.ndarray) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        distances = cdist(mat, self.centroids)
        labels = np.argmin(distances, axis=1)
        return np.mean([np.linalg.norm(mat[i]-self.centroids[labels[i]])**2 for i in range(len(mat))])

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        return self.centroids
