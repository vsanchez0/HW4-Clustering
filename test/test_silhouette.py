# write your silhouette score unit tests here
import numpy as np
import pytest
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.metrics import silhouette_samples
from cluster import (
        KMeans, 
        Silhouette, 
        make_clusters,
        plot_clusters,
        plot_multipanel)

def test_silhouette():
    np.random.seed(42)
    X=np.random.rand(100,2)
    k=3

    sklearn_kmeans=SklearnKMeans(n_clusters=k, random_state=42, n_init=10)
    labels=sklearn_kmeans.fit_predict(X)

    custom_silhouette=Silhouette()
    custom_scores=custom_silhouette.score(X, labels)
    sklearn_scores=silhouette_samples(X, labels)

    assert custom_scores.shape == sklearn_scores.shape, 'Silhouette scores shape mismatch'
    assert np.allclose(custom_scores, sklearn_scores, atol=1e-3), 'Silhouette scores do not match within tol'

if __name__ == '__main__':
    pytest.main()