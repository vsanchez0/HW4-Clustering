# Write your k-means unit tests here
import numpy as np
import pytest
from sklearn.cluster import KMeans as SklearnKmeans
from cluster import (
        KMeans, 
        Silhouette, 
        make_clusters,
        plot_clusters,
        plot_multipanel)

def test_kmeans():
    np.random.seed(42)
    X=np.random.rand(100,2)
    k=3

    custom_kmeans=KMeans(k=k)
    custom_kmeans.fit(X)
    custom_labels = custom_kmeans.predict(X)

    sklearn_kmeans = SklearnKmeans(n_clusters=k, random_state=42, n_init=10)
    sklearn_labels=sklearn_kmeans.fit_predict(X)

    custom_centroids=custom_kmeans.get_centroids()
    sklearn_centroids=sklearn_kmeans.cluster_centers_
    assert custom_centroids.shape == sklearn_centroids.shape,'Centroids shape mismatch'

    assert len(custom_labels)==len(sklearn_labels), 'Labels length mismatch'

if __name__ == '__main__':
    pytest.main()