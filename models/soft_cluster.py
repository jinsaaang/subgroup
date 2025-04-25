import numpy as np
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

"""
1) model based: GMM/DP-GMM, Spectral Clustering(Affinity), HDBSCAN Distance Measure

2) embedding based: joint space clustering, linear shift, ...
"""

def _kmeans_cluster(self, z_np, K=None):
    # clustering은 추후 교체 예정, warm start(이전 centroids로 initialize)
    if K is None:
        K = int(np.sqrt(len(z_np)))
    km = KMeans(K, n_init=20, random_state=0).fit(z_np)
    return km.labels_