import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

"""
1) model based: GMM/DP-GMM, Spectral Clustering(Affinity), HDBSCAN Distance Measure

2) embedding based: joint space clustering, linear shift, ...
"""

class SpectralClustering:
    def __init__(self, n_clusters=None, eigengap_threshold=0.1):
        """
        Initialize the SoftCluster class.

        Parameters:
        - n_clusters: Number of clusters. If None, it will be determined using the eigengap heuristic.
        - eigengap_threshold: Threshold for determining the optimal number of clusters using the eigengap.
        """
        self.n_clusters = n_clusters
        self.eigengap_threshold = eigengap_threshold

    def _compute_eigengap(self, eigenvalues):
        """
        Compute the eigengap to determine the optimal number of clusters.

        Parameters:
        - eigenvalues: Sorted eigenvalues of the Laplacian matrix.

        Returns:
        - Optimal number of clusters based on the eigengap heuristic.
        """
        gaps = np.diff(eigenvalues)
        optimal_clusters = np.argmax(gaps > self.eigengap_threshold) + 1
        return optimal_clusters
    
    def error_weighted_affinity(z, r_adj, gamma_z=1.0, gamma_r=1.0):
        A_z = rbf_kernel(z, z, gamma=gamma_z)
        R = np.abs(r_adj[:, None] - r_adj[None, :])
        A_r = np.exp(-gamma_r * R)
        return A_z * A_r

    def fit(self, X):
        """
        Fit the soft clustering model to the data.

        Parameters:
        - X: Input data, shape (n_samples, n_features).

        Returns:
        - cluster_probabilities: Soft cluster assignments, shape (n_samples, n_clusters).
        """
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(X)

        # Compute eigenvalues and eigenvectors of the Laplacian
        eigenvalues, eigenvectors = np.linalg.eigh(similarity_matrix)
        eigenvalues = np.sort(eigenvalues)

        # Determine the number of clusters using eigengap heuristic if not provided
        if self.n_clusters is None:
            self.n_clusters = self._compute_eigengap(eigenvalues)

        # Perform spectral clustering
        spectral = SpectralClustering(n_clusters=self.n_clusters, affinity=self.error_weighted_affinity, assign_labels='discretize')
        labels = spectral.fit_predict(similarity_matrix)

        # Compute soft cluster probabilities
        cluster_probabilities = np.zeros((X.shape[0], self.n_clusters))
        for i, label in enumerate(labels):
            cluster_probabilities[i, label] = 1.0

        return cluster_probabilities
    
class DPGMM:
    def __init__(self, cluster_config):
        """
        Plain DP-GMM without r_adj or external weighting.

        Parameters:
        - n_components: Maximum number of components
        - weight_concentration_prior: Prior strength (lower → fewer clusters)
        - max_iter: Number of EM steps
        - random_state: Reproducibility
        """
        self.n_components = cluster_config['n_components']
        self.weight_concentration_prior = cluster_config['weight_concentration_prior']
        self.max_iter = cluster_config['max_iter']
        self.random_state = cluster_config['random_state']
        self.model = None
        self.q_probs = None

    def fit(self, z):
        """
        Fit the plain DP-GMM to data.

        Parameters:
        - z: (n_samples, latent_dim), latent representations
        """
        z = np.array(z)
        self.model = BayesianGaussianMixture(
            n_components=self.n_components,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=self.weight_concentration_prior,
            covariance_type='full',
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.model.fit(z)
        self.q_probs = self.model.predict_proba(z)

    def predict_proba(self, z=None):
        """Return soft assignments for new data or training data."""
        if z is None:
            return self.q_probs
        else:
            return self.model.predict_proba(z)

    def predict(self, z=None):
        """Return hard assignments for new data or training data."""
        if z is None:
            return np.argmax(self.q_probs, axis=1)
        else:
            q = self.model.predict_proba(z)
            return np.argmax(q, axis=1)

    def get_model(self):
        """Return the fitted BayesianGaussianMixture model."""
        return self.model