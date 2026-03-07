import numpy as np
from sklearn.mixture import GaussianMixture
class FuzzyClusterer:
    """
    Implements fuzzy clustering using Gaussian Mixture Model.
    Each document belongs to multiple clusters with probabilities.
    """
    def __init__(self, n_clusters: int = 12):
        self.n_clusters = n_clusters
        self.model = GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            random_state=42
        )
    def fit(self, embeddings: np.ndarray):
        #Train the clustering model.
        self.model.fit(embeddings)
    def cluster_distribution(self, embedding: np.ndarray):
        #Returns probability distribution over clusters.
        probs = self.model.predict_proba([embedding])[0]
        return {
            "cluster_distribution": probs.tolist(),
            "dominant_cluster": int(np.argmax(probs))
        }