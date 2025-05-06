import numpy as np

class KMeans:
    def __init__(self, n_clusters, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters  # Number of clusters
        self.max_iter = max_iter      # Maximum iterations
        self.tol = tol                # Convergence threshold
        self.centroids_ = None        # Cluster centroids
        self.labels_ = None           # Cluster labels for each point

    def fit(self, X):
        """Fit K-Means clustering to the data."""
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly
        rng = np.random.default_rng(42)
        random_idx = rng.choice(n_samples, self.n_clusters, replace=False)
        self.centroids_ = X[random_idx].copy()

        for _ in range(self.max_iter):
            # Assignment step: assign points to nearest centroid
            old_centroids = self.centroids_.copy()
            self.labels_ = self._assign_clusters(X)

            # Update step: recompute centroids
            self._update_centroids(X)

            # Check convergence
            if np.all(np.linalg.norm(self.centroids_ - old_centroids, axis=1) < self.tol):
                break

        return self

    def _assign_clusters(self, X):
        """Assign each point to the nearest centroid."""
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = np.sum((X - self.centroids_[k]) ** 2, axis=1)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X):
        """Update centroids as the mean of assigned points."""
        for k in range(self.n_clusters):
            cluster_points = X[self.labels_ == k]
            if len(cluster_points) > 0:
                self.centroids_[k] = np.mean(cluster_points, axis=0)
            # If cluster is empty, keep centroid unchanged (or reinitialize randomly)
            # Here, we keep it unchanged for simplicity

    def predict(self, X):
        """Predict cluster labels for the data."""
        if self.labels_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._assign_clusters(X)

# Example usage
if __name__ == "__main__":
    # Generate synthetic 2D data
    np.random.seed(42)
    n_samples = 300
    X = np.concatenate([
        np.random.normal(loc=[2, 2], scale=0.5, size=(n_samples // 3, 2)),  # Cluster 1
        np.random.normal(loc=[-2, -2], scale=0.5, size=(n_samples // 3, 2)),  # Cluster 2
        np.random.normal(loc=[2, -2], scale=0.5, size=(n_samples // 3, 2)),  # Cluster 3
    ])

    # Apply K-Means
    model = KMeans(n_clusters=3, max_iter=100, tol=1e-4)
    model.fit(X)

    # Get labels
    labels = model.predict(X)
    print("Cluster labels:", labels)
    print("Number of clusters:", len(np.unique(labels)))

    # Visualize results
    import matplotlib.pyplot as plt
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
    plt.scatter(model.centroids_[:, 0], model.centroids_[:, 1], c='red', marker='x', s=200, label='Centroids')
    plt.title("K-Means Clustering")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

# Complexity Analysis:
# Let n = number of samples, d = number of features, K = number of clusters, I = number of iterations.
# - Initialization:
#   - Select K random centroids: O(Kd) for copying K points.
#   Total: O(Kd).
# - Main Loop (per iteration):
#   - Assignment Step:
#     - Compute distances to K centroids for n points: O(nKd) (each distance is O(d)).
#     - Assign clusters: O(nK) for argmin over K distances.
#     Total: O(nKd).
#   - Update Step:
#     - Compute mean for each cluster: O(nd) (sum n points of d dimensions, divided by cluster size).
#     Total: O(nd).
#   - Convergence check: O(Kd) for comparing centroids.
#   Total per iteration: O(nKd + nd + Kd) ≈ O(nKd).
# - Total for I iterations: O(InKd).
# Overall Time Complexity:
#   - Training: O(InKd + Kd) ≈ O(InKd) (dominant term).
#   - Prediction: O(nKd) (compute distances and assign clusters for n points).
# Space Complexity:
#   - Input data: O(nd).
#   - Centroids: O(Kd).
#   - Labels: O(n).
#   - Distance matrix (temporary): O(nK).
#   Total: O(nd + Kd + nK + n) ≈ O(nd + nK).