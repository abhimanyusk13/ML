import numpy as np

class DBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None  # Cluster labels for each point

    def fit(self, X):
        """Fit DBSCAN clustering to the data."""
        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, -1)  # Initialize all points as noise (-1)
        cluster_id = 0  # Current cluster label

        for point_idx in range(n_samples):
            if self.labels_[point_idx] != -1:  # Skip visited points
                continue

            # Find neighbors
            neighbors = self._get_neighbors(X, point_idx)

            if len(neighbors) < self.min_samples:  # Not a core point
                continue

            # Start a new cluster
            self.labels_[point_idx] = cluster_id
            self._expand_cluster(X, point_idx, neighbors, cluster_id)
            cluster_id += 1

        return self

    def _get_neighbors(self, X, point_idx):
        """Return indices of points within eps distance of point_idx."""
        distances = np.sqrt(np.sum((X - X[point_idx]) ** 2, axis=1))
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X, point_idx, neighbors, cluster_id):
        """Expand cluster by adding density-reachable points."""
        queue = list(neighbors)

        while queue:
            neighbor_idx = queue.pop(0)
            if self.labels_[neighbor_idx] != -1:  # Skip already processed points
                continue

            self.labels_[neighbor_idx] = cluster_id
            neighbor_neighbors = self._get_neighbors(X, neighbor_idx)

            if len(neighbor_neighbors) >= self.min_samples:  # Core point
                # Add new neighbors to queue (excluding already processed points)
                for nn_idx in neighbor_neighbors:
                    if self.labels_[nn_idx] == -1:
                        queue.append(nn_idx)

    def predict(self, X):
        """Return cluster labels for the data (same as fit for DBSCAN)."""
        if self.labels_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.labels_

# Example usage
if __name__ == "__main__":
    # Generate synthetic 2D data
    np.random.seed(42)
    n_samples = 100
    X = np.concatenate([
        np.random.normal(loc=[2, 2], scale=0.5, size=(n_samples // 3, 2)),  # Cluster 1
        np.random.normal(loc=[-2, -2], scale=0.5, size=(n_samples // 3, 2)),  # Cluster 2
        np.random.normal(loc=[2, -2], scale=0.5, size=(n_samples // 3, 2)),  # Cluster 3
        np.random.uniform(low=-4, high=4, size=(n_samples // 10, 2))  # Noise
    ])

    # Apply DBSCAN
    model = DBSCAN(eps=0.5, min_samples=5)
    model.fit(X)

    # Get labels
    labels = model.predict(X)
    print("Cluster labels:", labels)
    print("Number of clusters:", len(np.unique(labels[labels != -1])))
    print("Number of noise points:", np.sum(labels == -1))

    # Visualize results
    import matplotlib.pyplot as plt
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
    plt.title("DBSCAN Clustering")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()