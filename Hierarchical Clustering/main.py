import numpy as np

class HierarchicalClustering:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters  # Desired number of clusters
        self.labels_ = None           # Cluster labels for each point
        self.merge_history_ = []      # Track merges for dendrogram

    def _euclidean_distance(self, x, y):
        """Compute Euclidean distance between two points."""
        return np.sqrt(np.sum((x - y) ** 2))

    def _average_linkage(self, cluster1, cluster2, X):
        """Compute average linkage distance between two clusters."""
        distances = [
            self._euclidean_distance(X[i], X[j])
            for i in cluster1 for j in cluster2
        ]
        return np.mean(distances) if distances else float('inf')

    def fit(self, X):
        """Fit hierarchical clustering using agglomerative approach."""
        n_samples = X.shape[0]
        
        # Initialize each point as its own cluster
        clusters = [[i] for i in range(n_samples)]
        distances = np.full((n_samples, n_samples), float('inf'))
        
        # Compute initial distance matrix
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                distances[i, j] = distances[j, i] = self._euclidean_distance(X[i], X[j])

        current_clusters = clusters.copy()
        cluster_id = n_samples  # ID for new clusters

        # Merge until desired number of clusters
        while len(current_clusters) > self.n_clusters:
            # Find closest pair of clusters
            min_dist = float('inf')
            merge_i = merge_j = None
            for i in range(len(current_clusters)):
                for j in range(i + 1, len(current_clusters)):
                    dist = self._average_linkage(current_clusters[i], current_clusters[j], X)
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j

            if merge_i is None or merge_j is None:
                break

            # Merge clusters
            new_cluster = current_clusters[merge_i] + current_clusters[merge_j]
            self.merge_history_.append((current_clusters[merge_i], current_clusters[merge_j], min_dist, cluster_id))

            # Update clusters
            current_clusters.append(new_cluster)
            current_clusters.pop(max(merge_i, merge_j))
            current_clusters.pop(min(merge_i, merge_j))

            cluster_id += 1

        # Assign labels based on final clusters
        self.labels_ = np.full(n_samples, -1)
        for cluster_idx, cluster in enumerate(current_clusters):
            for point_idx in cluster:
                self.labels_[point_idx] = cluster_idx

        return self

    def predict(self, X):
        """Return cluster labels (same as fit for hierarchical clustering)."""
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
    ])

    # Apply Hierarchical Clustering
    model = HierarchicalClustering(n_clusters=3)
    model.fit(X)

    # Get labels
    labels = model.predict(X)
    print("Cluster labels:", labels)
    print("Number of clusters:", len(np.unique(labels)))

    # Visualize results
    import matplotlib.pyplot as plt
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
    plt.title("Hierarchical Clustering (Average Linkage)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

# Complexity Analysis:
# Let n = number of samples, d = number of features, k = final number of clusters.
# - Initialization:
#   - Compute initial distance matrix: O(n^2d) for n*(n-1)/2 Euclidean distances, each O(d).
#   - Cluster initialization: O(n).
#   Total: O(n^2d + n).
# - Main Loop:
#   - Number of iterations: O(n - k), as we merge from n to k clusters.
#   - Per iteration:
#     - Find closest pair:
#       - Compute average linkage for all pairs of current clusters.
#       - Initially O(n^2) pairs, each computing O(n^2) point-pair distances (worst case, large clusters).
#       - Average linkage: O(|C_i||C_j|d) for clusters C_i, C_j; worst case O(n^2d) per pair.
#       - Total for all pairs: O(n^4d) in worst case (simplified, as cluster sizes grow).
#     - Update clusters: O(n) for merging and updating lists.
#     - Total per iteration: O(n^4d + n).
#   - Total for n - k iterations: O((n - k)n^4d + (n - k)n) ≈ O(n^5d) (dominant term).
# - Label Assignment: O(n) to assign labels to points.
# Overall Time Complexity:
#   - Training: O(n^5d + n^2d + n) ≈ O(n^5d) (worst case, naive implementation).
#   - Prediction: O(n) (return stored labels).
# Space Complexity:
#   - Distance matrix: O(n^2).
#   - Clusters and merge history: O(n^2) (storing point indices and merge records).
#   - Input data: O(nd).
#   - Labels: O(n).
#   Total: O(n^2 + nd).
# Note: This is a naive implementation. Optimized versions (e.g., using priority queues or nearest-neighbor chains) can reduce time complexity to O(n^2d) or O(n^2 log n).