import numpy as np

class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k  # Number of neighbors
        self.X_train = None  # Training features
        self.y_train = None  # Training labels

    def fit(self, X, y):
        """Store training data."""
        self.X_train = X.copy()
        self.y_train = y.copy()
        return self

    def _euclidean_distance(self, x1, x2):
        """Compute Euclidean distance between two points."""
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        """Predict class labels for test points."""
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        """Predict class for a single test point."""
        # Compute distances to all training points
        distances = np.array([self._euclidean_distance(x, x_train) for x_train in self.X_train])
        
        # Find K nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]
        
        # Majority vote
        return np.bincount(k_labels).argmax()

# Example usage
if __name__ == "__main__":
    # Generate synthetic 2D binary classification data
    np.random.seed(42)
    n_samples = 100
    X = np.concatenate([
        np.random.normal(loc=[2, 2], scale=0.5, size=(n_samples // 2, 2)),  # Class 0
        np.random.normal(loc=[-2, -2], scale=0.5, size=(n_samples // 2, 2)),  # Class 1
    ])
    y = np.concatenate([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

    # Train KNN
    model = KNearestNeighbors(k=3)
    model.fit(X, y)

    # Predict
    y_pred = model.predict(X)
    print("Predictions:", y_pred)
    print("Accuracy:", np.mean(y_pred == y))

    # Visualize results
    import matplotlib.pyplot as plt
    # Create mesh grid for decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary and points
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
    plt.title("K-Nearest Neighbors Classification (k=3)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

# Complexity Analysis:
# Let n = number of training samples, m = number of test samples, d = number of features, k = number of neighbors.
# - Training (fit):
#   - Store training data: O(nd) for copying X and O(n) for y.
#   Total: O(nd + n) ≈ O(nd).
# - Prediction:
#   - For each test point (m points):
#     - Compute distances to n training points: O(nd) (each distance is O(d)).
#     - Sort distances to find k nearest: O(n log n) for sorting n distances.
#     - Majority vote among k neighbors: O(k) for counting votes.
#     Total per test point: O(nd + n log n + k).
#   - Total for m test points: O(mnd + mn log n + mk) ≈ O(mnd + mn log n).
# Overall Time Complexity:
#   - Training: O(nd).
#   - Prediction: O(mnd + mn log n).
# Space Complexity:
#   - Training data: O(nd) for X_train, O(n) for y_train.
#   - Distances (temporary): O(n) per test point.
#   - Output labels: O(m).
#   Total: O(nd + n + m) ≈ O(nd + m).
# Note: This is a naive implementation. Optimized versions (e.g., using KD-trees or Ball-trees) can reduce prediction complexity to O(m log n) for low-dimensional data.