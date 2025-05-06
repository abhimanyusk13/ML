import numpy as np

class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx  # Feature to split on
        self.threshold = threshold      # Threshold for split
        self.left = left               # Left child node
        self.right = right             # Right child node
        self.value = value             # Predicted value (for leaf nodes)

class RegressionTree:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y, depth=0)
        return self

    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape

        # Stopping criteria
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            np.var(y) < 1e-10):
            return Node(value=np.mean(y))

        # Find best split
        best_feature, best_threshold, best_mse = self._best_split(X, y)

        if best_mse == float('inf'):  # No valid split
            return Node(value=np.mean(y))

        # Split data
        left_idx = X[:, best_feature] <= best_threshold
        right_idx = X[:, best_feature] > best_threshold

        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            return Node(value=np.mean(y))

        # Recursively build subtrees
        left = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx], y[right_idx], depth + 1)

        return Node(feature_idx=best_feature, threshold=best_threshold, left=left, right=right)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best_mse = float('inf')
        best_feature = None
        best_threshold = None

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_idx = X[:, feature_idx] <= threshold
                right_idx = X[:, feature_idx] > threshold

                if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                    continue

                mse = self._weighted_mse(y[left_idx], y[right_idx])
                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_mse

    def _weighted_mse(self, y_left, y_right):
        n_left, n_right = len(y_left), len(y_right)
        n_total = n_left + n_right
        if n_total == 0:
            return float('inf')
        mse_left = np.var(y_left) * n_left if n_left > 0 else 0
        mse_right = np.var(y_right) * n_right if n_right > 0 else 0
        return (mse_left + mse_right) / n_total

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

class GradientBoostingMachine:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees_ = []
        self.initial_prediction_ = None

    def fit(self, X, y):
        # Initialize with mean prediction
        self.initial_prediction_ = np.mean(y)
        predictions = np.full_like(y, self.initial_prediction_)

        # Iteratively add trees
        for _ in range(self.n_estimators):
            # Compute residuals (negative gradient for MSE)
            residuals = y - predictions

            # Train tree on residuals
            tree = RegressionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)

            # Update predictions
            predictions += self.learning_rate * tree.predict(X)
            self.trees_.append(tree)

        return self

    def predict(self, X):
        # Start with initial prediction
        predictions = np.full(X.shape[0], self.initial_prediction_)

        # Add predictions from each tree
        for tree in self.trees_:
            predictions += self.learning_rate * tree.predict(X)
        return predictions

# Example usage
if __name__ == "__main__":
    # Generate synthetic regression data
    np.random.seed(42)
    X = np.linspace(-5, 5, 100).reshape(-1, 1)
    y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

    # Train GBM
    model = GradientBoostingMachine(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X, y)

    # Predict
    y_pred = model.predict(X)
    print("First 5 predictions:", y_pred[:5])

    # Visualize results
    import matplotlib.pyplot as plt
    plt.scatter(X, y, c='blue', label='Data')
    plt.plot(X, y_pred, c='red', label='GBM Prediction')
    plt.title("Gradient Boosting Machine Regression")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

# Complexity Analysis:
# Let n = number of samples, d = number of features, T = number of trees, D = max depth of trees.
# - Initialization:
#   - Compute initial prediction (mean): O(n).
# - Per Tree (T iterations):
#   - Compute residuals: O(n).
#   - Decision Tree Fit:
#     - For each node (up to 2^D nodes in a tree of depth D):
#       - Test all features and thresholds (assume O(n) unique thresholds per feature).
#       - Splitting: O(ndn) for evaluating splits (n thresholds, d features, O(n) for MSE).
#       - Total for one tree: O(ndn * 2^D) in worst case (simplified, assumes balanced splits).
#     - Building tree: O(ndn * 2^D).
#   - Predict with tree: O(n * D) for n samples traversing depth D.
#   - Update predictions: O(n).
#   Total per tree: O(ndn * 2^D + nD).
# - Total for T trees: O(T * (ndn * 2^D + nD)) = O(Tndn * 2^D + TnD).
# - Prediction:
#   - For n samples, T trees, depth D: O(nTD).
# Overall Time Complexity:
#   - Training: O(Tndn * 2^D + TnD + n) ≈ O(Tn^2d * 2^D) (dominant term).
#   - Prediction: O(nTD).
# Space Complexity:
#   - Storage for X, y: O(nd + n).
#   - Trees: O(T * 2^D) nodes (each node stores feature_idx, threshold, value).
#   - Predictions/residuals: O(n).
#   Total: O(nd + n + T * 2^D).