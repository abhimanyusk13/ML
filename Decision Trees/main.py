import numpy as np

class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx  # Feature to split on
        self.threshold = threshold      # Threshold for split
        self.left = left               # Left child node
        self.right = right             # Right child node
        self.value = value             # Predicted class (for leaf nodes)

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y, depth=0)
        return self

    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Stopping criteria
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_classes == 1):
            return Node(value=self._majority_class(y))

        # Find best split
        best_feature, best_threshold, best_gini = self._best_split(X, y)

        if best_gini == float('inf'):  # No valid split found
            return Node(value=self._majority_class(y))

        # Split data
        left_idx = X[:, best_feature] <= best_threshold
        right_idx = X[:, best_feature] > best_threshold

        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            return Node(value=self._majority_class(y))

        # Recursively build subtrees
        left = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx], y[right_idx], depth + 1)

        return Node(feature_idx=best_feature, threshold=best_threshold, left=left, right=right)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                # Split data
                left_idx = X[:, feature_idx] <= threshold
                right_idx = X[:, feature_idx] > threshold

                if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                    continue

                # Compute Gini index
                gini = self._weighted_gini(y[left_idx], y[right_idx])
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gini

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def _weighted_gini(self, y_left, y_right):
        n_left, n_right = len(y_left), len(y_right)
        n_total = n_left + n_right
        if n_total == 0:
            return float('inf')
        return (n_left / n_total) * self._gini(y_left) + (n_right / n_total) * self._gini(y_right)

    def _majority_class(self, y):
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

# Example usage
if __name__ == "__main__":
    # Generate synthetic binary classification data
    np.random.seed(42)
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # Labels: {0, 1}

    # Train Decision Tree
    model = DecisionTree(max_depth=3, min_samples_split=2)
    model.fit(X, y)

    # Predict
    y_pred = model.predict(X)
    print("Predictions:", y_pred)
    print("Accuracy:", np.mean(y_pred == y))