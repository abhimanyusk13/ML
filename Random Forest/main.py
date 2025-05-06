import numpy as np
from collections import Counter
from math import log2

class DecisionTree:
    def __init__(self, max_depth=None, min_samples=1, n_features=None):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.n_features = n_features
        self.root = None

    class Node:
        def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
            self.feature_index = feature_index
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def fit(self, X, y):
        # determine number of features to consider at each split
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        # stopping criteria
        if (depth >= self.max_depth or
            n_samples <= self.min_samples or
            len(np.unique(y)) == 1):
            leaf_val = self._most_common_label(y)
            return self.Node(value=leaf_val)

        # select random features for split
        feat_idxs = np.random.choice(n_features, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)

        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return self.Node(feature_index=best_feat, threshold=best_thresh, left=left, right=right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for idx in feat_idxs:
            thresholds = np.unique(X[:, idx])
            for thr in thresholds:
                gain = self._information_gain(y, X[:, idx], thr)
                if gain > best_gain:
                    best_gain, split_idx, split_thresh = gain, idx, thr
        return split_idx, split_thresh

    def _information_gain(self, y, col, thr):
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(col, thr)
        if len(left_idxs)==0 or len(right_idxs)==0:
            return 0
        n = len(y)
        e_l = self._entropy(y[left_idxs])
        e_r = self._entropy(y[right_idxs])
        child_entropy = (len(left_idxs)/n)*e_l + (len(right_idxs)/n)*e_r
        return parent_entropy - child_entropy

    def _split(self, col, thr):
        left = np.argwhere(col <= thr).flatten()
        right = np.argwhere(col > thr).flatten()
        return left, right

    def _entropy(self, y):
        counts = np.bincount(y)
        ps = counts / len(y)
        return -np.sum([p * log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        branch = node.left if x[node.feature_index] <= node.threshold else node.right
        return self._traverse_tree(x, branch)

class RandomForest:
    def __init__(self, n_trees=100, max_depth=None, min_samples=1, sample_size=1.0, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.sample_size = sample_size
        self.n_features = n_features
        self.trees = []

    def _subsample(self, X, y):
        n_samples = X.shape[0]
        sample_n = int(n_samples * self.sample_size)
        idxs = np.random.choice(n_samples, sample_n, replace=True)
        return X[idxs], y[idxs]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples=self.min_samples,
                n_features=self.n_features
            )
            X_samp, y_samp = self._subsample(X, y)
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)

    def predict(self, X):
        # collect predictions from each tree
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)  # shape: (n_samples, n_trees)
        # majority vote
        return np.array([Counter(row).most_common(1)[0][0] for row in tree_preds])

# Complexity analysis:
# Training complexity: O(n_trees * m * n * log(n))
#  - n_trees: number of trees
#  - m: number of features considered at each split
#  - n: number of training samples
#  - log(n): average depth of each tree 
# Prediction complexity: O(n_trees * n)
#  - n: number of samples to predict
#  - n_trees: number of trees
