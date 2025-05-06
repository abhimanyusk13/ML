import numpy as np

class GaussianNaiveBayes:
    def __init__(self):
        self.class_priors_ = None  # P(y)
        self.means_ = None         # Means for each class and feature
        self.variances_ = None     # Variances for each class and feature
        self.classes_ = None       # Unique class labels

    def fit(self, X, y):
        """Fit Gaussian Naive Bayes by computing priors, means, and variances."""
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Initialize storage
        self.class_priors_ = np.zeros(n_classes)
        self.means_ = np.zeros((n_classes, n_features))
        self.variances_ = np.zeros((n_classes, n_features))
        
        # Compute parameters for each class
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_priors_[idx] = len(X_c) / n_samples
            self.means_[idx] = np.mean(X_c, axis=0)
            self.variances_[idx] = np.var(X_c, axis=0) + 1e-10  # Add small constant to avoid zero variance
        
        return self

    def _gaussian_pdf(self, x, mean, var):
        """Compute Gaussian probability density for a feature."""
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))

    def predict_proba(self, X):
        """Predict posterior probabilities for each class."""
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        log_posteriors = np.zeros((n_samples, n_classes))
        
        for idx, c in enumerate(self.classes_):
            # Compute log P(y)
            log_prior = np.log(self.class_priors_[idx])
            
            # Compute log P(x | y) = sum(log P(x_i | y))
            log_likelihood = np.zeros(n_samples)
            for i in range(n_features):
                log_likelihood += np.log(self._gaussian_pdf(X[:, i], self.means_[idx, i], self.variances_[idx, i]) + 1e-10)
            
            log_posteriors[:, idx] = log_prior + log_likelihood
        
        # Normalize to get probabilities
        posteriors = np.exp(log_posteriors - np.max(log_posteriors, axis=1, keepdims=True))
        posteriors /= np.sum(posteriors, axis=1, keepdims=True)
        return posteriors

    def predict(self, X):
        """Predict class labels by selecting the class with highest posterior."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

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

    # Train Gaussian Naive Bayes
    model = GaussianNaiveBayes()
    model.fit(X, y)

    # Predict
    y_pred = model.predict(X)
    print("Class priors:", model.class_priors_)
    print("Means:", model.means_)
    print("Variances:", model.variances_)
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
    plt.title("Gaussian Naive Bayes Classification")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

# Complexity Analysis:
# Let n = number of samples, d = number of features, k = number of classes, m = number of test samples.
# - Training (fit):
#   - Compute unique classes: O(n).
#   - For each class (k classes):
#     - Filter samples: O(n).
#     - Compute prior: O(n).
#     - Compute mean: O(nd/k) (assuming balanced classes).
#     - Compute variance: O(nd/k).
#   - Total for k classes: O(kn + knd/k) = O(kn + nd).
#   Total: O(n + kn + nd) ≈ O(nd + kn).
# - Prediction:
#   - For each test sample (m samples):
#     - For each class (k classes):
#       - Compute log P(x_i | y) for d features: O(d).
#       - Sum log-likelihoods and add prior: O(d).
#     - Total per sample: O(kd).
#     - Normalize probabilities: O(k).
#   - Total for m samples: O(mkd + mk) ≈ O(mkd).
# Overall Time Complexity:
#   - Training: O(nd + kn).
#   - Prediction: O(mkd).
# Space Complexity:
#   - Input data: O(nd) for X, O(n) for y.
#   - Parameters: O(k) for priors, O(kd) for means, O(kd) for variances.
#   - Output probabilities/labels: O(mk) for posteriors, O(m) for labels.
#   Total: O(nd + n + kd + mk + m) ≈ O(nd + kd + mk).