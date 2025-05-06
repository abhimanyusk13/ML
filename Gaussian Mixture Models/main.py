import numpy as np

class GaussianMixtureModel:
    def __init__(self, n_components, max_iter=100, tol=1e-4):
        self.n_components = n_components  # Number of Gaussian components
        self.max_iter = max_iter          # Maximum EM iterations
        self.tol = tol                    # Convergence threshold
        self.means_ = None                # Means of Gaussians
        self.covs_ = None                 # Covariance matrices
        self.weights_ = None              # Mixture weights
        self.log_likelihoods_ = []        # Track log-likelihood

    def _initialize_parameters(self, X):
        """Initialize means, covariances, and weights."""
        n_samples, n_features = X.shape
        # Randomly initialize means from data points
        rng = np.random.default_rng(42)
        random_idx = rng.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[random_idx]
        # Initialize covariances as identity matrices
        self.covs_ = np.array([np.cov(X.T, bias=True) for _ in range(self.n_components)])
        # Initialize equal weights
        self.weights_ = np.full(self.n_components, 1 / self.n_components)

    def _gaussian_pdf(self, X, mean, cov):
        """Compute Gaussian probability density."""
        n_features = X.shape[1]
        diff = X - mean
        inv_cov = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)
        exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
        coef = 1 / np.sqrt((2 * np.pi) ** n_features * det_cov)
        return coef * np.exp(exponent)

    def _e_step(self, X):
        """Compute responsibilities (E-Step)."""
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            responsibilities[:, k] = self.weights_[k] * self._gaussian_pdf(X, self.means_[k], self.covs_[k])

        # Normalize responsibilities
        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True) + 1e-10  # Avoid division by zero
        return responsibilities

    def _m_step(self, X, responsibilities):
        """Update parameters (M-Step)."""
        n_samples = X.shape[0]
        Nk = np.sum(responsibilities, axis=0) + 1e-10  # Avoid division by zero

        # Update weights
        self.weights_ = Nk / n_samples

        # Update means
        self.means_ = np.zeros_like(self.means_)
        for k in range(self.n_components):
            self.means_[k] = np.sum(responsibilities[:, k][:, np.newaxis] * X, axis=0) / Nk[k]

        # Update covariances
        self.covs_ = np.zeros_like(self.covs_)
        for k in range(self.n_components):
            diff = X - self.means_[k]
            self.covs_[k] = (responsibilities[:, k][:, np.newaxis] * diff).T @ diff / Nk[k]
            self.covs_[k] += np.eye(self.covs_[k].shape[0]) * 1e-6  # Regularization

    def _log_likelihood(self, X):
        """Compute log-likelihood of the data."""
        n_samples = X.shape[0]
        likelihood = np.zeros(n_samples)
        for k in range(self.n_components):
            likelihood += self.weights_[k] * self._gaussian_pdf(X, self.means_[k], self.covs_[k])
        return np.sum(np.log(likelihood + 1e-10))

    def fit(self, X):
        """Fit the GMM using EM algorithm."""
        self._initialize_parameters(X)

        for iteration in range(self.max_iter):
            # E-Step
            responsibilities = self._e_step(X)
            # M-Step
            self._m_step(X, responsibilities)
            # Compute log-likelihood
            log_likelihood = self._log_likelihood(X)
            self.log_likelihoods_.append(log_likelihood)

            # Check convergence
            if iteration > 0 and abs(self.log_likelihoods_[-1] - self.log_likelihoods_[-2]) < self.tol:
                break

        return self

    def predict(self, X):
        """Predict cluster labels based on highest responsibility."""
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)

# Example usage
if __name__ == "__main__":
    # Generate synthetic 2D data
    np.random.seed(42)
    n_samples = 300
    X = np.concatenate([
        np.random.multivariate_normal(mean=[2, 2], cov=[[0.5, 0], [0, 0.5]], size=n_samples // 3),
        np.random.multivariate_normal(mean=[-2, -2], cov=[[0.5, 0], [0, 0.5]], size=n_samples // 3),
        np.random.multivariate_normal(mean=[2, -2], cov=[[0.5, 0], [0, 0.5]], size=n_samples // 3)
    ])

    # Apply GMM
    model = GaussianMixtureModel(n_components=3, max_iter=100, tol=1e-4)
    model.fit(X)

    # Predict clusters
    labels = model.predict(X)
    print("Cluster labels:", labels)
    print("Number of clusters:", len(np.unique(labels)))

    # Visualize results
    import matplotlib.pyplot as plt
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
    plt.scatter(model.means_[:, 0], model.means_[:, 1], c='red', marker='x', s=200, label='Means')
    plt.title("Gaussian Mixture Model Clustering")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

# Complexity Analysis:
# Let n = number of samples, d = number of features, K = number of components, I = number of iterations.
# - Initialization:
#   - Means: O(n + Kd) for selecting K random points and copying.
#   - Covariances: O(d^2) per component, total O(Kd^2).
#   - Weights: O(K).
#   Total: O(n + Kd^2).
# - E-Step:
#   - Gaussian PDF: O(d^2) for matrix operations (inverse, determinant) + O(nd) for exponent computation per point/component.
#   - For n points and K components: O(nKd^2).
#   - Normalization: O(nK).
#   Total per iteration: O(nKd^2).
# - M-Step:
#   - Weights: O(nK) for summing responsibilities.
#   - Means: O(nKd) for weighted sum.
#   - Covariances: O(nKd^2) for computing weighted outer products.
#   Total per iteration: O(nKd^2).
# - Log-Likelihood: Similar to E-Step, O(nKd^2).
# - Total per EM iteration: O(nKd^2).
# - For I iterations: O(InKd^2).
# Overall Complexity: O(nKd^2I + Kd^2 + n).
# Space Complexity:
# - Storage for X: O(nd).
# - Means: O(Kd).
# - Covariances: O(Kd^2).
# - Responsibilities: O(nK).
# - Total: O(nd + Kd^2 + nK).