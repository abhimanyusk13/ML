import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components  # Number of principal components to keep
        self.components_ = None           # Principal components (V_k)
        self.mean_ = None                 # Mean of each feature
        self.explained_variance_ = None   # Singular values squared
        self.explained_variance_ratio_ = None  # Variance ratios

    def fit(self, X):
        """Fit PCA by computing principal components using SVD."""
        n_samples, n_features = X.shape
        
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute SVD
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Select top k components
        self.components_ = Vt[:self.n_components].T  # Shape: (n_features, n_components)
        self.explained_variance_ = (s ** 2) / (n_samples - 1)  # Variance per component
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(self.explained_variance_)
        
        return self

    def transform(self, X):
        """Project data onto principal components."""
        X_centered = X - self.mean_
        return X_centered @ self.components_

    def fit_transform(self, X):
        """Fit PCA and transform data."""
        self.fit(X)
        return self.transform(X)

# Example usage
if __name__ == "__main__":
    # Generate synthetic 2D data
    np.random.seed(42)
    n_samples = 100
    # Create correlated data
    X = np.random.randn(n_samples, 2)
    X[:, 1] = X[:, 1] * 0.5 + X[:, 0] * 0.5  # Introduce correlation

    # Apply PCA
    model = PCA(n_components=2)
    X_transformed = model.fit_transform(X)

    print("Principal Components:\n", model.components_)
    print("Explained Variance Ratio:", model.explained_variance_ratio_)
    print("First 5 transformed points:\n", X_transformed[:5])

    # Visualize results
    import matplotlib.pyplot as plt
    # Plot original data
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.5)
    # Plot principal components as arrows
    mean = model.mean_
    for i in range(model.n_components):
        vec = model.components_[:, i] * np.sqrt(model.explained_variance_[i]) * 2
        plt.arrow(mean[0], mean[1], vec[0], vec[1], color='red', width=0.05)
    plt.title("Original Data with Principal Components")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)

    # Plot transformed data
    plt.subplot(1, 2, 2)
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c='green', alpha=0.5)
    plt.title("Transformed Data (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Complexity Analysis:
# Let n = number of samples, d = number of features, k = number of components, m = number of test samples.
# - Training (fit):
#   - Center data: O(nd) for computing mean and subtracting.
#   - SVD: O(min(n^2d, nd^2)) for full SVD (using