import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, tol=1e-4):
        self.learning_rate = learning_rate  # Learning rate for gradient descent
        self.n_iterations = n_iterations    # Maximum iterations
        self.tol = tol                      # Convergence threshold
        self.weights_ = None                # Weight vector
        self.bias_ = None                   # Bias term
        self.loss_history_ = []             # Track loss for monitoring

    def _sigmoid(self, z):
        """Compute sigmoid function."""
        return 1 / (1 + np.exp(-z.clip(min=-500, max=500)))  # Clip to avoid overflow

    def fit(self, X, y):
        """Fit logistic regression using gradient descent."""
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights_ = np.random.randn(n_features) * 0.01
        self.bias_ = 0.0

        for _ in range(self.n_iterations):
            # Forward pass
            y_pred = self._sigmoid(X @ self.weights_ + self.bias_)

            # Compute loss (binary cross-entropy)
            loss = -np.mean(y * np.log(y_pred + 1e-10) + (1 - y) * np.log(1 - y_pred + 1e-10))
            self.loss_history_.append(loss)

            # Compute gradients
            dw = (1 / n_samples) * (X.T @ (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights_ -= self.learning_rate * dw
            self.bias_ -= self.learning_rate * db

            # Check convergence
            if len(self.loss_history_) > 1 and abs(self.loss_history_[-1] - self.loss_history_[-2]) < self.tol:
                break

        return self

    def predict_proba(self, X):
        """Predict probability of class 1."""
        return self._sigmoid(X @ self.weights_ + self.bias_)

    def predict(self, X):
        """Predict class labels (0 or 1)."""
        return (self.predict_proba(X) >= 0.5).astype(int)

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

    # Train Logistic Regression
    model = LogisticRegression(learning_rate=0.1, n_iterations=1000, tol=1e-4)
    model.fit(X, y)

    # Predict
    y_pred = model.predict(X)
    print("Learned weights:", model.weights_)
    print("Learned bias:", model.bias_)
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
    plt.title("Logistic Regression Classification")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

# Complexity Analysis:
# Let n = number of samples, d = number of features, I = number of iterations.
# - Training (fit):
#   - Initialization: O(d) for weights, O(1) for bias.
#   - Per iteration:
#     - Predict (sigmoid): O(nd) for matrix-vector multiplication (X @ w) + O(n) for sigmoid.
#     - Loss (binary cross-entropy): O(n) for computing log terms.
#     - Gradients:
#       - dw: O(nd) for X.T @ (y_pred - y).
#       - db: O(n) for sum.
#     - Update: O(d) for weights, O(1) for bias.
#     Total per iteration: O(nd + n + n + nd + n + d) ≈ O(nd).
#   - Total for I iterations: O(Ind).
# Overall Time Complexity:
#   - Training: O(Ind + d) ≈ O(Ind).
#   - Prediction: O(nd) for matrix-vector multiplication and sigmoid.
# Space Complexity:
#   - Input data: O(nd) for X, O(n) for y.
#   - Weights: O(d).
#   - Bias: O(1).
#   - Predictions/loss (temporary): O(n).
#   - Loss history: O(I).
#   Total: O(nd + n + d + I) ≈ O(nd + I).