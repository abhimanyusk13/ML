import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, tol=1e-4):
        self.learning_rate = learning_rate  # Learning rate for gradient descent
        self.n_iterations = n_iterations    # Maximum iterations
        self.tol = tol                      # Convergence threshold
        self.weights_ = None                # Weight vector
        self.bias_ = None                   # Bias term
        self.loss_history_ = []             # Track loss for monitoring

    def fit(self, X, y):
        """Fit linear regression using gradient descent."""
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights_ = np.random.randn(n_features) * 0.01
        self.bias_ = 0.0

        for _ in range(self.n_iterations):
            # Forward pass
            y_pred = self.predict(X)

            # Compute loss (MSE)
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history_.append(loss)

            # Compute gradients
            dw = (2 / n_samples) * (X.T @ (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights_ -= self.learning_rate * dw
            self.bias_ -= self.learning_rate * db

            # Check convergence
            if len(self.loss_history_) > 1 and abs(self.loss_history_[-1] - self.loss_history_[-2]) < self.tol:
                break

        return self

    def predict(self, X):
        """Predict target values."""
        return X @ self.weights_ + self.bias_

# Example usage
if __name__ == "__main__":
    # Generate synthetic 1D regression data
    np.random.seed(42)
    n_samples = 100
    X = np.linspace(-5, 5, n_samples).reshape(-1, 1)
    y = 2 * X.ravel() + 1 + np.random.normal(0, 0.5, n_samples)  # y = 2x + 1 + noise

    # Train Linear Regression
    model = LinearRegression(learning_rate=0.01, n_iterations=1000, tol=1e-4)
    model.fit(X, y)

    # Predict
    y_pred = model.predict(X)
    print("Learned weights:", model.weights_)
    print("Learned bias:", model.bias_)
    print("First 5 predictions:", y_pred[:5])

    # Visualize results
    import matplotlib.pyplot as plt
    plt.scatter(X, y, c='blue', label='Data')
    plt.plot(X, y_pred, c='red', label='Linear Regression')
    plt.title("Linear Regression (Gradient Descent)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

# Complexity Analysis:
# Let n = number of samples, d = number of features, I = number of iterations.
# - Training (fit):
#   - Initialization: O(d) for weights, O(1) for bias.
#   - Per iteration:
#     - Predict: O(nd) for matrix-vector multiplication (X @ w).
#     - Loss: O(n) for computing MSE.
#     - Gradients:
#       - dw: O(nd) for X.T @ (y_pred - y).
#       - db: O(n) for sum.
#     - Update: O(d) for weights, O(1) for bias.
#     Total per iteration: O(nd + n + nd + n + d) ≈ O(nd).
#   - Total for I iterations: O(Ind).
# Overall Time Complexity:
#   - Training: O(Ind + d) ≈ O(Ind).
#   - Prediction: O(nd) for matrix-vector multiplication.
# Space Complexity:
#   - Input data: O(nd) for X, O(n) for y.
#   - Weights: O(d).
#   - Bias: O(1).
#   - Predictions/loss (temporary): O(n).
#   - Loss history: O(I).
#   Total: O(nd + n + d + I) ≈ O(nd + I).
# Note: For closed-form solution (normal equation), complexity is O(n^3 + n^2d) due to matrix inversion, but gradient descent is preferred for large n.