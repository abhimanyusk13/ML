import numpy as np

class SVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        # Convert labels to -1 or +1
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Subgradient descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    # Only regularization term contributes
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    # Hinge loss term plus regularization
                    dw = 2 * self.lambda_param * self.w - y_[idx] * x_i
                    db = -y_[idx]
                # Update parameters
                self.w -= self.lr * dw
                self.b -= self.lr * db

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.where(linear_output >= 0, 1, 0)

# Complexity analysis:
# Training time: O(n_iters * n_samples * n_features)
# Prediction time: O(n_samples_test * n_features)
