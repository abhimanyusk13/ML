import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, epochs=1000, tol=1e-4):
        self.layer_sizes = layer_sizes      # List of layer sizes (input, hidden, output)
        self.learning_rate = learning_rate  # Learning rate for gradient descent
        self.epochs = epochs                # Number of training epochs
        self.tol = tol                      # Convergence threshold
        self.parameters_ = {}               # Weights and biases
        self.cache_ = {}                    # Intermediate values for backprop
        self.loss_history_ = []             # Track loss

    def _initialize_parameters(self):
        """Initialize weights and biases."""
        np.random.seed(42)
        for l in range(1, len(self.layer_sizes)):
            # He initialization for ReLU
            self.parameters_[f'W{l}'] = np.random.randn(self.layer_sizes[l], self.layer_sizes[l-1]) * np.sqrt(2 / self.layer_sizes[l-1])
            self.parameters_[f'b{l}'] = np.zeros((self.layer_sizes[l], 1))

    def _sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _relu(self, z):
        """ReLU activation function."""
        return np.maximum(0, z)

    def _relu_derivative(self, z):
        """Derivative of ReLU."""
        return (z > 0).astype(float)

    def forward(self, X):
        """Forward propagation."""
        n_samples = X.shape[0]
        self.cache_['A0'] = X.T  # Shape: (d, n)
        
        A = X.T
        for l in range(1, len(self.layer_sizes)):
            W = self.parameters_[f'W{l}']
            b = self.parameters_[f'b{l}']
            
            # Linear transformation
            Z = W @ A + b  # Shape: (layer_size[l], n)
            self.cache_[f'Z{l}'] = Z
            
            # Activation
            if l == len(self.layer_sizes) - 1:  # Output layer
                A = self._sigmoid(Z)
            else:  # Hidden layers
                A = self._relu(Z)
            self.cache_[f'A{l}'] = A
        
        return A.T  # Shape: (n, output_size)

    def compute_loss(self, y_pred, y):
        """Compute binary cross-entropy loss."""
        y = y.reshape(-1, 1)
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def backward(self, X, y, y_pred):
        """Backpropagation."""
        n_samples = X.shape[0]
        y = y.reshape(-1, 1)
        grads = {}
        
        # Output layer gradient
        dA = y_pred - y  # Shape: (n, 1)
        dA = dA.T        # Shape: (1, n)
        
        for l in range(len(self.layer_sizes) - 1, 0, -1):
            A_prev = self.cache_[f'A{l-1}']  # Shape: (layer_size[l-1], n)
            Z = self.cache_[f'Z{l}']
            W = self.parameters_[f'W{l}']
            
            # Gradient of Z
            if l == len(self.layer_sizes) - 1:  # Sigmoid output
                dZ = dA
            else:  # ReLU hidden
                dZ = dA * self._relu_derivative(Z)
            
            # Gradients for W and b
            grads[f'dW{l}'] = (1 / n_samples) * (dZ @ A_prev.T)  # Shape: (layer_size[l], layer_size[l-1])
            grads[f'db{l}'] = (1 / n_samples) * np.sum(dZ, axis=1, keepdims=True)  # Shape: (layer_size[l], 1)
            
            # Gradient for previous layer
            if l > 1:
                dA = (W.T @ dZ)  # Shape: (layer_size[l-1], n)
        
        return grads

    def fit(self, X, y):
        """Train neural network using gradient descent."""
        self._initialize_parameters()
        
        for epoch in range(self.epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y_pred, y)
            self.loss_history_.append(loss)
            
            # Backward pass
            grads = self.backward(X, y, y_pred)
            
            # Update parameters
            for l in range(1, len(self.layer_sizes)):
                self.parameters_[f'W{l}'] -= self.learning_rate * grads[f'dW{l}']
                self.parameters_[f'b{l}'] -= self.learning_rate * grads[f'db{l}']
            
            # Check convergence
            if epoch > 0 and abs(self.loss_history_[-1] - self.loss_history_[-2]) < self.tol:
                break
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
        
        return self

    def predict_proba(self, X):
        """Predict probability of class 1."""
        return self.forward(X)

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

    # Train Neural Network
    model = NeuralNetwork(layer_sizes=[2, 10, 10, 1], learning_rate=0.1, epochs=1000, tol=1e-4)
    model.fit(X, y)

    # Predict
    y_pred = model.predict(X)
    print("Accuracy:", np.mean(y_pred.flatten() == y))

    # Visualize results
    import matplotlib.pyplot as plt
    # Create mesh grid for decision-plant boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary and points
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
    plt.title("Neural Network Classification")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

# Complexity Analysis:
# Let n = number of samples, d = input size, h_l = size of layer l, L = number of layers, E = number of epochs.
# - Training (fit):
#   - Initialization: O(sum(h_l * h_{l-1})) for weights, O(sum(h_l)) for biases, where h_0 = d, h_L = output size.
#   - Per epoch:
#     - Forward Pass:
#       - For each layer l: O(n * h_l * h_{l-1}) for W[l] @ A[l-1], O(n * h_l) for activation.
#       - Total: O(n * sum(h_l * h_{l-1})) for L layers.
#     - Loss: O(n) for binary cross-entropy.
#     - Backward Pass:
#       - Output layer: O(n * h_L) for dA.
#       - Hidden layers: O(n * h_l * h_{l-1}) for dW, O(n * h_l) for db, O(n * h_{l-1} * h_l) for dA.
#       - Total: O(n * sum(h_l * h_{l-1})) for L layers.
#     - Parameter updates: O(sum(h_l * h_{l-1})) for weights and biases.
#     Total per epoch: O(n * sum(h_l * h_{l-1}) + sum(h_l * h_{l-1})).
#   - Total for E epochs: O(E * (n * sum(h_l * h_{l-1}) + sum(h_l * h_{l-1}))).
# Overall Time Complexity:
#   - Training: O(E * n * sum(h_l * h_{l-1})) (assuming n dominates parameter updates).
#   - Prediction: O(n * sum(h_l * h_{l-1})) for forward pass.
# Space Complexity:
#   - Input data: O(nd) for X, O(n) for y.
#   - Parameters: O(sum(h_l * h_{l-1})) for weights, O(sum(h_l)) for biases.
#   - Cache (Z, A): O(n * sum(h_l)) for activations across layers.
#   - Gradients: O(sum(h_l * h_{l-1})) for dW, O(sum(h_l)) for db.
#   - Loss history: O(E).
#   Total: O(nd + n * sum(h_l) + sum(h_l * h_{l-1}) + E) ≈ O(nd + n * sum(h_l) + sum(h_l * h_{l-1})).