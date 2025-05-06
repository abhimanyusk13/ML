import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, seq_length=10):
        self.input_size = input_size    # Dimension of input (d)
        self.hidden_size = hidden_size  # Dimension of hidden state (h)
        self.output_size = output_size  # Dimension of output
        self.learning_rate = learning_rate
        self.seq_length = seq_length    # Length of input sequence (T)
        
        # Initialize weights and biases
        self.Wf = np.random.randn(hidden_size, hidden_size + input_size) * 0.1  # Forget gate
        self.Wi = np.random.randn(hidden_size, hidden_size + input_size) * 0.1  # Input gate
        self.Wc = np.random.randn(hidden_size, hidden_size + input_size) * 0.1  # Cell candidate
        self.Wo = np.random.randn(hidden_size, hidden_size + input_size) * 0.1  # Output gate
        self.bf = np.zeros((hidden_size, 1))  # Forget gate bias
        self.bi = np.zeros((hidden_size, 1))  # Input gate bias
        self.bc = np.zeros((hidden_size, 1))  # Cell candidate bias
        self.bo = np.zeros((hidden_size, 1))  # Output gate bias
        self.Wy = np.random.randn(output_size, hidden_size) * 0.1  # Output layer
        self.by = np.zeros((output_size, 1))  # Output bias

    def _sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _tanh(self, x):
        """Tanh activation function."""
        return np.tanh(x)

    def forward(self, X):
        """Forward pass through LSTM for a sequence."""
        T = X.shape[0]  # Sequence length
        h = np.zeros((T + 1, self.hidden_size, 1))  # Hidden states
        c = np.zeros((T + 1, self.hidden_size, 1))  # Cell states
        f = np.zeros((T, self.hidden_size, 1))      # Forget gates
        i = np.zeros((T, self.hidden_size, 1))      # Input gates
        cc = np.zeros((T, self.hidden_size, 1))     # Cell candidates
        o = np.zeros((T, self.hidden_size, 1))      # Output gates
        cache = []  # Store intermediate values for backprop

        for t in range(T):
            # Concatenate previous hidden state and current input
            concat = np.vstack((h[t], X[t].reshape(-1, 1)))
            
            # Forget gate
            f[t] = self._sigmoid(self.Wf @ concat + self.bf)
            # Input gate
            i[t] = self._sigmoid(self.Wi @ concat + self.bi)
            # Cell candidate
            cc[t] = self._tanh(self.Wc @ concat + self.bc)
            # Output gate
            o[t] = self._sigmoid(self.Wo @ concat + self.bo)
            
            # Update cell state
            c[t + 1] = f[t] * c[t] + i[t] * cc[t]
            # Update hidden state
            h[t + 1] = o[t] * self._tanh(c[t + 1])
            
            cache.append((concat, f[t], i[t], cc[t], o[t], c[t]))

        # Output prediction from final hidden state
        y_pred = self.Wy @ h[T] + self.by
        return y_pred, h, c, cache

    def backward(self, X, y, y_pred, h, c, cache):
        """Backpropagation through time."""
        T = X.shape[0]
        dWf, dWi, dWc, dWo = np.zeros_like(self.Wf), np.zeros_like(self.Wi), np.zeros_like(self.Wc), np.zeros_like(self.Wo)
        dbf, dbi, dbc, dbo = np.zeros_like(self.bf), np.zeros_like(self.bi), np.zeros_like(self.bc), np.zeros_like(self.bo)
        dWy, dby = np.zeros_like(self.Wy), np.zeros_like(self.by)
        
        # Gradient of loss w.r.t. output
        dy = (2 / X.shape[0]) * (y_pred - y)  # MSE derivative
        dWy = dy @ h[T].T
        dby = dy
        dh_next = self.Wy.T @ dy
        dc_next = np.zeros_like(c[0])

        for t in reversed(range(T)):
            concat, f, i, cc, o, c_prev = cache[t]
            
            # Gradients w.r.t. hidden and cell states
            dc = dc_next + dh_next * o * (1 - self._tanh(c[t + 1]) ** 2)
            df = dc * c_prev * f * (1 - f)
            di = dc * cc * i * (1 - i)
            dcc = dc * i * (1 - cc ** 2)
            do = dh_next * self._tanh(c[t + 1]) * o * (1 - o)
            
            # Gradients w.r.t. weights and biases
            dWf += df @ concat.T
            dWi += di @ concat.T
            dWc += dcc @ concat.T
            dWo += do @ concat.T
            dbf += df
            dbi += di
            dbc += dcc
            dbo += do
            
            # Gradient w.r.t. previous hidden state
            dconcat = (self.Wf.T @ df + self.Wi.T @ di + self.Wc.T @ dcc + self.Wo.T @ do)
            dh_next = dconcat[:self.hidden_size]
            dc_next = dc * f

        # Clip gradients to avoid exploding gradients
        for grad in [dWf, dWi, dWc, dWo, dbf, dbi, dbc, dbo, dWy, dby]:
            np.clip(grad, -1, 1, out=grad)

        return dWf, dWi, dWc, dWo, dbf, dbi, dbc, dbo, dWy, dby

    def fit(self, X, y, epochs=100):
        """Train LSTM using gradient descent."""
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(X)):
                # Forward pass
                y_pred, h, c, cache = self.forward(X[i])
                
                # Compute loss (MSE)
                loss = np.mean((y_pred - y[i]) ** 2)
                total_loss += loss
                
                # Backward pass
                grads = self.backward(X[i], y[i], y_pred, h, c, cache)
                dWf, dWi, dWc, dWo, dbf, dbi, dbc, dbo, dWy, dby = grads
                
                # Update parameters
                self.Wf -= self.learning_rate * dWf
                self.Wi -= self.learning_rate * dWi
                self.Wc -= self.learning_rate * dWc
                self.Wo -= self.learning_rate * dWo
                self.bf -= self.learning_rate * dbf
                self.bi -= self.learning_rate * dbi
                self.bc -= self.learning_rate * dbc
                self.bo -= self.learning_rate * dbo
                self.Wy -= self.learning_rate * dWy
                self.by -= self.learning_rate * dby
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss / len(X)}")

    def predict(self, X):
        """Predict outputs for sequences."""
        y_preds = []
        for x in X:
            y_pred, _, _, _ = self.forward(x)
            y_preds.append(y_pred.flatten())
        return np.array(y_preds)

# Example usage
if __name__ == "__main__":
    # Generate synthetic time series data (sine wave)
    np.random.seed(42)
    t = np.linspace(0, 20, 1000)
    data = np.sin(t)
    
    # Create sequences of length 10 to predict the next value
    seq_length = 10
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    X = np.array(X).reshape(-1, seq_length, 1)  # Shape: (n_samples, seq_length, input_size)
    y = np.array(y).reshape(-1, 1)              # Shape: (n_samples, output_size)

    # Train LSTM
    model = LSTM(input_size=1, hidden_size=10, output_size=1, learning_rate=0.01, seq_length=seq_length)
    model.fit(X, y, epochs=50)

    # Predict
    y_pred = model.predict(X)
    print("First 5 predictions:", y_pred[:5])

    # Visualize results
    import matplotlib.pyplot as plt
    plt.plot(t[seq_length:], y, label='True', c='blue')
    plt.plot(t[seq_length:], y_pred, label='Predicted', c='red')
    plt.title("LSTM Time Series Prediction (Sine Wave)")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

# Complexity Analysis:
# Let n = number of samples, d = input size, h = hidden size, T = sequence length, E = number of epochs.
# - Training (fit):
#   - Per sample per epoch:
#     - Forward Pass:
#       - Per time step (T steps):
#         - Concatenate and gate computations: O(h(d + h)) for matrix multiplications (e.g., Wf @ concat).
#         - Activations (sigmoid, tanh): O(h).
#         - Cell/hidden state updates: O(h).
#         Total per time step: O(h(d + h)).
#       - Total for T steps: O(Th(d + h)).
#       - Output layer: O(h).
#       Total forward: O(Th(d + h) + h).
#     - Loss (MSE): O(1) for scalar output.
#     - Backward Pass:
#       - Per time step (T steps):
#         - Gradient computations: O(h(d + h)) for matrix operations and derivatives.
#         - Weight gradients: O(h(d + h)) for outer products.
#       - Total for T steps: O(Th(d + h)).
#       - Output layer gradients: O(h).
#       Total backward: O(Th(d + h) + h).
#     - Parameter updates: O(h(d + h)) for weight matrices, O(h) for biases.
#     Total per sample: O(Th(d + h)).
#   - Total for n samples and E epochs: O(EnTh(d + h)).
# Overall Time Complexity:
#   - Training: O(EnTh(d + h)).
#   - Prediction: O(nTh(d + h)) for n samples.
# Space Complexity:
#   - Input data: O(nTd) for X, O(n) for y.
#   - Parameters: O(h(d + h)) for Wf, Wi, Wc, Wo; O(h) for biases; O(h) for Wy, by.
#   - Hidden/cell states and cache: O(Th) per sample.
#   - Gradients: O(h(d + h)) for weight gradients, O(h) for bias gradients.
#   Total: O(nTd + h(d + h) + Th + n) ≈ O(nTd + h(d + h)).