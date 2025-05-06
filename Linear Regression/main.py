import numpy as np

class LinearRegressionFromScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.m = 0  # Slope
        self.b = 0  # Intercept
        self.cost_history = []

    def predict(self, X):
        """Compute predictions: y = mx + b"""
        return self.m * X + self.b

    def compute_cost(self, X, y):
        """Compute Mean Squared Error"""
        n = len(X)
        predictions = self.predict(X)
        cost = (1/n) * np.sum((predictions - y) ** 2)
        return cost

    def fit(self, X, y):
        """Train the model using gradient descent"""
        n = len(X)
        
        for _ in range(self.n_iterations):
            # Compute predictions
            predictions = self.predict(X)
            
            # Compute gradients
            dm = -(2/n) * np.sum(X * (y - predictions))
            db = -(2/n) * np.sum(y - predictions)
            
            # Update parameters
            self.m -= self.learning_rate * dm
            self.b -= self.learning_rate * db
            
            # Store cost for monitoring
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)
        
        return self

# Example usage
if __name__ == "__main__":
    # Sample data
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])  # Linear relationship: y = 2x

    # Initialize and train model
    model = LinearRegressionFromScratch(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)

    # Print learned parameters
    print(f"Slope (m): {model.m:.2f}, Intercept (b): {model.b:.2f}")

    # Make predictions
    predictions = model.predict(X)
    print(f"Predictions: {predictions}")

    # Plot cost history (optional)
    import matplotlib.pyplot as plt
    plt.plot(model.cost_history)
    plt.xlabel("Iteration")
    plt.ylabel("Cost (MSE)")
    plt.title("Cost vs. Iteration")
    plt.show()