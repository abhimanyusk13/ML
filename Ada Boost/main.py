import numpy as np

class DecisionStump:
    def __init__(self):
        self.polarity = 1  # Direction of inequality (1 or -1)
        self.feature_idx = None  # Feature to split on
        self.threshold = None  # Threshold for split
        self.alpha = None  # Weight of the stump in ensemble

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column <= self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
        return predictions

class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.clfs = []  # List to store weak learners

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize weights
        w = np.full(n_samples, 1 / n_samples)
        
        for _ in range(self.n_estimators):
            # Train a decision stump
            stump = DecisionStump()
            min_error = float('inf')
            
            # Find best feature, threshold, and polarity
            for feature_idx in range(n_features):
                X_column = X[:, feature_idx]
                thresholds = np.unique(X_column)
                
                for threshold in thresholds:
                    for polarity in [1, -1]:
                        predictions = np.ones(n_samples)
                        if polarity == 1:
                            predictions[X_column <= threshold] = -1
                        else:
                            predictions[X_column > threshold] = -1
                        
                        error = np.sum(w[predictions != y])
                        
                        if error < min_error:
                            min_error = error
                            stump.polarity = polarity
                            stump.threshold = threshold
                            stump.feature_idx = feature_idx
            
            # Compute stump weight (alpha)
            eps = 1e-10  # Avoid division by zero
            stump.alpha = 0.5 * np.log((1 - min_error + eps) / (min_error + eps))
            
            # Update sample weights
            predictions = stump.predict(X)
            w *= np.exp(-stump.alpha * y * predictions)
            w /= np.sum(w)  # Normalize weights
            
            # Store the stump
            self.clfs.append(stump)
        
        return self

    def predict(self, X):
        # Combine weak learners' predictions
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        return np.sign(y_pred)

# Example usage
if __name__ == "__main__":
    # Generate synthetic binary classification data
    np.random.seed(42)
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]])
    y = np.array([1, 1, 1, 1, -1, -1, -1, -1])  # Labels: {+1, -1}
    
    # Train AdaBoost
    model = AdaBoost(n_estimators=10)
    model.fit(X, y)
    
    # Predict
    y_pred = model.predict(X)
    print("Predictions:", y_pred)
    print("Accuracy:", np.mean(y_pred == y))