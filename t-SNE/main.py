import numpy as np

def _pairwise_distances(X):
    sum_X = np.sum(np.square(X), axis=1)
    return -2 * np.dot(X, X.T) + sum_X[:, None] + sum_X[None, :]

def _binary_search_perplexity(dist_row, target_perplexity, tol=1e-5, max_iter=50):
    betamin, betamax = -np.inf, np.inf
    beta = 1.0
    for _ in range(max_iter):
        P = np.exp(-dist_row * beta)
        P[dist_row == 0] = 0
        sumP = np.sum(P)
        if sumP == 0:
            break
        H = np.log(sumP) + beta * np.sum(dist_row * P) / sumP
        perplexity = np.exp(H)
        diff = perplexity - target_perplexity
        if abs(diff) < tol:
            break
        if diff > 0:
            betamin = beta
            beta = (beta + betamax) / 2 if betamax != np.inf else beta * 2
        else:
            betamax = beta
            beta = (beta + betamin) / 2 if betamin != -np.inf else beta / 2
    return P / sumP

class TSNE:
    def __init__(self, n_components=2, perplexity=30.0, early_exaggeration=4.0,
                 learning_rate=200.0, n_iter=1000, random_state=None):
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state

    def fit_transform(self, X):
        np.random.seed(self.random_state)
        n_samples = X.shape[0]

        # 1. Compute pairwise distances and P affinities
        dists = _pairwise_distances(X)
        P = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            P[i, :] = _binary_search_perplexity(dists[i, :], self.perplexity)
        P = (P + P.T) / (2 * n_samples)
        P *= self.early_exaggeration

        # 2. Initialize low-dim embedding
        Y = np.random.normal(0, 1e-4, size=(n_samples, self.n_components))
        gains = np.ones_like(Y)
        momentum = 0.5
        y_prev = np.zeros_like(Y)

        for it in range(self.n_iter):
            # 3. Compute low-dim affinities Q
            sum_Y = np.sum(np.square(Y), axis=1)
            num = 1 / (1 + (-2 * np.dot(Y, Y.T) + sum_Y[:, None] + sum_Y[None, :]))
            np.fill_diagonal(num, 0)
            Q = num / np.sum(num)

            # 4. Gradient of KL divergence
            PQ = P - Q
            grads = np.zeros_like(Y)
            for i in range(n_samples):
                diff = (Y[i, :] - Y)  # shape (n_samples, n_components)
                grads[i, :] = 4 * np.sum((PQ[:, i][:, None] * num[:, i][:, None]) * diff, axis=0)

            # 5. Update with momentum and adaptive gains
            gains = (gains + 0.2) * ((grads > 0) != (y_prev > 0)) + \
                    (gains * 0.8) * ((grads > 0) == (y_prev > 0))
            gains[gains < 0.01] = 0.01

            y_step = momentum * y_prev - self.learning_rate * (gains * grads)
            Y += y_step
            y_prev = y_step

            # 6. Stop early exaggeration halfway
            if it == int(self.n_iter / 2):
                P /= self.early_exaggeration
                momentum = 0.8

        return Y

# Complexity analysis:
# Time complexity:
#  - Computing high-dim distances: O(n^2 * d)
#  - Perplexity search: O(n^2 * log(1/ε)) ≈ O(n^2)
#  - Each gradient step: O(n^2 * n_iter)
# Overall: O(n^2 * (d + n_iter))
#
# Space complexity:
#  - Storing pairwise matrices P, Q, distances: O(n^2)
