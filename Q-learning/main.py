import numpy as np

class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.actions = ['up', 'down', 'left', 'right']  # Possible actions
        self.goal = (4, 4)                              # Goal state
        self.obstacle = (2, 2)                          # Obstacle state
        self.start = (0, 0)                             # Start state

    def get_next_state(self, state, action):
        """Compute next state given current state and action."""
        row, col = state
        if action == 'up':
            row = max(row - 1, 0)
        elif action == 'down':
            row = min(row + 1, self.size - 1)
        elif action == 'left':
            col = max(col - 1, 0)
        elif action == 'right':
            col = min(col + 1, self.size - 1)
        return (row, col)

    def get_reward(self, state):
        """Return reward for reaching a state."""
        if state == self.goal:
            return 1
        if state == self.obstacle:
            return -1
        return 0

    def is_terminal(self, state):
        """Check if state is terminal (goal or obstacle)."""
        return state == self.goal or state == self.obstacle

class QLearning:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=1000):
        self.env = env
        self.alpha = alpha          # Learning rate
        self.gamma = gamma          # Discount factor
        self.epsilon = epsilon      # Exploration rate
        self.episodes = episodes    # Number of training episodes
        self.q_table = {}           # Q-table: {state: {action: q_value}}

        # Initialize Q-table
        for row in range(env.size):
            for col in range(env.size):
                state = (row, col)
                self.q_table[state] = {action: 0.0 for action in env.actions}

    def choose_action(self, state):
        """Choose action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return np.random.choice(self.env.actions)
        q_values = self.q_table[state]
        max_q = max(q_values.values())
        # Select action with max Q-value (break ties randomly)
        return np.random.choice([a for a, q in q_values.items() if q == max_q])

    def fit(self):
        """Train Q-learning agent."""
        for episode in range(self.episodes):
            state = self.env.start
            steps = 0
            max_steps = 100  # Prevent infinite loops
            
            while not self.env.is_terminal(state) and steps < max_steps:
                # Choose action
                action = self.choose_action(state)
                
                # Get next state and reward
                next_state = self.env.get_next_state(state, action)
                reward = self.env.get_reward(next_state)
                
                # Update Q-value
                current_q = self.q_table[state][action]
                max_next_q = max(self.q_table[next_state].values())
                self.q_table[state][action] = current_q + self.alpha * (
                    reward + self.gamma * max_next_q - current_q
                )
                
                state = next_state
                steps += 1
            
            # Decay epsilon
            self.epsilon = max(0.01, self.epsilon * 0.995)
            
            if episode % 100 == 0:
                print(f"Episode {episode}, Epsilon: {self.epsilon:.4f}")
        
        return self

    def predict(self, state):
        """Return optimal action for a given state."""
        q_values = self.q_table[state]
        max_q = max(q_values.values())
        return np.random.choice([a for a, q in q_values.items() if q == max_q])

    def evaluate_policy(self):
        """Simulate policy from start state and return path."""
        path = [self.env.start]
        state = self.env.start
        steps = 0
        max_steps = 100
        
        while not self.env.is_terminal(state) and steps < max_steps:
            action = self.predict(state)
            state = self.env.get_next_state(state, action)
            path.append(state)
            steps += 1
        
        return path

# Example usage
if __name__ == "__main__":
    # Initialize environment and Q-learning
    env = GridWorld(size=5)
    model = QLearning(env, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=1000)
    
    # Train model
    model.fit()
    
    # Evaluate policy
    path = model.evaluate_policy()
    print("Learned path:", path)
    
    # Visualize results
    import matplotlib.pyplot as plt
    grid = np.zeros((env.size, env.size))
    grid[env.goal] = 1  # Goal
    grid[env.obstacle] = -1  # Obstacle
    
    # Plot grid
    plt.imshow(grid, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='State (1=Goal, -1=Obstacle)')
    
    # Plot path
    path_rows, path_cols = zip(*path)
    plt.plot(path_cols, path_rows, 'g*-', label='Learned Path')
    plt.scatter([env.start[1]], [env.start[0]], c='blue', s=100, label='Start')
    plt.title("Q-learning: Learned Path in Grid World")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.legend()
    plt.grid(True)
    plt.show()

# Complexity Analysis:
# Let S = number of states, A = number of actions, E = number of episodes, T = max steps per episode.
# - Training (fit):
#   - Initialize Q-table: O(S * A) for S states and A actions.
#   - Per episode (E episodes):
#     - Per step (up to T steps):
#       - Choose action: O(A) for epsilon-greedy (max over A actions).
#       - Get next state: O(1) for deterministic grid world.
#       - Get reward: O(1).
#       - Update Q-value: O(A) for finding max Q(s', a').
#       Total per step: O(A).
#     - Total per episode: O(T * A).
#   - Total for E episodes: O(E * T * A).
#   Total: O(S * A + E * T * A) ≈ O(E * T * A) (assuming E*T dominates initialization).
# - Prediction:
#   - Choose action: O(A) for finding max Q(s, a).
#   - For a path of length P: O(P * A).
# Overall Time Complexity:
#   - Training: O(E * T * A).
#   - Prediction: O(P * A) for a path of P steps.
# Space Complexity:
#   - Q-table: O(S * A).
#   - Environment storage: O(S) for grid world states.
#   - Path (for evaluation): O(P).
#   Total: O(S * A + S + P) ≈ O(S * A).
# Note: For large state spaces, function approximation (e.g., neural networks) can reduce space complexity but increase computational cost.