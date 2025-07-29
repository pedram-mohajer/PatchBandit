# bandit.py
import random
import numpy as np
import torch


# ------------------------------------------------------------------
# Simple ε-greedy grid bandit (legacy baseline)
# ------------------------------------------------------------------
class EpsGridBandit:
    def __init__(self, epsilon=0.2, rng=None):
        self.epsilon = epsilon
        self.rng = np.random.default_rng() if rng is None else rng
        self.counts = {}
        self.rewards = {}


    def select_arm(self, valid_cells):
        """
        ε-greedy selection with proper exploration / exploitation.
        - All cells are tuples (hash-safe).
        - Exploit phase uses *all* visited cells, not only unexplored ones.
        """
        valid_cells = [tuple(c) for c in valid_cells]
        if not valid_cells:
            return (0, 0)

        unexplored = [c for c in valid_cells if c not in self.counts]
        explore_pool = unexplored if unexplored else valid_cells

        # ε-exploration
        if self.rng.random() < self.epsilon:
            return self.rng.choice(explore_pool)

        # exploitation: pick cell with highest average reward
        avg_reward = {
            c: self.rewards.get(c, 0.0) / max(1, self.counts.get(c, 1))
            for c in valid_cells
        }
        return max(avg_reward.items(), key=lambda kv: kv[1])[0]
    

    def update(self, cell, reward):
        """
        Log reward; make sure the key is always a tuple.
        """
        cell = tuple(cell)
        self.counts[cell]  = self.counts.get(cell, 0) + 1
        self.rewards[cell] = self.rewards.get(cell, 0.0) + reward


    # def update(self, cell, reward):
    #     #self.counts[cell] = self.counts.get(cell, 0) + 1
    #     #self.rewards[cell] = self.rewards.get(cell, 0.0) + reward
    #     cell = tuple(cell)          # make hash-safe
    #     self.counts[cell]  = self.counts.get(cell, 0) + 1
    #     self.rewards[cell] = self.rewards.get(cell, 0.0) + reward


# ------------------------------------------------------------------
# === NEW === Contextual Linear-UCB Bandit
#   - Maintains linear model r = w^T x + noise
#   - Chooses arm maximising μ + α*σ (upper confidence bound)
#   - Works with low-dim numeric features per cell.
# ------------------------------------------------------------------
class ContextualUCBPlacementBandit:
    def __init__(self, feat_dim, alpha=1.0, min_pulls=1, seed=None):
        self.feat_dim = feat_dim
        self.alpha = alpha
        self.min_pulls = min_pulls
        self.rng = np.random.default_rng(seed)

        # A = X^T X + λI  (start as λI)
        self.A = np.eye(feat_dim, dtype=np.float32) * 1e-3
        # b = X^T r
        self.b = np.zeros((feat_dim,), dtype=np.float32)

        self.history = {}  # cell -> count

    def _theta(self):
        # Solve A^{-1} b
        return np.linalg.solve(self.A, self.b)

    def select_arm(self, valid_cells, feat_fn):
        """
        feat_fn(cell) -> np.ndarray[D]
        """
        if not valid_cells:
            return (0, 0)

        theta = self._theta()
        A_inv = np.linalg.inv(self.A)

        best_cell, best_ucb = None, -1e9
        for c in valid_cells:
            x = feat_fn(c).astype(np.float32)
            # enforce min exploration
            n = self.history.get(c, 0)
            if n < self.min_pulls:
                return c
            mu = float(np.dot(theta, x))
            sigma = float(np.sqrt(x @ A_inv @ x))
            ucb = mu + self.alpha * sigma
            if ucb > best_ucb:
                best_ucb = ucb
                best_cell = c
        return best_cell if best_cell is not None else self.rng.choice(valid_cells)

    def update(self, cell, reward, feat_vec):
        cell = tuple(cell)           # ensure hash-safe key
        x = feat_vec.astype(np.float32)
        x_col = x[:, None]
        self.A += x_col @ x_col.T
        self.b += reward * x
        self.history[cell] = self.history.get(cell, 0) + 1
