import numpy as np
from typing import Tuple

class HiddenMarkovModel:
    """Hidden Markov Model for discrete observation spaces.
    
    Implements the Forward-Backward algorithm and Viterbi decoding.
    
    Attributes:
        A: Transition probability matrix (n_states, n_states).
        B: Emission probability matrix (n_states, n_obs).
        pi: Initial state probability distribution (n_states,).
    """

    def __init__(self, n_states: int, n_obs: int):
        self.n_states = n_states
        self.n_obs = n_obs
        
        # Initialize uniformly
        self.A = np.ones((n_states, n_states)) / n_states
        self.B = np.ones((n_states, n_obs)) / n_obs
        self.pi = np.ones(n_states) / n_states

    def forward(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Runs the Forward algorithm to compute alpha probabilities."""
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        c = np.zeros(T)  # Scaling factors to prevent underflow
        
        # Initialization
        alpha[0] = self.pi * self.B[:, observations[0]]
        c[0] = 1.0 / np.sum(alpha[0])
        alpha[0] *= c[0]
        
        # Recursion
        for t in range(1, T):
            alpha[t] = alpha[t-1].dot(self.A) * self.B[:, observations[t]]
            c[t] = 1.0 / np.sum(alpha[t])
            alpha[t] *= c[t]
            
        return alpha, c

    def backward(self, observations: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Runs the Backward algorithm to compute beta probabilities."""
        T = len(observations)
        beta = np.zeros((T, self.n_states))
        
        # Initialization
        beta[-1] = 1.0 * c[-1]
        
        # Recursion
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A.dot(self.B[:, observations[t+1]] * beta[t+1])) * c[t]
            
        return beta

    def fit(self, observations: np.ndarray, n_iters: int = 100, tol: float = 1e-4) -> None:
        """Trains the HMM using the Baum-Welch (EM) algorithm."""
        T = len(observations)
        
        for iteration in range(n_iters):
            alpha, c = self.forward(observations)
            beta = self.backward(observations, c)
            
            # E-step
            gamma = alpha * beta / np.expand_dims(c, axis=1)
            gamma /= np.sum(gamma, axis=1, keepdims=True)
            
            xi = np.zeros((T - 1, self.n_states, self.n_states))
            for t in range(T - 1):
                numerator = np.outer(alpha[t], self.B[:, observations[t+1]] * beta[t+1]) * self.A
                xi[t] = numerator / np.sum(numerator)
                
            # M-step
            pi_new = gamma[0]
            A_new = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0)[:, np.newaxis]
            
            B_new = np.zeros_like(self.B)
            for k in range(self.n_obs):
                mask = observations == k
                B_new[:, k] = np.sum(gamma[mask], axis=0) / np.sum(gamma, axis=0)
                
            # Check convergence
            if np.linalg.norm(self.A - A_new) < tol:
                break
                
            self.pi, self.A, self.B = pi_new, A_new, B_new

    def viterbi(self, observations: np.ndarray) -> np.ndarray:
        """Finds the most likely sequence of hidden states."""
        T = len(observations)
        viterbi = np.zeros((T, self.n_states))
        backpointer = np.zeros((T, self.n_states), dtype=int)
        
        # Log-space for numerical stability
        log_A = np.log(self.A + 1e-10)
        log_B = np.log(self.B + 1e-10)
        log_pi = np.log(self.pi + 1e-10)
        
        viterbi[0] = log_pi + log_B[:, observations[0]]
        
        for t in range(1, T):
            for s in range(self.n_states):
                prob = viterbi[t-1] + log_A[:, s] + log_B[s, observations[t]]
                viterbi[t, s] = np.max(prob)
                backpointer[t, s] = np.argmax(prob)
                
        # Backtrack
        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(viterbi[-1])
        for t in range(T - 2, -1, -1):
            path[t] = backpointer[t+1, path[t+1]]
            
        return path
