import numpy as np


class HMM:
    def __init__(self, pi=None, A=None, B=None):
        self.pi = pi  # prior state distribution
        self.A = A  # transition matrix
        self.B = B  # emission matrix
        self.state = None  # the hidden state of the model

    def get_observations(self, n=1):
        assert isinstance(n, int), "the number of observations must be an integer"
        assert n > 0, "the number of observations must be at least 1"
        n_states = len(self.pi)
        n_symbols = self.B.shape[1]

        observation = np.zeros(n)

        if self.state is None:
            self.state = np.random.choice(n_states, p=self.pi)

        for i in range(n):
            observation[i] = np.random.choice(n_symbols, p=self.B[self.state])
            self.state = np.random.choice(n_states, p=self.A[self.state])

        return observation

    def evaluate_observations(self, observations):
        """
        Given the current hidden markov model, evaluates the probability of observing the provided symbols.

        :param observations: a sequence of symbols
        :return: the probability of the observations
        """
        obs_prob = 1
        state_probs = self.pi
        for observation in observations:
            obs_prob *= np.dot(state_probs, self.B[:, observation])
            state_probs = state_probs @ self.A

        return obs_prob

    def decode_observations(self, observations):
        pass

    def train(self, X):
        pass
