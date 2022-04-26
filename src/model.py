import numpy as np


class HMM:
    def __init__(self, pi, A, B):
        """
        The HMM (defined by its parameters) is denoted as lambda.

        :param pi: prior state distribution
        :param A: the state-transition matrix
        :param B: the symbol-emission probability matrix
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.state = None

    def randomize(self):
        """
        Uniformly randomizes the parameters of the model.

        :return: None
        """
        self.pi = np.random.random(self.pi.shape)
        self.pi /= np.sum(self.pi)

        self.A = np.random.random(self.A.shape)
        self.A /= np.sum(self.A, axis=1, keepdims=True)

        self.B = np.random.random(self.B.shape)
        self.B /= np.sum(self.B, axis=1, keepdims=True)

    def reset_state(self):
        """
        Sets the state back to an undefined value.

        :return: None
        """
        self.state = None

    def get_observations(self, T=1, reset_state=False):
        """
        Produces a random sequence of observations based on the HMM.

        :param T: the number of observations
        :param reset_state: a boolean indicating whether to reset the state
        :return: a list of T observations
        """
        n_states = len(self.pi)
        n_symbols = self.B.shape[1]

        if self.state is None or reset_state:
            self.state = np.random.choice(n_states, p=self.pi)

        observations = np.zeros(T, dtype=int)
        for t in range(T):
            observations[t] = np.random.choice(n_symbols, p=self.B[self.state])
            self.state = np.random.choice(n_states, p=self.A[self.state])

        return observations

    def get_alpha(self, observations):
        """
        Finds the probability distribution of states for a time t given all observations leading up to (and including)
        time t. It creates an array with the distributions for all times. The distributions are found by splitting the
        probability into its conditional and prior at each step and summing over the conditional. For all these
        probabilities, there is an implied conditional on lambda.

        alpha = p(O_1,...,O_t,S_t=k) = p(O_t|S_t=k) * sum_i[p(O_1,...,O_[t-1],S_[t-1]=i) * p(S_t=k|S_[t-1]=i)]

        alpha is the conventional name for this probability. It is also referred to as the "forward pass variable"
        since it is used in the forward pass of other algorithms.

        :param observations: a sequence of observations
        :return: a 2D array representing a sequence of distributions
        """
        observations = observations.astype(int)
        T = len(observations)
        n_states = len(self.pi)

        alpha = np.zeros((T, n_states))
        alpha[0] = self.pi * self.B[:, observations[0]]
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * self.B[:, observations[t]]

        return alpha

    def get_likelihood(self, observations):
        """
        Given the current hidden markov model, evaluates the probability of observing the provided symbols.
        This can be accomplished by either summing the final probabilities of the forward variable, alpha, or taking the
        first probabilities of the backward variable, beta, with the priors of the HMM and the first observation.

        P(O_1,...,O_T) = sum_k[p(O_1,...,O_T,S_t=k)] = sum_k[alpha_T[k]]

        :param observations: a sequence of symbols
        :return: the probability
        """
        observations = observations.astype(int)
        T = len(observations)
        alpha = self.get_alpha(observations)
        return np.sum(alpha[T - 1])

    def viterbi(self, observations):
        """
        Finds the most likely sequence of states to produce the observations using the Viterbi algorithm.

        :param observations: a sequence of observations symbols
        :return: a sequence of states
        """
        observations = observations.astype(int)
        T = len(observations)
        n_states = len(self.pi)

        # fill the viterbi dynamic programming matrix
        v = np.zeros((T, n_states))
        v[0] = self.pi * self.B[:, observations[0]]
        for t in range(1, T):
            v[t] = self.B[:, observations[t]] * np.max(v[t - 1].reshape((-1, 1)) * self.A, axis=0)

        # backtrace to get the optimal sequence of states
        backtrace = np.zeros(T, dtype=int)
        backtrace[T - 1] = np.argmax(v[T - 1])
        for t in range(T - 2, -1, -1):  # iterates backward T-2, T-3, ..., 0
            backtrace[t] = np.argmax(self.A[:, backtrace[t + 1]] * v[t])

        return backtrace

    def get_beta(self, observations):
        """
        Finds the conditional probability of the set of observations following a given state. This is done for all
        times t and results in an array. The probabilities are found by splitting the probability into its conditional
        and prior at each step and summing over the conditional. For all these probabilities, there is an implied
        conditional on lambda.

        beta = p(O_[t+1],...,O_T|S_t=k) = sum_i[p(S_t=k|S_[t+1]=j) * p(O_[t+1]|j) * p(O_[t+1],...,O_T|S_t=i)]

        beta is the conventional name for this probability. It is also referred to as the "backward pass variable"
        since it is used in the backward pass of other algorithms.

        :param observations: a sequence of observations
        :return: a 2D array representing a sequence of conditional probability sets
        """
        observations = observations.astype(int)
        T = len(observations)
        n_states = len(self.pi)

        beta = np.ones((T, n_states))
        for t in range(T - 2, -1, -1):  # iterate backwards from T-2, T-2, ..., 0
            beta[t] = self.A @ (self.B[:, observations[t + 1]] * beta[t + 1])

        return beta

    def get_state_distributions(self, observations):
        """
        Finds the probability distribution of states for a time t given all observations. It creates an array with the
        distributions for all times. The distributions are found by taking the product of the probabilities of the
        forward pass probabilities (alpha) and backward pass probabilities (beta). The resulting probability, is not
        a distribution, so the multiplication is followed by a normalization step. For all these probabilities, there
        is an implied conditional on lambda.

        p(S_t=k|O_1,...,O_T) ~ p(O_1,...,O_t,S_t=k) * p(O_[t+1],...,O_T|S_t=k)

        gamma is the conventional name for this probability.

        :param observations: a sequence of observations
        :return: a 2D array representing a sequence of distributions
        """
        observations = observations.astype(int)
        alpha = self.get_alpha(observations)
        beta = self.get_beta(observations)
        gamma = alpha * beta
        return gamma / np.sum(gamma, axis=1)  # normalizes each distribution in gamma

    def train(self, observations, n_epochs=1, randomize=False):
        """
        Uses the Baum-Welch algorithm to perform EM on the HMM and find a local minima of the parameter space.

        :param observations: a sequence of observations
        :param n_epochs: the number of epochs for which to train
        :param randomize: a boolean indicating if the model should be uniformly randomized before training
        :return: None
        """
        observations = observations.astype(int)
        T = observations.shape[0]
        n_states = len(self.pi)
        uniform_over_states = np.full(n_states, 1 / n_states)

        self.reset_state()
        if randomize:
            self.randomize()

        for _ in range(n_epochs):
            # EXPECTATION STEP
            alpha = self.get_alpha(observations)
            beta = self.get_beta(observations)

            gamma = alpha * beta  # P(S_t|O_1,...,O_T)
            gamma[np.all(gamma == 0.0, axis=1)] = uniform_over_states  # set zero-probability columns to be uniform
            norms = np.sum(gamma, axis=1)
            gamma /= norms.reshape((-1, 1))  # normalize the state distribution at each time step

            xi = np.zeros((T - 1, n_states, n_states))  # P(S_t,S_[t+1]|O_1,...,O_T)
            for i in range(n_states):
                for j in range(n_states):
                    xi[:, i, j] = alpha[:-1, i] * self.A[i, j] * self.B[j, observations[1:]] * beta[1:, j]
            xi /= norms[:-1].reshape((-1, 1, 1))  # normalize the transition distribution at each time step

            # MAXIMIZATION STEP
            self.pi = gamma[0]

            xi_sum = np.sum(xi, axis=0)  # sum over all time steps
            xi_sum[np.all(xi_sum == 0.0, axis=1)] = uniform_over_states  # set zero-probability columns to be uniform
            self.A = xi_sum / np.sum(xi_sum, axis=1).reshape(-1, 1)  # normalize each outgoing transition distribution

            gamma_sum = np.sum(gamma, axis=0)  # sum over all time steps
            for sym in range(self.B.shape[1]):
                # to avoid division by zero, we check the value in gamma_sum using np where
                self.B[:, sym] = np.where(gamma_sum, np.sum(gamma[observations == sym], axis=0) / gamma_sum, 0)

    def get_stationary_distributions(self):
        """
        Computes and returns the stationary distributions of the hmm.
        Note that there may be more than one stationary distribution.

        :return: a 2D numpy array where each row is a stationary distribution
        """
        eig_vals, eig_vecs = np.linalg.eig(self.A.T)
        eig_vals, eig_vecs = eig_vals.real, eig_vecs.real.T
        stationary_distributions = eig_vecs[np.isclose(eig_vals, 1.0)]  # since (A x p)=(1 * p) for a stationary
        return stationary_distributions / np.sum(stationary_distributions, axis=1).reshape((-1, 1))
