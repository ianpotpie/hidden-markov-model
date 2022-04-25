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
        for i, row in enumerate(self.A):
            self.A[i] /= np.sum(row)
        self.B = np.random.random(self.B.shape)
        for i, row in enumerate(self.B):
            self.B[i] /= np.sum(row)

    def reset_state(self):
        """
        Sets the state back to an undefined value.

        :return: None
        """
        self.state = None

    def get_observations(self, n=1, reset_state=True):
        """
        Give a random sequence of observations based on the HMM.

        :param n: the number of observations
        :param reset_state: a boolean indicating whether to reset the state
        :return: a sequence of n observations
        """
        assert isinstance(n, int), "the number of observations must be an integer"
        assert n > 0, "the number of observations must be at least 1"
        n_states = len(self.pi)
        n_symbols = self.B.shape[1]

        if self.state is None or reset_state:
            self.state = np.random.choice(n_states, p=self.pi)

        observation = np.zeros(n)
        for i in range(n):
            observation[i] = np.random.choice(n_symbols, p=self.B[self.state])
            self.state = np.random.choice(n_states, p=self.A[self.state])

        return observation

    def _get_alpha(self, observations):
        """
        Finds the probability distribution of states for a time t given all observations leading up to (and including)
        time t. It creates an array with the distributions for all times. The distributions are found by splitting the
        probability into its conditional and prior at each step and summing over the conditional. For all these
        probabilities, there is an implied conditional on lambda.

        p(O_1,...,O_t,S_t=k) = p(O_t|S_t=k) * sum_i[p(O_1,...,O_[t-1],S_[t-1]=i) * p(S_t=k|S_[t-1]=i)]

        alpha is the conventional name for this probability. It is also referred to as the "forward pass variable"
        since it is used in the forward pass of other algorithms.

        :param observations: a sequence of observations
        :return: a 2D array representing a sequence of distributions
        """
        T = len(observations)
        n_states = len(self.pi)

        alpha = np.zeros((T, n_states))
        alpha[0] = self.pi * self.B[:, observations[0]]
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * self.B[:, observations[t]]  # TODO: check this is correct

        return alpha

    def get_likelihood(self, observations):
        """
        Given the current hidden markov model, evaluates the probability of observing the provided symbols.
        This can be accomplished by either summing the final probabilities of the forward variable, alpha, or taking the
        first probabilities of the backward variable, beta, with the priors of the HMM and the first observation.

        :param observations: a sequence of symbols
        :return: the probability
        """
        T = len(observations)
        alpha = self._get_alpha(observations)  # alpha = p(O_1,...,O_t,S_t)
        return np.sum(alpha[T - 1])  # sum_k[p(O_1,...,O_t,S_t=k)] = p(O_1,...,O_t)

    def viterbi(self, observations):
        """
        Finds the most likely sequence of states to produce the observations using the Viterbi algorithm.

        :param observations: a sequence of observations symbols
        :return: a sequence of states
        """
        T = len(observations)
        n_states = len(self.pi)

        # fill the viterbi dynamic programming matrix
        v = np.zeros((T, n_states))
        v[0] = self.pi * self.B[:, observations[0]]
        for t in range(1, T):
            v[t] = self.B[:, observations[t]] * np.max(v[t - 1] * self.A, axis=1)  # TODO: check this

        # backtrace to get the optimal sequence of states
        opt_states = np.zeros(T)  # the most-likely sequence of states
        opt_states[T - 1] = np.argmax(v[T - 1])
        for i in range(1, T):
            t = T - (i + 1)  # iterates backward T-2, T-3, ..., 0
            opt_states[t] = np.argmax(self.A[:, opt_states[t + 1]] * v[t])

        return opt_states

    def _get_beta(self, observations):
        """
        Finds the conditional probability of the set of observations following a given state. This is done for all
        times t and results in an array. The probabilities are found by splitting the probability into its conditional
        and prior at each step and summing over the conditional. For all these probabilities, there is an implied
        conditional on lambda.

        p(O_[t+1],...,O_T|S_t=k) = sum_i[p(S_t=k|S_[t+1]=j) * p(O_[t+1]|j) * p(O_[t+1],...,O_T|S_t=i) ]

        beta is the conventional name for this probability. It is also referred to as the "backward pass variable"
        since it is used in the backward pass of other algorithms.

        :param observations: a sequence of observations
        :return: a 2D array representing a sequence of conditional probability sets
        """
        T = len(observations)
        n_states = len(self.pi)

        beta = np.ones((T, n_states))  # beta = p(O_[t+1] ,... ,O_T | S_t=k)
        for i in range(1, T):
            t = T - (i + 1)  # iterate backwards from T-2, T-2, ..., 0
            beta[t] = self.A @ (self.B[:, observations[t + 1]] * beta[t + 1])  # TODO: check this is correct

        return beta

    def get_state_probs(self, observations):
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
        alpha = self._get_alpha(observations)
        beta = self._get_beta(observations)
        gamma = alpha * beta
        return gamma / np.sum(gamma, axis=1)  # normalizes each distribution in gamma

    def train(self, observations, n_epochs=1, randomize=False):
        """
        Uses the Baum-Welch algorithm to perform EM on the HMM and find a local minima of the parameter space.

        :param observations: a sequence of observations
        :param n_epochs: the number of epochs for which to train
        :param randomize:
        :return: None
        """
        # n_examples = len(X)
        T = observations.shape[0]

        self.reset_state()
        if randomize:
            self.randomize()

        for _ in range(n_epochs):
            # expectation step
            alpha = self._get_alpha(observations)
            beta = self._get_beta(observations)
            gamma = alpha * beta
            xi = alpha[:-1] * self.A * self.B[:, observations[1:]] * beta[1:]
            norm_constant = np.sum(gamma, axis=1)
            gamma /= norm_constant
            xi /= norm_constant

            # maximization step
            self.pi = gamma[0]

            self.A = np.sum(xi, axis=0)
            self.A /= np.sum(self.A, axis=1)

            for v in range(self.B.shape[1]):
                self.B[:, v] = np.sum(np.where(observations == v, gamma), axis=0)
            self.B /= np.sum(gamma, axis=0)

    def get_stationary(self):
        """
        Computes and returns the stationary distributions of the hmm.
        Note that there may be more than one stationary distribution.

        :return: a list of distributions over the states.
        """
        eig_vals, eig_vecs = np.linalg.eig(self.A.T)
        stationaries = eig_vecs[:, np.isclose(eig_vals, 1)]  # since (A x p)=(1 * p) for a stationary distribution
        for i, vec in enumerate(stationaries):
            stationaries[i] /= np.sum(vec)  # this normalizes the eigenvectors to valid distributions
        return stationaries
