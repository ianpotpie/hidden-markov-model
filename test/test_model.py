import unittest
import numpy as np
from src.model import HMM


def make_uniform_hmm(n_states, n_symbols):
    """
    Creates an HMM with uniform probability over the prior, transitions, and emissions.

    :param n_states: the number of states in the hmm
    :param n_symbols: the number of symbols in the hmm
    :return: a HMM
    """
    pi = np.full(n_states, 1 / n_states)
    A = np.full((n_states, n_states), 1 / n_states)
    B = np.full((n_states, n_symbols), 1 / n_symbols)
    return HMM(pi, A, B)


def make_cyclic_hmm(n_states, uniform_prior=False):
    """
    Makes an HMM that transitions to the following state with probability 1.
    Each state corresponds to symbol and emits that symbol with probability 1.

    :param n_states: the number of states in the hmm
    :param uniform_prior: a boolean indicating if the prior will be uniform (otherwise it starts at 0)
    :return: a HMM
    """
    pi = np.zeros(n_states)
    if uniform_prior:
        pi += np.ones(n_states) / n_states
    else:
        pi[0] = 1
    A = np.zeros((n_states, n_states))
    for i in range(n_states):
        A[i, (i + 1) % n_states] = 1
    B = np.identity(n_states)
    return HMM(pi, A, B)


class CustomAssertions:
    @staticmethod
    def assertSameShape(array1, array2):
        if not np.all(array1.shape == array2.shape):
            raise AssertionError(f"{array1} and {array2} have different shapes")

    @staticmethod
    def assertArrayEqual(array1, array2):
        if not np.all(array1.shape == array2.shape):
            raise AssertionError(f"{array1} and {array2} have different shapes")
        if not np.all(array1 == array2):
            raise AssertionError(f"{array1} != {array2}")

    @staticmethod
    def assertArrayClose(array1, array2):
        if not np.all(array1.shape == array2.shape):
            raise AssertionError(f"{array1} and {array2} have different shapes")
        if not np.all(np.isclose(array1, array2)):
            raise AssertionError(f"{array1} !~ {array2}")

    @staticmethod
    def assertValidDistribution(array):
        item_sum = np.sum(array)
        if not np.isclose(item_sum, 1.0):
            raise AssertionError(f"{array} is not a valid distribution")


class TestRandomize(unittest.TestCase, CustomAssertions):

    def test01(self):
        """
        A 1-state 1-symbol HMM should always have a prior:

        p = [1]
        """
        hmm = make_uniform_hmm(1, 1)
        hmm.randomize()
        self.assertArrayClose(hmm.pi, np.ones(1))

    def test02(self):
        """
        A 1-state 1-symbol HMM should always have a transition matrix:

        A = [[1]]
        """
        hmm = make_uniform_hmm(1, 1)
        hmm.randomize()
        self.assertArrayClose(hmm.A, np.ones((1, 1)))

    def test03(self):
        """
        A 1-state 1-symbol HMM should always have a symbol emission matrix:

        B = [[1]]
        """
        hmm = make_uniform_hmm(1, 1)
        hmm.randomize()
        self.assertArrayClose(hmm.B, np.ones((1, 1)))

    def test04(self, n_states=2, n_symbols=2):
        """
        Tests that a randomized n-state m-symbol HMM has valid prior distribution.

        :param n_states: the number of states
        :param n_symbols: the number of symbols
        """
        hmm = make_uniform_hmm(n_states, n_symbols)
        hmm.randomize()
        self.assertValidDistribution(hmm.pi)

    def test05(self, n_states=2, n_symbols=2):
        """
        Tests that a randomized n-state m-symbol HMM has valid transition distributions.

        :param n_states: the number of states
        :param n_symbols: the number of symbols
        """
        hmm = make_uniform_hmm(n_states, n_symbols)
        hmm.randomize()
        for transition_probs in hmm.A:
            self.assertValidDistribution(transition_probs)

    def test06(self, n_states=2, n_symbols=2):
        """
        Tests that a randomized n-state m-symbol HMM has valid emission distributions.

        :param n_states: the number of states
        :param n_symbols: the number of symbols
        """
        hmm = make_uniform_hmm(n_states, n_symbols)
        hmm.randomize()
        for emission_probs in hmm.B:
            self.assertValidDistribution(emission_probs)

    def test07(self, max_states=10, max_symbols=10):
        """
        Runs a grid search on the possible number of states and symbols (from 1 to max).

        :param max_states: the maximum number of states to test on
        :param max_symbols: the maximum number of symbols to test on
        """
        for n_states in range(1, max_states):
            for n_symbols in range(1, max_symbols):
                hmm = make_uniform_hmm(n_states, n_symbols)
                hmm.randomize()
                self.assertValidDistribution(hmm.pi)
                for transition_probs in hmm.A:
                    self.assertValidDistribution(transition_probs)
                for emission_probs in hmm.B:
                    self.assertValidDistribution(emission_probs)


class TestGetObservations(unittest.TestCase, CustomAssertions):

    def test01(self, T=10):
        """
        Test the getting T observations from a single-state single-symbol HMM.

        :param T: the number of observations (observations time)
        """
        hmm = make_uniform_hmm(1, 1)
        true_observations = np.zeros(T)
        test_observations = hmm.get_observations(T)
        self.assertArrayEqual(test_observations, true_observations)

    def test02(self, n_states=5, T=10):
        """
        A Cyclic HMM with 5 states should just cycle through the states.

        :param n_states: the number of states in the HMM
        :param T: the number of observations (observations time)
        """
        hmm = make_cyclic_hmm(n_states)
        true_observations = np.arange(T) % n_states
        test_observations = hmm.get_observations(T)
        self.assertArrayEqual(test_observations, true_observations)

    def test03(self, n_states=5, T=10):
        """
        When taking multiple observations, they should continue from one another.

        :param n_states: the number of states in the HMM
        :param T: the number of observations (observation time)
        """
        hmm = make_cyclic_hmm(n_states)
        true_observations1 = np.arange(0, T) % n_states
        test_observations1 = hmm.get_observations(T)
        self.assertArrayEqual(test_observations1, true_observations1)
        true_observations2 = np.arange(T, 2 * T) % n_states
        test_observations2 = hmm.get_observations(T)
        self.assertArrayEqual(test_observations2, true_observations2)

    def test04(self, n_states=5, T=10):
        """
        When reset, a cyclic HMM should start from the first state.

        :param n_states: the number of states in the HMM
        :param T: the number of observations (observation time)
        """
        hmm = make_cyclic_hmm(n_states)
        true_observations = np.arange(T) % n_states
        test_observations1 = hmm.get_observations(T)
        self.assertArrayEqual(test_observations1, true_observations)
        test_observations2 = hmm.get_observations(T, reset_state=True)
        self.assertArrayEqual(test_observations2, true_observations)


class TestGetAlpha(unittest.TestCase, CustomAssertions):

    def test01(self, T=10):
        """
        The HMM with 1 state and 1 symbol should produce probabilities of 1

        :param T: the number of observations (observation time)
        """
        hmm = make_uniform_hmm(1, 1)
        observations = np.zeros(T)
        true_alpha = np.ones((T, 1))
        test_alpha = hmm.get_alpha(observations)
        self.assertArrayClose(test_alpha, true_alpha)

    def test02(self, n_states=2, T=10):
        """
        The HMM with (uniformly visited) n states and 1 symbol should produce probabilities of 1/n in every position.

        :param n_states: the number of states in the HMM
        :param T: the number of observations (observation time)
        """
        hmm = make_uniform_hmm(n_states, 1)
        observations = np.zeros(T)
        true_alpha = np.full((T, n_states), 1 / n_states)
        test_alpha = hmm.get_alpha(observations)
        self.assertArrayClose(test_alpha, true_alpha)

    def test03(self, n_symbols=2, T=10):
        """
        The HMM with 1 state and n (uniformly emitted) symbols should produce probabilities
        that diminish by 1/n every step.

        :param n_symbols: the number of symbols produced by the HMM
        :param T: the number of observations (observation time)
        """
        hmm = make_uniform_hmm(1, n_symbols)
        observations = np.zeros(T)
        true_alpha = (1 / n_symbols) ** np.arange(1, T + 1).reshape((T, 1))
        test_alpha = hmm.get_alpha(observations)
        self.assertArrayClose(test_alpha, true_alpha)

    def test04(self):
        """
        A randomized HMM with 1 symbol and 5 states will have valid distributions at each step.
        """
        # TODO: IMPLEMENT
        pass


class TestGetLikelihood(unittest.TestCase, CustomAssertions):
    # TODO: IMPLEMENT
    pass


class TestViterbi(unittest.TestCase, CustomAssertions):
    # TODO: IMPLEMENT
    pass

    def test01(self, n_symbols=5, T=10):
        """
        A 1-state n-symbol HMM should always iterate over the same state.

        :param n_symbols: the number of symbols emitted by the HMM
        :param T: the number of observations (observation time)
        """
        hmm = make_uniform_hmm(1, n_symbols)
        observations = hmm.get_observations(T)
        true_states = np.zeros(T, dtype=int)
        test_states = hmm.viterbi(observations)
        self.assertArrayEqual(true_states, test_states)

    def test02(self, n_states=5, T=10):
        """
        An n-state cyclic HMM should always iterate over the same state-cycle.

        :param n_states: the number of states in the HMM
        :param T: the number of observations (observation time)
        """
        hmm = make_cyclic_hmm(n_states)
        observations = hmm.get_observations(T)
        true_states = np.arange(T, dtype=int) % n_states
        test_states = hmm.viterbi(observations)
        self.assertArrayEqual(true_states, test_states)

    def test03(self, n_states=5, T=10):
        """
        An n-state cyclic HMM with laplace smoothing should always iterate over the same state-cycle.

        :param n_states: the number of states in the HMM
        :param T: the number of observations (observation time)
        """
        hmm = make_cyclic_hmm(n_states)
        delta = 0.1
        hmm.pi = (hmm.pi + (delta / n_states)) / (1 + delta)
        for i in range(n_states):
            hmm.A[i] = (hmm.A[i] + (delta / n_states)) / (1 + delta)
            hmm.B[i] = (hmm.B[i] + (delta / n_states)) / (1 + delta)
        observations = np.arange(T, dtype=int) % n_states
        true_states = np.arange(T, dtype=int) % n_states
        test_states = hmm.viterbi(observations)
        self.assertArrayEqual(true_states, test_states)


class TestGetBeta(unittest.TestCase, CustomAssertions):
    # TODO: IMPLEMENT
    pass


class TestGetStateDistributions(unittest.TestCase, CustomAssertions):
    # TODO: IMPLEMENT
    pass


class TestTrain(unittest.TestCase, CustomAssertions):

    def test01(self, n_observations=10):
        """
        A 1-state 1-symbol HMM should have an initial distribution of [1] after training.

        :param n_observations: the number of observations on which to train
        """
        hmm = make_uniform_hmm(1, 1)
        observations = np.zeros(n_observations)
        hmm.train(observations)
        self.assertArrayClose(hmm.pi, np.array([1]))

    def test02(self, n_observations=10):
        """
        A 1-state 1-symbol HMM should have a transition distribution of [[1]] after training.

        :param n_observations: the number of observations on which to train
        """
        hmm = make_uniform_hmm(1, 1)
        observations = np.zeros(n_observations)
        hmm.train(observations)
        self.assertArrayClose(hmm.A, np.array([[1]]))

    def test03(self, n_observations=10):
        """
        A 1-state 1-symbol HMM should have an emission distribution of [[1]] after training.

        :param n_observations: the number of observations on which to train
        """
        hmm = make_uniform_hmm(1, 1)
        observations = np.zeros(n_observations)
        hmm.train(observations)
        self.assertArrayClose(hmm.A, np.array([[1]]))

    def test04(self, n_states=5, n_epochs=20, n_observations=10):
        """
        An n-state cyclic HMM with laplace smoothing applied should be trained back to cyclic hmm.
        Note that n_observations > n_states in order to guarantee effective training

        :param n_states: the number of states in the HMM
        :param n_epochs: the number of epochs for which to train
        :param n_observations: the number of observations on which to train
        """
        hmm = make_cyclic_hmm(n_states)
        observations = hmm.get_observations(n_observations)
        blurred_hmm = make_cyclic_hmm(n_states)
        delta = 0.1
        blurred_hmm.pi = (hmm.pi + (delta / n_states)) / (1 + delta)
        for i in range(n_states):
            blurred_hmm.A[i] = (hmm.A[i] + (delta / n_states)) / (1 + delta)
            blurred_hmm.B[i] = (hmm.B[i] + (delta / n_states)) / (1 + delta)
        blurred_hmm.train(observations, n_epochs)
        self.assertArrayClose(hmm.pi, blurred_hmm.pi)
        self.assertArrayClose(hmm.A, blurred_hmm.A)
        self.assertArrayClose(hmm.B, blurred_hmm.B)


class TestGetStationaryDistributions(unittest.TestCase, CustomAssertions):

    def test01(self):
        """
        A 1-state 1-symbol HMM has a single stationary distribution [1]
        """
        hmm = make_uniform_hmm(1, 1)
        true_stationary_dists = np.ones((1, 1))
        test_stationary_dists = hmm.get_stationary_distributions()
        self.assertArrayClose(true_stationary_dists, test_stationary_dists)

    def test02(self, n_states=5):
        """
        An HMM with n states and an identity transition matrix should have n stationary distributions. These should be
        the standard basis.

        :param n_states: the number of states in the HMM
        """
        pi = np.full(n_states, 1 / n_states)
        A = np.identity(n_states)
        B = np.identity(n_states)
        hmm = HMM(pi, A, B)
        true_stationary_dists = np.identity(n_states)
        test_stationary_dists = hmm.get_stationary_distributions()
        self.assertArrayClose(true_stationary_dists, test_stationary_dists)

    def test03(self, n_states=5):
        """
        An HMM with n states and a reversed identity transition matrix should have n stationary distributions. These
        should be the standard basis.

        :param n_states: the number of states in the HMM
        """
        pi = np.full(n_states, 1 / n_states)
        A = np.identity(n_states)[::-1]
        B = np.identity(n_states)
        hmm = HMM(pi, A, B)
        true_stationary_dists = np.zeros((n_states // 2 + 1, n_states))
        for i in range(n_states // 2 + 1):
            true_stationary_dists[i, i] += 0.5
            true_stationary_dists[i, -(i + 1)] += 0.5
        test_stationary_dists = hmm.get_stationary_distributions()
        self.assertArrayClose(true_stationary_dists, test_stationary_dists)

    def test04(self, n_states=5):
        """
        If the HMM transition matrix is uniform, then the stationary is uniform.

        :param n_states: the number of states in the HMM
        """
        hmm = make_uniform_hmm(n_states, n_states)
        true_stationary_dists = np.full((1, n_states), 1 / n_states)
        test_stationary_dists = hmm.get_stationary_distributions()
        self.assertArrayClose(true_stationary_dists, test_stationary_dists)

    def test05(self, n_states=5):
        """
        If the HMM transition matrix is cyclical, then the stationary is uniform (as long as it hits every state).

        :param n_states: the number of states in the HMM
        """
        hmm = make_cyclic_hmm(n_states)
        true_stationary_dists = np.full((1, n_states), 1 / n_states)
        test_stationary_dists = hmm.get_stationary_distributions()
        self.assertArrayClose(true_stationary_dists, test_stationary_dists)

    def test06(self):
        """
        This is a case that we have pre-computed to be true.
        """
        pi = np.ones(3) / 3
        A = np.array([[0.5, 0.5, 0.],
                      [0.25, 0.5, 0.25],
                      [0., 0.5, 0.5]])
        B = np.identity(3)
        hmm = HMM(pi, A, B)
        true_stationary_dists = np.array([[0.25, 0.5, 0.25]])
        test_stationary_dists = hmm.get_stationary_distributions()
        self.assertArrayClose(true_stationary_dists, test_stationary_dists)


if __name__ == '__main__':
    unittest.main()
