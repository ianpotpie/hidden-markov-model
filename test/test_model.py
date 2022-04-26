import unittest
import numpy as np
from model import HMM


class NumpyAssertions:
    def assertArrayEqual(self, array1, array2):
        if not np.all(array1 == array2):
            raise AssertionError(f"{array1} != {array2}")

    def assertArrayClose(self, array1, array2):
        if not np.all(np.isclose(array1, array2)):
            raise AssertionError(f"{array1} !~ {array2}")


class TestRandomize(unittest.TestCase):

    def

class Test3CycleHMM(unittest.TestCase):

    def setUp(self) -> None:
        pi = np.array([1.0, 0.0, 0.0])
        A = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        B = np.identity(3)
        self.hmm = HMM(pi, A, B)

    def test_init(self):
        self.assertTrue((self.hmm.pi == np.array([1.0, 0.0, 0.0])).all())
        self.assertTrue((self.hmm.A == np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])).all())
        self.assertTrue((self.hmm.B == np.identity(3)).all())

    def test_randomize(self):
        self.hmm.randomize()
        self.assertFalse((self.hmm.pi == np.array([1.0, 0.0, 0.0])).any())
        self.assertFalse((self.hmm.A == np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])).any())
        self.assertFalse((self.hmm.B == np.identity(3)).any())

    def test_get_5_observations(self):
        self.assertTrue((self.hmm.get_observations(5) == np.array([0, 1, 2, 0, 1])).all())

    def test_get_alpha_makes_distributions(self):
        observations = self.hmm.get_observations(5)
        alpha = self.hmm._get_alpha(observations)

    def test_3cycle(self):
        pass


class TestStationaryDistributions(unittest.TestCase, NumpyAssertions):

    def test_1x1_identity(self):
        pi = np.ones(1)
        A = np.array([[1]])
        B = np.identity(1)
        hmm = HMM(pi, A, B)
        true_stationaries = np.array([[1]])
        test_stationaries = hmm.get_stationaries()
        self.assertArrayClose(test_stationaries, true_stationaries)

    def test_2x2_identity(self):
        pi = np.ones(2) / 2
        A = np.array([[1, 0],
                      [0, 1]])
        B = np.identity(2)
        hmm = HMM(pi, A, B)
        true_stationaries = np.array([[0, 1], [1, 0]])
        test_stationaries = hmm.get_stationaries()
        self.assertArrayClose(test_stationaries, true_stationaries)

    def test_2x2_identity_transpose(self):
        pi = np.ones(2) / 2
        A = np.array([[0, 1],
                      [1, 0]])
        B = np.identity(2)
        hmm = HMM(pi, A, B)
        true_stationaries = np.array([[0.5, 0.5]])
        test_stationaries = hmm.get_stationaries()
        self.assertArrayClose(test_stationaries, true_stationaries)

    def test_2x2_uniform(self):
        pi = np.ones(2) / 2
        A = np.array([[0.5, 0.5],
                      [0.5, 0.5]])
        B = np.identity(2)
        hmm = HMM(pi, A, B)
        true_stationaries = np.array([[0.5, 0.5]])
        test_stationaries = hmm.get_stationaries()
        self.assertArrayClose(test_stationaries, true_stationaries)

    def test_3x3_identity(self):
        pi = np.ones(3) / 3
        A = np.array([[1., 0., 0.],
                      [0., 1., 0.],
                      [0., 0., 1.]])
        B = np.identity(3)
        hmm = HMM(pi, A, B)
        true_stationaries = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        test_stationaries = hmm.get_stationaries()
        self.assertArrayClose(test_stationaries, true_stationaries)

    def test_3x3_1(self):
        pi = np.ones(3) / 3
        A = np.array([[0.5, 0.5, 0.],
                      [0.25, 0.5, 0.25],
                      [0., 0.5, 0.5]])
        B = np.identity(3)
        hmm = HMM(pi, A, B)
        true_stationaries = np.array([[0.25, 0.5, 0.25]])
        test_stationaries = hmm.get_stationaries()
        self.assertArrayClose(test_stationaries, true_stationaries)


if __name__ == '__main__':
    unittest.main()
