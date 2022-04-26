from model import HMM
import numpy as np


def main():
    pi = np.array([1.0, 0.0, 0.0])
    A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    B = np.identity(3)
    hmm = HMM(pi, A, B)
    observations = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    hmm.train(observations, n_epochs=10, randomize=False)
    print(hmm.get_stationary())


if __name__ == "__main__":
    main()
