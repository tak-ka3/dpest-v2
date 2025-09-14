import numpy as np
from dpest.utils.privacy import epsilon_from_samples_matrix


def test_epsilon_from_samples_matrix_discrete():
    P = np.array([[0, 0]] * 25 + [[0, 1]] * 25 + [[1, 0]] * 25 + [[1, 1]] * 25)
    Q = np.array([[0, 0]] * 40 + [[0, 1]] * 30 + [[1, 0]] * 20 + [[1, 1]] * 10)
    eps = epsilon_from_samples_matrix(P, Q)
    assert np.isclose(eps, np.log(2.5), atol=1e-6)
