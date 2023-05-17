"""Dynamic mode decomposition demo."""
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import koopman.simulate


def dmd(
    state_history: List[np.array], r: int = None
) -> Tuple[np.array, np.array, np.array]:
    """
    Implements SVD-based dynamic mode decomposition.

    Assumes state_history[0] is an np.array of the state at time t = 0 and state_history[-1] grabs the the last
    state.
    """
    state_history = np.array(state_history)
    last_state = state_history[:-1].copy()
    state = state_history[1:].copy()

    U, S, V = np.linalg.svd(last_state, full_matrices=False)

    koopman_matrix = (
        U[:, :r].conj().T @ state @ V[:r, :].conj() @ np.diag(np.reciprocal(S[:r]))
    )

    eigenvalues, eigenvectors = np.linalg.eig(koopman_matrix)

    return koopman_matrix, eigenvalues, eigenvectors



if __name__ == "__main__":
    timeseries = koopman.simulate.TimeSeries()

    steps = 100
    timeseries.forward(timesteps=100)
    r = 2

    koopman_matrix, eigenvalues, eigenvectors = dmd(timeseries.state_history, r)
    print("Koopman operator:\n", koopman_matrix)
    print("Eigen values:\n", eigenvalues)
    print("Eigen vectors:\n", eigenvectors)
