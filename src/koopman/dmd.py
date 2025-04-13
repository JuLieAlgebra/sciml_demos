"""Dynamic mode decomposition demo."""
from typing import Tuple

import numpy as np
import simulate


def dmd(state_history: list, r: int = None) -> Tuple[np.array, np.array, np.array]:
    """
    SVD-based dynamic mode decomposition.

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
    # Example usage for multivariate time series.
    timeseries = simulate.TimeSeries()

    steps = 100
    timeseries.forward(steps)

    A, eigenvalues, eigenvectors = dmd(timeseries.state_history)

    print("Koopman operator:\n", A)
    print("Eigen values:\n", eigenvalues)
    print("Eigen vectors:\n", eigenvectors)
