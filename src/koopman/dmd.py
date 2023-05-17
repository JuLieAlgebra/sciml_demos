"""Dynamic mode decomposition demo."""
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import koopman.simulate

def dmd(
    state_history: List[np.array], r: int = None
) -> Tuple[np.array, np.array, np.array]:
    """
    Implements SVD-based Dynamic Mode Decomposition (DMD).

    Given a state history of the system, this function computes the DMD matrix and its corresponding eigenvalues
    and eigenvectors. The DMD provides a low-dimensional approximation of the system dynamics and can be used for
    forecasting, mode analysis, and control.

    Args:
        state_history: A list of numpy arrays representing the state history of the system.
            Each numpy array represents the (flattened) state at a specific time step. The list should be ordered in time,
            where `state_history[0]` represents the state at time t = 0 and `state_history[-1]` represents the
            state at the final time step.
        r: The rank truncation parameter for the DMD matrix. It determines the number of dominant modes to retain.
            If not specified (None), the full-rank DMD matrix is computed.

    Returns:
        A tuple containing the computed DMD matrix, eigenvalues, and eigenvectors.
        - koopman_matrix: The computed Koopman matrix representing the linear dynamics of the system.
        - eigenvalues: The eigenvalues of the Koopman matrix.
        - eigenvectors: The eigenvectors of the Koopman matrix.

    Notes:
        - The state history should have at least two time steps for DMD computation.
        - The first time step in `state_history` corresponds to the initial condition, and the subsequent time steps
          represent the evolution of the system.
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
