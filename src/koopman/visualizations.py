import matplotlib.pyplot as plt
import numpy as np


<<<<<<< HEAD:src/koopman/visualizations.py
def plot_original_vs_dmd_reconstruction(original, A, eigenvalues, eigenvectors):
    original = np.array(original)
    x0 = original[0]
    timesteps = original.shape[0]

    # Initial condition projected into DMD modes
    b = np.linalg.pinv(eigenvectors) @ (A @ x0)
=======
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
>>>>>>> 32bb1b0 (Added more description to docs):koopman/dmd.py

    # Time dynamics
    time = np.arange(timesteps)
    dynamics = np.array([b * np.exp(eigenvalues * t) for t in time])
    reconstruction = (eigenvectors @ dynamics.T).T.real  # shape: (timesteps, features)

    plt.figure()
    for i in range(original.shape[1]):
        plt.plot(original[:, i], label=f"Original dim {i}")
        plt.plot(reconstruction[:, i], "--", label=f"Reconstructed dim {i}")
    plt.title("Original vs Reconstructed State Trajectories (DMD)")
    plt.xlabel("Time step")
    plt.ylabel("State value")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_eigenvalues(eigenvalues):
    plt.figure()
    plt.scatter(np.real(eigenvalues), np.imag(eigenvalues))
    plt.axhline(0, color="gray", linestyle="--")
    plt.axvline(0, color="gray", linestyle="--")
    plt.xlabel("Real part")
    plt.ylabel("Imaginary part")
    plt.title("DMD Eigenvalues (Complex Plane)")
    plt.grid(True)
    plt.show()


def plot_modes(eigenvectors):
    plt.figure()
    for i in range(eigenvectors.shape[1]):
        plt.plot(np.real(eigenvectors[:, i]), label=f"Mode {i}")
    plt.title("DMD Modes")
    plt.xlabel("State Dimension")
    plt.ylabel("Mode Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_dmd_output(
    state_history: np.ndarray,
    koopman_matrix: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> None:
    """
    Visualizes the output of the dynamic mode decomposition (DMD).

    Args:
        state_history: An array of shape (N, D) representing the state history, where N is the number
            of time steps and D is the dimensionality of the state.
        koopman_matrix: The computed Koopman matrix.
        eigenvalues: The eigenvalues of the Koopman matrix.
        eigenvectors: The eigenvectors of the Koopman matrix.

    Returns:
        None (Displays the plots).
    """

    # Extract the real and imaginary parts of the eigenvalues
    real_parts = np.real(eigenvalues)
    imag_parts = np.imag(eigenvalues)

    # Eigenvalues in the complex plane
    plt.figure(figsize=(8, 6))
    plt.scatter(real_parts, imag_parts, c="b", marker="o", label="Eigenvalues")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.title("Eigenvalues in the Complex Plane")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot the original and reconstructed state trajectories
    num_time_steps = len(state_history)
    time_steps = range(num_time_steps) 

    original_states = np.array(state_history)
    reconstructed_states = np.dot(eigenvectors, koopman_matrix)
    print(reconstructed_states)
    print(reconstructed_states.shape)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(time_steps, original_states[:, 0], "b-", label="Dimension 1 (Original)")
    plt.plot(time_steps, original_states[:, 1], "r-", label="Dimension 2 (Original)")
    plt.xlabel("Time Steps")
    plt.ylabel("State Value")
    plt.title("Original State Trajectories")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(
        time_steps,
        reconstructed_states[:, 0],
        "b--",
        label="Dimension 1 (Reconstructed)",
    )
    plt.plot(
        time_steps,
        reconstructed_states[:, 1],
        "r--",
        label="Dimension 2 (Reconstructed)",
    )
    plt.xlabel("Time Steps")
    plt.ylabel("State Value")
    plt.title("Reconstructed State Trajectories")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
