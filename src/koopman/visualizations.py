import matplotlib.pyplot as plt
import numpy as np


def plot_original_vs_dmd_reconstruction(original, A, eigenvalues, eigenvectors):
    original = np.array(original)
    x0 = original[0]
    timesteps = original.shape[0]

    # Initial condition projected into DMD modes
    b = np.linalg.pinv(eigenvectors) @ (A @ x0)

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
