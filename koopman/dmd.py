"""Dynamic mode decomposition demo."""
import numpy as np

def dmd(last_state: np.array, state: np.array):
	"""DMD for two states."""
	data_matrix = np.stack((last_state.flatten(), state.flatten()), axis=-1)
	svd = np.linalg.svd(data_matrix)


