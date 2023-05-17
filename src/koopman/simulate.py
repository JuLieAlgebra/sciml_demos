from typing import Callable, List

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


class TimeSeries:
    """Generate data for dynamic mode decomposition."""

    def __init__(self, x0: List[float] = None, ft: List[Callable] = None):
        """
        Initialize the TimeSeries object.

        Args:
            x0: The initial state of the system as a list of floats. If not provided, it defaults to a list of ones.
            ft: A list of callable functions representing the system dynamics. Each function represents the evolution
                of a specific state variable. If not provided, it defaults to a 2-dimensional system with predefined
                functions `f1` and `f2`.

        Notes:
            - If `ft` is not provided, the system is assumed to be 2-dimensional, and `f1` and `f2` are used as the
              default system dynamics.
        """
        self.dim = len(ft) if ft else 2
        self.state = np.array(x0) if x0 else np.ones(self.dim)
        self.ft = ft if ft else [self.f1, self.f2]
        self.t = 0
        self.state_history = [self.state]

    def f1(self, x_last: np.array, t: int) -> float:
        """
        Function for x_1(x_1{t-1}, t).
        Describes a recursive dynamic of the previous state and the current timestep.

        Args:
            x_last: The value of the state, x, at time t-1.
            t: The current time step.

        Returns:
            float: The value of x_1 at time t.
        """
        return 0.9 * np.sin(0.5 * x_last[0] * t)

    def f2(self, x_last: np.array, t: int) -> float:
        """
        Function for x_2(x_2{t-1}, t).
        Describes a recursive dynamic of the previous state and the current timestep.

        Args:
            x_last: The value of the state, x, at time t-1.
            t: The current time step.

        Returns:
            float: The value of x_2 at time t.
        """
        return 0.9 * np.sin(0.3 * x_last[1] * t)

    def forward(self, timesteps: int = 1) -> None:
        """
        Generate the next state of the system.

        Args:
            timesteps: The number of time steps to evolve the system. Defaults to 1.

        Notes:
            - The state of the system is updated by applying the system dynamics functions `ft` for the specified
              number of time steps.
            - Gaussian noise is added to each state variable during the update.
        """
        for t in range(self.t, self.t + timesteps):
            # Feed the previous state and the current time to the dynamics functions and add noise.
            self.state = np.array(
                [f(self.state_history[t], t) for f in self.ft]
            ) + np.random.normal(size=self.dim)
            self.state_history.append(self.state)
        self.t += timesteps

    def plot(self):
        """Plot the evolution of the system state over time."""
        history = np.array(self.state_history)
        steps = len(history)

        plt.title(f"{self.dim}-dimensional plot of state evolution")
        if self.dim == 2:
            plt.plot(np.arange(steps), history[:, 0], label="f1(t)")
            plt.plot(np.arange(steps), history[:, 1], label="f2(t)")
            plt.legend()
            plt.xlabel("Time steps")

        elif self.dim == 1:
            plt.plot(np.arange(steps), history.flatten())
            plt.xlabel("Time steps")
        else:
            raise ValueError(f"Expected self.dim to be 1 or 2, received {self.dim}")

        plt.ylabel("f(t)")
        plt.show()


if __name__ == "__main__":
    # Example usage for TimeSeries.
    timeseries = TimeSeries()
    timeseries.forward(timesteps=100)
    timeseries.plot()
