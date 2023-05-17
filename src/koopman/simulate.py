from typing import Callable, List

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


class TimeSeries:
    """Generate data for dynamic mode decomposition."""

    def __init__(self, x0: List[float] = None, ft: List[Callable] = None):
        self.dim = len(ft) if ft else 2
        self.state = np.array(x0) if x0 else np.ones(self.dim)
        self.ft = ft if ft else [self.f1, self.f2]
        self.t = 0
        self.state_history = [self.state]

    def f1(self, t: int) -> float:
        """Function for x_1_t"""
        return 0.9 * np.sin(0.5 * self.state_history[t][0] * t)

    def f2(self, t: int) -> float:
        """Function for x_2_t"""
        return 0.9 * np.sin(0.3 * self.state_history[t][1] * t)

    def forward(self, timesteps: int = 1) -> None:
        """Produce next state."""
        for i in range(timesteps):
            self.state = np.array([f(self.t) for f in self.ft]) + np.random.normal(
                size=self.dim
            )
            self.state_history.append(self.state)
        self.t += timesteps

    def plot(self):
        """Plotting functionality"""
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

    steps = 100
    timeseries.forward(timesteps=100)

    timeseries.plot()
