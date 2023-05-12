from typing import List, Callable

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class TimeSeries:
    """Generate data for dynamic mode decomposition."""
    def __init__(self, x0: List[float]=None, ft: List[Callable]=None):
        self.dim = len(ft) if ft else 2
        self.state = np.array(x0) if x0 else np.ones(self.dim)
        self.ft = ft if ft else [self.f1, self.f2]
        self.t = 0
        self.state_history = [self.state]

    def f1(self, t: np.array) -> np.array:
        """ """
        return 0.9*np.sin(0.5*t)

    def f2(self, t: np.array) -> np.array:
        """ """
        return 0.9*np.sin(0.3*t)

    def forward(self) -> None:
        """Produce next state."""
        self.state = np.array([f(self.t) for f in self.ft]) + np.random.normal(size=self.dim)
        self.state_history.append(self.state)
        self.t +=1

    def plot(self):
        """Plotting functionality"""
        history = np.array(self.state_history)
        steps = len(history)

        plt.title(f"{self.dim}-dimensional plot of state evolution")
        if self.dim == 2:
            # plt.plot(history[:, 0], history[:, 1])
            plt.plot(np.arange(steps),history[:, 0], label='f1(t)')
            plt.plot(np.arange(steps), history[:, 1], label='f2(t)')
            plt.legend()
            plt.xlabel("Time steps")

        elif self.dim == 1:
            plt.plot(np.arange(steps), history.flatten())
            plt.xlabel("Time steps")

        plt.ylabel("f(t)")
        plt.show()


if __name__ == '__main__':
    # Example usage
    timeseries = TimeSeries()

    steps = 100
    for i in range(steps):
        timeseries.forward()

    timeseries.plot()
