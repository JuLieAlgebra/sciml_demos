""" 
Demonstrate two ways of solving the 1D wave equation with Dirichlet boundary conditions
and a particular initial condition. PINNs are not typically a useful tool for problems concerning
PDEs, but this is a demonstration of their usage as an ansatz.
"""
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ----------------------------
# Finite Difference Solver
# ----------------------------
class FiniteDifferenceWaveSolver:
    """
    Finite Difference solver for the 1D wave equation

    The PDE is: u_{tt}=c^2*u_{xx}
    with Dirichlet boundary condition u(0,t)=u(L,t)=0 and initial conditions
    u(x,0)=sin(pi*x) and u_t(x,0)=0.
    """

    def __init__(self, L: float, c: float, T: float, nx: int, nt: int):
        self.L = L  # Spatial domain length
        self.c = c  # Wave speed
        self.T = T  # Total simulation time
        self.nx = nx  # Number of spatial grid points
        self.nt = nt  # Number of time steps

        self.dx = L / (nx - 1)
        self.dt = T / (nt - 1)
        self.lambda_val = c * self.dt / self.dx  # Courant number

        self.x = np.linspace(0, L, nx)  # x-grid
        self.t = np.linspace(0, T, nt)  # t-grid
        self.u = np.zeros((nt, nx))  # Solution array

    def initialize(self) -> None:
        """Set the initial conditions and enforce boundary conditions."""
        # Initial condition: u(x,0) = sin(pi*x)
        self.u[0, :] = np.sin(np.pi * self.x)

        # Using Taylor expansion for the first time step:
        # u(x, dt) ≈ u(x,0) + (1/2)*λ²*(u(x+dx,0) - 2*u(x,0) + u(x-dx,0))
        self.u[1, 1:-1] = self.u[0, 1:-1] + 0.5 * self.lambda_val**2 * (
            self.u[0, 2:] - 2 * self.u[0, 1:-1] + self.u[0, :-2]
        )
        # Enforce boundary conditions
        self.u[:, 0] = 0
        self.u[:, -1] = 0

    def run(self) -> None:
        """Time-march the solution using an explicit finite difference scheme."""
        self.initialize()
        for n in range(1, self.nt - 1):
            self.u[n + 1, 1:-1] = (
                2 * self.u[n, 1:-1]
                - self.u[n - 1, 1:-1]
                + self.lambda_val**2
                * (self.u[n, 2:] - 2 * self.u[n, 1:-1] + self.u[n, :-2])
            )

    def plot(self, step: int = 25) -> None:
        """Plot snapshots of the wave evolution."""
        plt.figure(figsize=(8, 6))
        for i in range(0, self.nt, step):
            plt.plot(self.x, self.u[i, :], label=f"t = {self.t[i]:.2f}")
        plt.xlabel("x")
        plt.ylabel("u")
        plt.title("1D Wave Equation (Finite Difference)")
        plt.legend()
        plt.show()


# ----------------------------
# PINN (Physics-Informed Neural Network) Solver
# ----------------------------
class WavePINN(nn.Module):
    """
    Neural network architecture for approximating the wave equation solution.

    The network takes as input (x,t) and outputs a scalar u(x,t).
    """

    def __init__(self):
        super(WavePINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Concatenate spatial and temporal coordinates
        X = torch.cat([x, t], dim=1)
        return self.net(X)


class WaveEquationPINNSolver:
    """
    A PINN solver for the 1D wave equation:

    u_{tt}=c^2*u_{xx}

    with u(0,t)=u(L,t)=0, u(x,0)=sin(pi*x), and u_t(x,0)=0.
    """

    def __init__(
        self,
        L: float,
        T: float,
        c: float,
        n_collocation: int = 10000,
        n_ic: int = 200,
        n_bc: int = 200,
        device: Optional[str] = None,
    ):
        self.L = L  # Spatial domain length
        self.T = T  # Temporal domain length
        self.c = c  # Wave speed

        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else torch.device(device)
        )
        print(f"Using {self.device}")
        self.model = WavePINN().to(self.device)

        # Collocation points for enforcing the PDE residual
        self.n_collocation = n_collocation
        self.x_coll = torch.rand(n_collocation, 1, device=self.device) * self.L
        self.t_coll = torch.rand(n_collocation, 1, device=self.device) * self.T

        # Initial condition: u(x,0) = sin(pi*x)
        self.n_ic = n_ic
        self.x_ic = torch.linspace(0, self.L, n_ic, device=self.device).unsqueeze(1)
        self.t_ic = torch.zeros_like(self.x_ic, device=self.device)
        self.u_ic = torch.sin(np.pi * self.x_ic)

        # Boundary conditions: u(0,t)=0 and u(L,t)=0
        self.n_bc = n_bc
        self.t_bc = torch.linspace(0, self.T, n_bc, device=self.device).unsqueeze(1)
        self.x_bc0 = torch.zeros_like(self.t_bc, device=self.device)
        self.x_bcL = self.L * torch.ones_like(self.t_bc, device=self.device)

        # For the initial velocity condition u_t(x,0) = 0, we keep a set of points with gradients
        self.x_ic_v = self.x_ic.clone().detach().requires_grad_(True)
        self.t_ic_v = self.t_ic.clone().detach().requires_grad_(True)

        self.mse_loss = nn.MSELoss()

    def pde_residual(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the PDE residual:
        \( u_{tt} - c^2 u_{xx} \)
        using automatic differentiation.
        """
        x.requires_grad_(True)
        t.requires_grad_(True)
        u = self.model(x, t)

        # First derivative in time
        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        # Second derivative in time
        u_tt = torch.autograd.grad(
            u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True
        )[0]

        # First derivative in space
        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        # Second derivative in space
        u_xx = torch.autograd.grad(
            u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
        )[0]

        return u_tt - self.c**2 * u_xx

    def loss(self) -> torch.Tensor:
        """Compute the total loss (PDE residual, initial and boundary conditions)."""
        # PDE Loss: enforce the PDE residual at collocation points
        x_coll = self.x_coll.clone().detach().requires_grad_(True)
        t_coll = self.t_coll.clone().detach().requires_grad_(True)
        residual = self.pde_residual(x_coll, t_coll)
        loss_pde = self.mse_loss(residual, torch.zeros_like(residual))

        # Initial condition loss: u(x,0) = sin(pi*x)
        u_ic_pred = self.model(self.x_ic, self.t_ic)
        loss_ic = self.mse_loss(u_ic_pred, self.u_ic)

        # Initial velocity loss: u_t(x,0) = 0
        u_ic_pred_v = self.model(self.x_ic_v, self.t_ic_v)
        u_t_ic = torch.autograd.grad(
            u_ic_pred_v,
            self.t_ic_v,
            grad_outputs=torch.ones_like(u_ic_pred_v),
            create_graph=True,
        )[0]
        loss_ic_v = self.mse_loss(u_t_ic, torch.zeros_like(u_t_ic))

        # Boundary condition loss: u(0,t)=0 and u(L,t)=0
        u_bc0 = self.model(self.x_bc0, self.t_bc)
        u_bcL = self.model(self.x_bcL, self.t_bc)
        loss_bc = self.mse_loss(u_bc0, torch.zeros_like(u_bc0)) + self.mse_loss(
            u_bcL, torch.zeros_like(u_bcL)
        )

        total_loss = loss_pde + loss_ic + loss_ic_v + loss_bc
        return total_loss

    def train(self, n_epochs: int = 1000, lr: float = 1e-3) -> None:
        """Train the PINN using the Adam optimizer."""
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            loss = self.loss()
            loss.backward()
            optimizer.step()
            if epoch % 500 == 0:
                print(f"Epoch {epoch:5d} - Loss: {loss.item():.6f}")

    def predict(self, n_plot: int = 101) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the trained model on a mesh grid for visualization."""
        x_star = torch.linspace(0, self.L, n_plot, device=self.device).unsqueeze(1)
        t_star = torch.linspace(0, self.T, n_plot, device=self.device).unsqueeze(1)
        X_star, T_star = torch.meshgrid(
            x_star.squeeze(1), t_star.squeeze(1), indexing="ij"
        )
        X_star_flat = X_star.reshape(-1, 1)
        T_star_flat = T_star.reshape(-1, 1)

        self.model.eval()
        with torch.no_grad():
            u_pred = self.model(X_star_flat, T_star_flat)
        u_pred = u_pred.cpu().numpy().reshape(n_plot, n_plot)
        return x_star.cpu().numpy(), t_star.cpu().numpy(), u_pred

    def plot(self, n_plot: int = 101, time_stride: int = 20) -> None:
        """
        Plot solution snapshots extracted from the PINN prediction.

        :param n_plot: Number of points in space and time (mesh grid resolution)
        :param time_stride: Stride for plotting time snapshots
        """
        x, t, u_pred = self.predict(n_plot=n_plot)
        plt.figure(figsize=(8, 6))
        for i in range(0, n_plot, time_stride):
            plt.plot(x, u_pred[:, i], label=f"t = {t[i][0]:.2f}")
        plt.xlabel("x")
        plt.ylabel("u")
        plt.title("1D Wave Equation (PINN Approximation)")
        plt.legend()
        plt.show()


# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    # Finite Difference Simulation
    fd_solver = FiniteDifferenceWaveSolver(L=1.0, c=1.0, T=1.0, nx=101, nt=251)
    fd_solver.run()
    fd_solver.plot(step=25)

    # PINN Simulation
    pinn_solver = WaveEquationPINNSolver(L=1.0, T=1.0, c=1.0)
    pinn_solver.train(n_epochs=5000, lr=1e-3)
    pinn_solver.plot(n_plot=101, time_stride=20)
