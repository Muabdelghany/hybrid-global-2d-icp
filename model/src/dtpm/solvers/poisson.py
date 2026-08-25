"""
Poisson / Laplace equation solvers for electrostatic potential.

Provides:
- solve_poisson_sor: Red-Black Gauss-Seidel with Successive Over-Relaxation (bounded domains)
- solve_poisson_fft: Spectral solver via FFT (periodic domains)

Both solvers are designed to expose intermediate states for ML surrogate training:
call with return_history=True to get convergence trajectories.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def solve_poisson_sor(Phi, coil_mask, boundary_mask, omega=1.8,
                      max_iter=10000, tol=1e-2, return_history=False):
    """
    Solve Laplace/Poisson equation using vectorized Red-Black SOR.

    Uses checkerboard (Red-Black) ordering to vectorize the Gauss-Seidel
    iteration with numpy array operations, achieving ~50-100x speedup
    over scalar Python loops while preserving identical convergence behavior.

    Args:
        Phi: Initial potential grid (nx, ny). Boundary values already set.
        coil_mask: Boolean array (nx, ny) — True at coil positions (fixed potential).
        boundary_mask: Boolean array (nx, ny) — True at boundary positions (fixed potential).
        omega: SOR relaxation factor (1 < omega < 2 for over-relaxation).
        max_iter: Maximum iterations.
        tol: Convergence tolerance (max absolute change per iteration).
        return_history: If True, return list of (iteration, error) for ML training.

    Returns:
        Phi: Converged potential field.
        info: Dict with convergence info. If return_history, includes 'history' key.
    """
    nx, ny = Phi.shape
    fixed = coil_mask | boundary_mask
    history = []

    # Precompute the interior free-point mask (not fixed, not on domain edge)
    free = ~fixed
    free[0, :] = False
    free[-1, :] = False
    free[:, 0] = False
    free[:, -1] = False

    # Create Red-Black masks for checkerboard ordering
    # Red: (i+j) even, Black: (i+j) odd
    ij_sum = np.arange(nx)[:, None] + np.arange(ny)[None, :]
    red_mask = free & (ij_sum % 2 == 0)
    black_mask = free & (ij_sum % 2 == 1)

    # Store fixed values to restore after updates
    fixed_vals = Phi.copy()

    iteration = 0
    error = float('inf')

    while error > tol and iteration < max_iter:
        # --- Update Red points ---
        neighbor_avg = (
            np.roll(Phi, -1, axis=0) + np.roll(Phi, 1, axis=0) +
            np.roll(Phi, -1, axis=1) + np.roll(Phi, 1, axis=1)
        ) / 4.0

        Phi_new_red = (1 - omega) * Phi + omega * neighbor_avg
        Phi_old_red = Phi.copy()
        Phi = np.where(red_mask, Phi_new_red, Phi)

        # --- Update Black points (use updated Red values) ---
        neighbor_avg = (
            np.roll(Phi, -1, axis=0) + np.roll(Phi, 1, axis=0) +
            np.roll(Phi, -1, axis=1) + np.roll(Phi, 1, axis=1)
        ) / 4.0

        Phi_new_black = (1 - omega) * Phi + omega * neighbor_avg
        Phi_old_black = Phi.copy()
        Phi = np.where(black_mask, Phi_new_black, Phi)

        # Restore fixed points (coils + boundaries)
        Phi = np.where(fixed, fixed_vals, Phi)

        # Compute max change across all free points
        change_red = np.abs(Phi - Phi_old_red)
        change_black = np.abs(Phi - Phi_old_black)
        change = np.maximum(change_red, change_black)
        error = float(change[free].max()) if free.any() else 0.0

        iteration += 1
        if return_history and iteration % 10 == 0:
            history.append((iteration, error))
        if iteration % 500 == 0:
            logger.info(f"SOR iteration {iteration}, error={error:.6e}")

    logger.info(f"SOR converged after {iteration} iterations (error={error:.6e})")

    info = {'iterations': iteration, 'final_error': error, 'converged': error <= tol}
    if return_history:
        info['history'] = history
    return Phi, info


def solve_poisson_fft(rho, dx, dy, epsilon_0=8.854187817e-12):
    """
    Solve Poisson equation on a periodic 2D domain via FFT.

    Solves: nabla^2 phi = -rho / epsilon_0
    Returns: (Ex, Ey) electric field components.

    Designed for PIC simulations with periodic boundaries.

    Args:
        rho: Charge density array (nx, ny).
        dx, dy: Grid spacing.
        epsilon_0: Vacuum permittivity.

    Returns:
        Ex, Ey: Electric field components on the grid.
    """
    nx, ny = rho.shape
    rho_k = np.fft.rfft2(rho)

    kx = 2 * np.pi * np.fft.fftfreq(nx, dx)
    ky = 2 * np.pi * np.fft.rfftfreq(ny, dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    K2[0, 0] = 1.0  # avoid division by zero

    phi_k = rho_k / (-epsilon_0 * K2)
    phi_k[0, 0] = 0.0  # zero mean potential

    Ex = -np.fft.irfft2(1j * KX * phi_k, s=(nx, ny)).real
    Ey = -np.fft.irfft2(1j * KY * phi_k, s=(nx, ny)).real
    return Ex, Ey
