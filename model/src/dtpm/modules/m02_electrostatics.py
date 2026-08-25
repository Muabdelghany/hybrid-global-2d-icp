"""
Module 02 — Electrostatic Potential & Electric Field Solver
=============================================================
Solves the Laplace equation for electric potential using SOR,
then computes E-field via central finite differences.

Physics:
- Laplace equation: nabla^2 Phi = 0 (source-free region)
- E = -grad(Phi)
- Gauss-Seidel with Successive Over-Relaxation (SOR)
- Dirichlet BCs on reactor walls and coil surfaces

ML Interface:
- solve() returns convergence history for surrogate training
- compute_electric_field() uses central differences (2nd order accurate)
  exposing (Phi, Ex, Ey, E_mag) as training data fields

Implementation notes:
- The E-field uses central differences, -(Phi[i+1]-Phi[i-1])/(2*dx)
- Colorbar labels carry [V/m] for E-field plots
"""

import os
import numpy as np
import logging

from ..solvers.poisson import solve_poisson_sor
from . import boundary_conditions as bc

logger = logging.getLogger(__name__)


def initialize_potential(nx, ny, coil_centers_idx, coil_radius_idx,
                         coil_potential, boundary_config):
    """
    Create initial potential field with boundary and coil conditions.

    Args:
        nx, ny: Grid dimensions.
        coil_centers_idx: List of (i, j) coil center indices.
        coil_radius_idx: Coil radius in grid points.
        coil_potential: Voltage applied to coils [V].
        boundary_config: Boundary condition configuration dict.

    Returns:
        Phi: Initial potential array (nx, ny).
        coil_mask: Boolean mask for coil positions.
        boundary_mask: Boolean mask for boundary positions.
    """
    Phi = np.zeros((nx, ny))

    # Set coil potentials
    coil_mask = np.zeros((nx, ny), dtype=bool)
    for xc, yc in coil_centers_idx:
        x_lo = max(0, xc - coil_radius_idx)
        x_hi = min(nx, xc + coil_radius_idx + 1)
        y_lo = max(0, yc - coil_radius_idx)
        y_hi = min(ny, yc + coil_radius_idx + 1)
        for i in range(x_lo, x_hi):
            for j in range(y_lo, y_hi):
                if (i - xc)**2 + (j - yc)**2 <= coil_radius_idx**2:
                    Phi[i, j] = coil_potential
                    coil_mask[i, j] = True

    # Apply wall boundary conditions
    Phi = bc.apply_boundary_conditions(Phi, boundary_config)
    boundary_mask = np.zeros((nx, ny), dtype=bool)
    for (i, j) in bc.get_boundary_points(boundary_config):
        if 0 <= i < nx and 0 <= j < ny:
            boundary_mask[i, j] = True

    n_coil = int(coil_mask.sum())
    n_bc = int(boundary_mask.sum())
    logger.info(f"Potential initialized: {nx}x{ny} grid, {n_coil} coil points, {n_bc} BC points")
    logger.info(f"  Phi range: [{Phi.min():.2f}, {Phi.max():.2f}] V")

    return Phi, coil_mask, boundary_mask


def solve_potential(Phi, coil_mask, boundary_mask, boundary_config,
                    omega=1.8, max_iter=10000, tol=1e-2, return_history=False):
    """
    Solve Laplace equation using SOR with boundary condition enforcement.

    After convergence, re-applies boundary conditions for consistency.

    Args:
        Phi: Initial potential (nx, ny) with BCs set.
        coil_mask: Boolean mask for coil positions.
        boundary_mask: Boolean mask for boundary positions.
        boundary_config: Boundary condition dict.
        omega: SOR relaxation parameter.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.
        return_history: If True, include convergence trajectory in result.

    Returns:
        Phi: Converged potential field (nx, ny).
        info: Convergence info dict.
    """
    Phi, info = solve_poisson_sor(
        Phi, coil_mask, boundary_mask,
        omega=omega, max_iter=max_iter, tol=tol,
        return_history=return_history
    )

    # Re-apply BCs after final iteration
    Phi = bc.apply_boundary_conditions(Phi, boundary_config)

    logger.info(f"Potential solved: {info['iterations']} iters, error={info['final_error']:.2e}")
    return Phi, info


def compute_electric_field(Phi, dx, dy):
    """
    Compute electric field from potential using central finite differences.

    Uses 2nd-order accurate central differences for interior points
    and 1st-order one-sided differences at boundaries.

    Args:
        Phi: Electric potential (nx, ny) [V].
        dx, dy: Grid spacing [m].

    Returns:
        Ex, Ey: Electric field components [V/m].
        E_mag: Electric field magnitude [V/m].
    """
    nx, ny = Phi.shape
    Ex = np.zeros((nx, ny))
    Ey = np.zeros((nx, ny))

    # Central differences for interior
    Ex[1:-1, :] = -(Phi[2:, :] - Phi[:-2, :]) / (2 * dx)
    Ey[:, 1:-1] = -(Phi[:, 2:] - Phi[:, :-2]) / (2 * dy)

    # One-sided differences at boundaries
    Ex[0, :] = -(Phi[1, :] - Phi[0, :]) / dx
    Ex[-1, :] = -(Phi[-1, :] - Phi[-2, :]) / dx
    Ey[:, 0] = -(Phi[:, 1] - Phi[:, 0]) / dy
    Ey[:, -1] = -(Phi[:, -1] - Phi[:, -2]) / dy

    E_mag = np.sqrt(Ex**2 + Ey**2)

    logger.info(f"E-field computed: |E|_max = {E_mag.max():.4e} V/m")
    return Ex, Ey, E_mag


def plot_electrostatic_results(Phi, Ex, Ey, E_mag, nx, ny, dx, dy,
                                coil_potential, save_folder):
    """Generate all electrostatic plots (potential, E-field, equipotentials)."""
    from ..utils.plotting import (plot_results, plot_vector_field,
                                   plot_equipotential_lines, plot_field_streamlines)

    os.makedirs(save_folder, exist_ok=True)
    x_mm = np.linspace(0, (nx - 1) * dx * 1e3, nx)
    y_mm = np.linspace(0, (ny - 1) * dy * 1e3, ny)

    # Potential distribution
    plot_results(Phi, 'electric_potential', save_folder=save_folder,
                 xlabel='x [mm]', ylabel='y [mm]',
                 title=f'Electric Potential (V_coil={coil_potential:.0f}V)',
                 x_data=x_mm, y_data=y_mm, cmap='jet',
                 colorbar_label='Electric Potential [V]')

    # E-field magnitude
    plot_results(E_mag, 'E_field_magnitude', save_folder=save_folder,
                 xlabel='x [mm]', ylabel='y [mm]',
                 title='Electric Field Magnitude',
                 x_data=x_mm, y_data=y_mm, cmap='hot',
                 colorbar_label='|E| [V/m]')

    # Vector field
    plot_vector_field(Ex, Ey, 'E_field_vectors',
                      'Electric Field Vectors', x_mm, y_mm, dx, dy,
                      field_type='electrostatic', save_folder=save_folder)

    # Streamlines
    plot_field_streamlines(Ex, Ey, 'E_field_streamlines',
                           'Electric Field Lines', x_mm, y_mm,
                           field_type='electrostatic', save_folder=save_folder)

    # Equipotential lines
    plot_equipotential_lines(Phi, coil_potential, save_folder, dx, dy)


# --- Pipeline interface ---

def run(state, config):
    """Pipeline-compatible entry point for M02."""
    sim = config.to_simulation_params() if hasattr(config, 'to_simulation_params') else config
    from ..core.grid import Grid
    grid = Grid(config.grid if hasattr(config, 'grid') else config['grid'])

    nx, ny, dx, dy = grid.nx, grid.ny, grid.dx, grid.dy
    coil_centers_idx = state.get('coil_centers_idx', [])
    coil_radius_idx = state.get('coil_radius_idx', 1)
    coil_potential = state['V_peak']

    # Get boundary config
    bc_config = config.boundary_conditions if hasattr(config, 'boundary_conditions') else {}
    # Convert to legacy format if needed
    if 'dirichlet_lines' in bc_config and 'dirichlet_boundaries' not in bc_config:
        bc_config = {
            'dirichlet_boundaries': {
                'lines': [(e[0], e[1], e[2]) for e in bc_config['dirichlet_lines']],
                'points': [],
            },
            'neumann_boundaries': {'lines': [], 'points': []},
        }

    sim_cfg = config.simulation if hasattr(config, 'simulation') else config.get('simulation', {})
    omega = sim_cfg.get('relaxation_omega', 1.8)
    max_iter = sim_cfg.get('max_iter', 10000)
    tol = sim_cfg.get('tolerance', 1e-2)

    Phi, coil_mask, boundary_mask = initialize_potential(
        nx, ny, coil_centers_idx, coil_radius_idx, coil_potential, bc_config)

    Phi, info = solve_potential(
        Phi, coil_mask, boundary_mask, bc_config,
        omega=omega, max_iter=max_iter, tol=tol, return_history=True)

    Ex, Ey, E_mag = compute_electric_field(Phi, dx, dy)

    return {
        'Phi': Phi,
        'Ex': Ex, 'Ey': Ey, 'E_mag': E_mag,
        'coil_mask': coil_mask,
        'boundary_mask': boundary_mask,
        'boundary_config': bc_config,
        'sor_info': info,
        'nx': nx, 'ny': ny, 'dx': dx, 'dy': dy,
    }
