"""
Module 06 — FDTD Electromagnetic Wave Simulation
==================================================
Finite-Difference Time-Domain (FDTD) solver for microwave field propagation
in the ICP reactor cavity.

Physics:
- Standard Yee grid with staggered E and H fields
- TM mode: Ez, Hx, Hy
- Maxwell's curl equations:
    dHx/dt = -(1/mu_0) * dEz/dy
    dHy/dt = +(1/mu_0) * dEz/dx
    dEz/dt = (1/epsilon_0) * (dHy/dx - dHx/dy)
- Courant-stable time stepping: dt = CFL * min(dx, dy) / c, CFL = 0.5
- Gaussian-modulated sinusoidal source at configurable positions
- Mur first-order absorbing boundary conditions

Implementation notes:
- Time stepping covers whole RF cycles, set by the fdtd_rf_cycles config key
- The Mur absorbing boundary uses the wave-equation time derivative
- Source position is configurable and defaults to the coil positions in state

ML Interface:
- Full field history Ez(x,y,t) is returned for PINN training
- Source function is parameterizable for surrogate model inputs
- CFL number and grid parameters exposed for physics-informed loss terms
"""

import os
import numpy as np
import logging

logger = logging.getLogger(__name__)


def run_fdtd(nx, ny, dx, dy, source_frequency, n_steps=None,
             n_rf_cycles=2.0,
             mu_0=4e-7 * np.pi, epsilon_0=8.854187817e-12,
             source_positions=None, return_history=False):
    """
    Run 2D TM-mode FDTD simulation with Mur 1st-order ABC.

    Args:
        nx, ny: Grid dimensions.
        dx, dy: Grid spacing [m].
        source_frequency: Source frequency [Hz].
        n_steps: Number of time steps (overrides n_rf_cycles if set).
        n_rf_cycles: Number of RF cycles to simulate (default: 2).
        mu_0, epsilon_0: Physical constants.
        source_positions: List of (i, j) source locations. Default: domain center.
        return_history: If True, return field snapshots for animation/ML.

    Returns:
        dict with Ez, Bx, By final fields and optionally history arrays.
    """
    c = 1.0 / np.sqrt(mu_0 * epsilon_0)
    dt = 0.5 * min(dx, dy) / c
    omega = 2 * np.pi * source_frequency
    cfl = c * dt / min(dx, dy)

    # Determine number of time steps
    if n_steps is None:
        T_rf = 1.0 / source_frequency
        n_steps = max(int(n_rf_cycles * T_rf / dt), 20)

    logger.info(f"FDTD: {nx}x{ny} grid, {n_steps} steps, dt={dt:.4e}s, "
                f"CFL={cfl:.3f}, T_sim={n_steps*dt*1e9:.2f} ns")

    if source_positions is None:
        source_positions = [(nx // 2, ny // 2)]

    # Initialize fields
    Ez = np.zeros((nx, ny))
    Hx = np.zeros((nx, ny))
    Hy = np.zeros((nx, ny))

    history_Ez = []

    # Mur ABC coefficients
    coeff_x = (c * dt - dx) / (c * dt + dx)
    coeff_y = (c * dt - dy) / (c * dt + dy)

    # Gaussian source parameters (in time steps)
    # Center at 1.5 RF periods, width = 0.5 RF periods
    T_rf_steps = max(int(2 * np.pi / (omega * dt)), 1)
    t_center = int(1.5 * T_rf_steps)
    t_width = max(int(0.5 * T_rf_steps), 5)

    # Energy tracking
    energy_history = []

    for n in range(n_steps):
        # Record history
        if return_history and n % max(1, n_steps // 50) == 0:
            history_Ez.append(Ez.copy())

        # Track EM energy
        if n % max(1, n_steps // 200) == 0:
            U_E = 0.5 * epsilon_0 * np.sum(Ez**2) * dx * dy
            U_H = 0.5 * mu_0 * np.sum(Hx**2 + Hy**2) * dx * dy
            energy_history.append((n, float(U_E + U_H)))

        # Save boundary and adjacent values (Mur ABC needs these)
        Ez_n_bnd_x0 = Ez[0, :].copy()
        Ez_n_adj_x0 = Ez[1, :].copy()
        Ez_n_bnd_xN = Ez[-1, :].copy()
        Ez_n_adj_xN = Ez[-2, :].copy()
        Ez_n_bnd_y0 = Ez[:, 0].copy()
        Ez_n_adj_y0 = Ez[:, 1].copy()
        Ez_n_bnd_yN = Ez[:, -1].copy()
        Ez_n_adj_yN = Ez[:, -2].copy()

        # Update H-field (half step ahead of E)
        Hx[:, :-1] -= (dt / (mu_0 * dy)) * (Ez[:, 1:] - Ez[:, :-1])
        Hy[:-1, :] += (dt / (mu_0 * dx)) * (Ez[1:, :] - Ez[:-1, :])

        # Update E-field
        Ez[1:-1, 1:-1] += (dt / epsilon_0) * (
            (Hy[1:-1, 1:-1] - Hy[:-2, 1:-1]) / dx -
            (Hx[1:-1, 1:-1] - Hx[1:-1, :-2]) / dy
        )

        # Gaussian-modulated sinusoidal source (soft source)
        gauss_env = np.exp(-((n - t_center) / t_width)**2)
        for (src_i, src_j) in source_positions:
            if 0 < src_i < nx - 1 and 0 < src_j < ny - 1:
                Ez[src_i, src_j] += np.sin(omega * n * dt) * gauss_env

        # Mur first-order absorbing boundary conditions
        # Ez^{n+1}[0] = Ez^n[1] + coeff * (Ez^{n+1}[1] - Ez^n[0])
        Ez[0, :]  = Ez_n_adj_x0 + coeff_x * (Ez[1, :]  - Ez_n_bnd_x0)
        Ez[-1, :] = Ez_n_adj_xN + coeff_x * (Ez[-2, :] - Ez_n_bnd_xN)
        Ez[:, 0]  = Ez_n_adj_y0 + coeff_y * (Ez[:, 1]  - Ez_n_bnd_y0)
        Ez[:, -1] = Ez_n_adj_yN + coeff_y * (Ez[:, -2] - Ez_n_bnd_yN)

    # Convert H to B
    Bx = mu_0 * Hx
    By = mu_0 * Hy

    logger.info(f"FDTD complete: |Ez|_max={np.abs(Ez).max():.4e} V/m, "
                f"|B|_max={np.sqrt(Bx**2+By**2).max():.4e} T")

    result = {
        'Ez_fdtd': Ez, 'Bx_fdtd': Bx, 'By_fdtd': By,
        'dt_fdtd': dt, 'cfl': cfl,
        'n_steps_fdtd': n_steps,
        'energy_history': energy_history,
    }
    if return_history:
        result['Ez_fdtd_history'] = history_Ez
    return result


def plot_fdtd_results(fields, nx, ny, dx, dy, save_folder):
    """Generate FDTD field plots and animations."""
    from ..utils.plotting import plot_results, create_field_animation

    os.makedirs(save_folder, exist_ok=True)
    x_mm = np.linspace(0, (nx - 1) * dx * 1e3, nx)
    y_mm = np.linspace(0, (ny - 1) * dy * 1e3, ny)

    for name, title, unit in [
        ('Ez_fdtd', 'FDTD Electric Field Ez', '[V/m]'),
        ('Bx_fdtd', 'FDTD Magnetic Field Bx', '[T]'),
        ('By_fdtd', 'FDTD Magnetic Field By', '[T]'),
    ]:
        if name in fields:
            plot_results(fields[name], name, save_folder=save_folder,
                         xlabel='r [mm]', ylabel='z [mm]',
                         title=title, x_data=x_mm, y_data=y_mm,
                         cmap='jet', colorbar_label=f'{title} {unit}')

    if 'Ez_fdtd_history' in fields and len(fields['Ez_fdtd_history']) > 1:
        anim_dir = os.path.join(save_folder, 'animations')
        create_field_animation(fields['Ez_fdtd_history'], 'Ez_fdtd_evolution',
                               x_mm, y_mm, 'FDTD Ez Evolution',
                               save_folder=anim_dir, show_geometry=False)


# --- Pipeline interface ---

def run(state, config):
    """Pipeline-compatible entry point for M06."""
    from ..core.units import PhysicalConstants as PC

    sim = config.simulation if hasattr(config, 'simulation') else config.get('simulation', {})
    circ = config.circuit if hasattr(config, 'circuit') else config.get('circuit', {})

    # Determine time stepping: prefer rf_cycles, fall back to explicit steps
    n_rf_cycles = sim.get('fdtd_rf_cycles', None)
    n_steps = sim.get('fdtd_time_steps', None) if n_rf_cycles is None else None
    if n_rf_cycles is None and n_steps is None:
        n_rf_cycles = 2.0  # default

    # Use coil positions as source if available
    source_positions = state.get('coil_centers_idx', None)

    fields = run_fdtd(
        nx=state['nx'], ny=state['ny'],
        dx=state['dx'], dy=state['dy'],
        source_frequency=circ['source_frequency'],
        n_steps=n_steps,
        n_rf_cycles=n_rf_cycles if n_rf_cycles is not None else 2.0,
        mu_0=PC.mu_0, epsilon_0=PC.epsilon_0,
        source_positions=source_positions,
        return_history=True,
    )
    return fields
