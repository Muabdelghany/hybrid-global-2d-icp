"""
Module 05 — Time-Harmonic Electromagnetic Field Solver
========================================================
Evolves EM fields from static initial conditions using Maxwell's equations
with an RF source term distributed across the ICP coil positions.

Physics:
- Uses Bs (from M04) as initial magnetic field
- Uses Phi (from M02) as initial electrostatic field
- Evolves via Yee-grid FDTD update equations over one RF cycle
- TM mode (2D): Ez, Hx, Hy
- Maxwell's equations:
    dHx/dt = -(1/mu_0) * dEz/dy
    dHy/dt = +(1/mu_0) * dEz/dx
    dEz/dt = (1/epsilon_0) * (dHy/dx - dHx/dy) + J_source
- Courant-stable time stepping: dt = 0.5 * min(dx, dy) / c
- Mur first-order absorbing boundary conditions

Implementation notes:
- The RF source is distributed across the coil positions
- The Mur absorbing boundary uses the wave-equation time derivative

ML Interface:
- Field history arrays (Ez_history, B_mag_history) provide time-series
  training data for temporal neural operators and PINNs
- Power deposition P = sigma * |E|^2 is a key output for chemistry coupling
"""

import os
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_em_fields(Phi, Bs_x, Bs_y, nx, ny, dx, dy,
                      source_frequency, I_peak, mu_0, epsilon_0,
                      coil_centers_idx=None, return_history=False):
    """
    Compute electromagnetic fields by evolving from static initial conditions.

    Uses Yee-grid FDTD with the magnetostatic solution as initial B-field
    and RF source distributed across coil positions.

    Args:
        Phi: Electrostatic potential (nx, ny) [V].
        Bs_x, Bs_y: Magnetostatic field components (nx, ny) [T].
        nx, ny: Grid dimensions.
        dx, dy: Grid spacing [m].
        source_frequency: RF source frequency [Hz].
        I_peak: Peak coil current [A].
        mu_0, epsilon_0: Physical constants.
        coil_centers_idx: List of (i, j) coil positions for RF source injection.
                         If None, uses domain center (legacy behavior).
        return_history: If True, return field snapshots for animation/ML.

    Returns:
        dict with Ex, Ey, Ez, E_mag, Bx, By, B_mag fields.
        If return_history: also includes 'Ez_history' and 'B_history'.
    """
    c = 1.0 / np.sqrt(mu_0 * epsilon_0)
    dt = 0.5 * min(dx, dy) / c
    omega = 2 * np.pi * source_frequency

    # Number of steps for one RF cycle
    n_steps = max(int(2 * np.pi / (omega * dt)), 20)
    logger.info(f"EM evolution: {n_steps} steps, dt={dt:.4e} s, f={source_frequency/1e6:.1f} MHz")

    # Source positions
    if coil_centers_idx is None or len(coil_centers_idx) == 0:
        coil_centers_idx = [(nx // 2, ny // 2)]
        logger.warning("No coil positions provided — using domain center as source")
    n_coils = len(coil_centers_idx)

    # Initialize from static solutions
    Ex_static = np.zeros((nx, ny))
    Ey_static = np.zeros((nx, ny))
    Ex_static[1:-1, :] = -(Phi[2:, :] - Phi[:-2, :]) / (2 * dx)
    Ey_static[:, 1:-1] = -(Phi[:, 2:] - Phi[:, :-2]) / (2 * dy)

    Ez = np.zeros((nx, ny))
    Hx = Bs_x / mu_0
    Hy = Bs_y / mu_0

    Ez_history = []
    B_history = []

    # Mur ABC coefficients
    coeff_x = (c * dt - dx) / (c * dt + dx)
    coeff_y = (c * dt - dy) / (c * dt + dy)

    for n in range(n_steps):
        # Store snapshots
        if return_history and n % max(1, n_steps // 50) == 0:
            Ez_history.append(Ez.copy())
            B_history.append(np.sqrt((mu_0 * Hx)**2 + (mu_0 * Hy)**2).copy())

        # Save boundary and adjacent values at current time step (before update)
        # Mur 1st-order: Ez^{n+1}[0] = Ez^n[1] + coeff*(Ez^{n+1}[1] - Ez^n[0])
        Ez_n_bnd_x0 = Ez[0, :].copy()
        Ez_n_adj_x0 = Ez[1, :].copy()
        Ez_n_bnd_xN = Ez[-1, :].copy()
        Ez_n_adj_xN = Ez[-2, :].copy()
        Ez_n_bnd_y0 = Ez[:, 0].copy()
        Ez_n_adj_y0 = Ez[:, 1].copy()
        Ez_n_bnd_yN = Ez[:, -1].copy()
        Ez_n_adj_yN = Ez[:, -2].copy()

        # FDTD update: H-field
        Hx[:, :-1] -= (dt / (mu_0 * dy)) * (Ez[:, 1:] - Ez[:, :-1])
        Hy[:-1, :] += (dt / (mu_0 * dx)) * (Ez[1:, :] - Ez[:-1, :])

        # FDTD update: E-field
        Ez[1:-1, 1:-1] += (dt / epsilon_0) * (
            (Hy[1:-1, 1:-1] - Hy[:-2, 1:-1]) / dx -
            (Hx[1:-1, 1:-1] - Hx[1:-1, :-2]) / dy
        )

        # RF source distributed across coil positions
        for (ic, jc) in coil_centers_idx:
            if 0 < ic < nx - 1 and 0 < jc < ny - 1:
                Ez[ic, jc] += (I_peak / n_coils) * np.sin(omega * n * dt)

        # Mur first-order absorbing boundary conditions
        # Ez^{n+1}[0] = Ez^n[1] + coeff * (Ez^{n+1}[1] - Ez^n[0])
        Ez[0, :]  = Ez_n_adj_x0 + coeff_x * (Ez[1, :]  - Ez_n_bnd_x0)
        Ez[-1, :] = Ez_n_adj_xN + coeff_x * (Ez[-2, :] - Ez_n_bnd_xN)
        Ez[:, 0]  = Ez_n_adj_y0 + coeff_y * (Ez[:, 1]  - Ez_n_bnd_y0)
        Ez[:, -1] = Ez_n_adj_yN + coeff_y * (Ez[:, -2] - Ez_n_bnd_yN)

    # Final fields
    Ex = Ex_static + np.gradient(-Ez, dx, axis=0)
    Ey = Ey_static + np.gradient(-Ez, dy, axis=1)
    E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    Bx = mu_0 * Hx
    By = mu_0 * Hy
    B_mag = np.sqrt(Bx**2 + By**2)

    logger.info(f"EM fields computed: |E|_max={E_mag.max():.4e} V/m, |B|_max={B_mag.max():.4e} T")

    result = {
        'Ex': Ex, 'Ey': Ey, 'Ez': Ez, 'E_mag': E_mag,
        'Bx': Bx, 'By': By, 'B_mag': B_mag,
    }
    if return_history:
        result['Ez_history'] = Ez_history
        result['B_history'] = B_history

    return result


def plot_em_results(fields, nx, ny, dx, dy, save_folder):
    """Generate electromagnetic field plots and animations."""
    from ..utils.plotting import (plot_results, plot_vector_field,
                                   create_field_animation)

    os.makedirs(save_folder, exist_ok=True)
    x_mm = np.linspace(0, (nx - 1) * dx * 1e3, nx)
    y_mm = np.linspace(0, (ny - 1) * dy * 1e3, ny)

    for name, label, unit in [
        ('E_mag', 'EM Electric Field Magnitude', '[V/m]'),
        ('B_mag', 'EM Magnetic Field Magnitude', '[T]'),
        ('Ez', 'EM Electric Field Ez', '[V/m]'),
    ]:
        if name in fields:
            plot_results(fields[name], f'{name}_em', save_folder=save_folder,
                         xlabel='r [mm]', ylabel='z [mm]',
                         title=label, x_data=x_mm, y_data=y_mm,
                         cmap='jet', colorbar_label=f'{label} {unit}')

    # Vector plots
    if all(k in fields for k in ('Ex', 'Ey')):
        plot_vector_field(fields['Ex'], fields['Ey'], 'E_vectors_em',
                          'EM Electric Field', x_mm, y_mm, dx, dy,
                          field_type='electromagnetic', save_folder=save_folder)

    if all(k in fields for k in ('Bx', 'By')):
        plot_vector_field(fields['Bx'], fields['By'], 'B_vectors_em',
                          'EM Magnetic Field', x_mm, y_mm, dx, dy,
                          field_type='electromagnetic', save_folder=save_folder)

    # Animations
    anim_dir = os.path.join(save_folder, 'animations')
    if 'Ez_history' in fields and len(fields['Ez_history']) > 1:
        create_field_animation(fields['Ez_history'], 'Ez_evolution',
                               x_mm, y_mm, 'Ez Field Evolution',
                               save_folder=anim_dir, show_geometry=False)
    if 'B_history' in fields and len(fields['B_history']) > 1:
        create_field_animation(fields['B_history'], 'B_evolution',
                               x_mm, y_mm, 'B Field Evolution',
                               save_folder=anim_dir)


# --- Pipeline interface ---

def run(state, config):
    """Pipeline-compatible entry point for M05."""
    from ..core.units import PhysicalConstants as PC

    sim = config.simulation if hasattr(config, 'simulation') else config.get('simulation', {})
    circ = config.circuit if hasattr(config, 'circuit') else config.get('circuit', {})

    fields = compute_em_fields(
        Phi=state['Phi'],
        Bs_x=state['Bs_x'],
        Bs_y=state['Bs_y'],
        nx=state['nx'], ny=state['ny'],
        dx=state['dx'], dy=state['dy'],
        source_frequency=circ['source_frequency'],
        I_peak=state['I_peak'],
        mu_0=PC.mu_0,
        epsilon_0=PC.epsilon_0,
        coil_centers_idx=state.get('coil_centers_idx', None),
        return_history=True,
    )

    # Update state with EM field results
    return {
        'Ex': fields['Ex'], 'Ey': fields['Ey'], 'Ez': fields['Ez'],
        'E_mag': fields['E_mag'],
        'Bx': fields['Bx'], 'By': fields['By'], 'B_mag': fields['B_mag'],
        'Ez_history': fields.get('Ez_history', []),
        'B_history': fields.get('B_history', []),
    }
