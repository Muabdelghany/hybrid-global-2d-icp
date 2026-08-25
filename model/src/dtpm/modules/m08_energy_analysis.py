"""
Module 08 — Energy Analysis & EEDF Computation
=================================================
Post-processes particle data from M07 to extract energy diagnostics:
- Electron Energy Distribution Function (EEDF)
- Energy conservation check
- Mean electron temperature
- Heating rate spatial profile

Physics:
- EEDF: f(epsilon) from particle velocity distribution
  epsilon = 0.5 * m_e * (vx^2 + vy^2 + vz^2) / eV
- Mean Te: Te = (2/3) * <epsilon>  (3D Maxwellian)
- Energy conservation: Delta_KE should equal integral of E dot J dt

Replaces legacy m08_non_collision_heating.py which used fragile
trajectory-gradient method for energy computation.

ML Interface:
- EEDF shape parameters for surrogate model training
- Energy conservation error as physics-informed loss term
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_eedf(vx, vy, vz, m_e, eV, n_bins=100, e_max=None):
    """
    Compute Electron Energy Distribution Function from particle velocities.

    Args:
        vx, vy, vz: Electron velocity components [m/s].
        m_e: Electron mass [kg].
        eV: 1 eV in Joules.
        n_bins: Number of energy bins.
        e_max: Maximum energy [eV]. Auto-detected if None.

    Returns:
        eedf: Normalized distribution f(epsilon).
        bin_centers: Energy bin centers [eV].
    """
    energies = 0.5 * m_e * (vx**2 + vy**2 + vz**2) / eV

    if e_max is None:
        e_max = min(np.percentile(energies, 99.5), energies.max())
        e_max = max(e_max, 1.0)  # at least 1 eV

    eedf, bin_edges = np.histogram(energies, bins=n_bins,
                                    range=(0, e_max), density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return eedf, bin_centers


def compute_energy_conservation(KE_history, work_history, dt):
    """
    Check energy conservation: Delta_KE = sum(work done by E-field).

    Args:
        KE_history: Array of total KE at each step [J].
        work_history: Array of work done per step [J].
        dt: Time step [s].

    Returns:
        dict with conservation metrics.
    """
    delta_KE = KE_history[-1] - KE_history[0]
    total_work = np.sum(work_history)

    if abs(total_work) > 1e-30:
        relative_error = abs(delta_KE - total_work) / abs(total_work)
    else:
        relative_error = abs(delta_KE) if abs(delta_KE) > 1e-30 else 0.0

    return {
        'delta_KE': float(delta_KE),
        'total_work': float(total_work),
        'conservation_error': float(relative_error),
    }


def compute_heating_profile(xe, ye, vxe, vye, vze, Ex, Ey,
                            nx, ny, dx, dy, m_e, q_e):
    """
    Compute spatial heating rate profile from particle data.

    P(r,z) = sum_particles [ q_e * (E_p dot v_p) ] binned onto grid.

    Args:
        xe, ye: Electron positions [m].
        vxe, vye, vze: Electron velocities [m/s].
        Ex, Ey: Electric field on grid (nx, ny) [V/m].
        nx, ny: Grid dimensions.
        dx, dy: Grid spacing.
        m_e: Electron mass.
        q_e: Electron charge.

    Returns:
        P_heat: Heating rate profile (nx, ny) [W/m^3].
    """
    from ..solvers.pic_solver import cic_gather

    Ex_p = cic_gather(xe, ye, Ex, nx, ny, dx, dy)
    Ey_p = cic_gather(xe, ye, Ey, nx, ny, dx, dy)

    # Power per particle: P = q * (E dot v)
    power_per_particle = q_e * (Ex_p * vxe + Ey_p * vye)

    # Deposit power onto grid
    P_heat = np.zeros((nx, ny))
    ix = np.clip((xe / dx).astype(int), 0, nx - 1)
    iy = np.clip((ye / dy).astype(int), 0, ny - 1)
    np.add.at(P_heat, (ix, iy), power_per_particle)

    # Normalize to W/m^3
    cell_volume = dx * dy * 1.0  # 2D: per unit depth
    P_heat /= cell_volume

    return P_heat


# --- Pipeline interface ---

def run(state, config):
    """Pipeline-compatible entry point for M08."""
    from ..core.units import PhysicalConstants as PC

    sim = config.simulation if hasattr(config, 'simulation') else config.get('simulation', {})
    dt = sim.get('dt', 1e-10)

    # Extract particle data from M07
    vxe = state.get('vxe', None)
    if vxe is None:
        logger.warning("M08: No particle data found — skipping energy analysis")
        return {}

    vye = state['vye']
    vze = state['vze']
    xe = state['xe']
    ye = state['ye']

    # --- EEDF ---
    eedf, eedf_bins = compute_eedf(vxe, vye, vze, PC.m_e, PC.eV)

    # --- Mean electron temperature ---
    energies = 0.5 * PC.m_e * (vxe**2 + vye**2 + vze**2) / PC.eV
    mean_Te_eV = float(2.0 / 3.0 * np.mean(energies))

    logger.info(f"Energy analysis: mean Te = {mean_Te_eV:.2f} eV, "
                f"median E = {np.median(energies):.2f} eV")

    # --- Energy conservation ---
    KE_history = state.get('KE_history', np.array([]))
    work_history = state.get('work_history', np.array([]))
    if len(KE_history) > 1:
        conservation = compute_energy_conservation(KE_history, work_history, dt)
        logger.info(f"  Energy conservation error: {conservation['conservation_error']:.4e}")
    else:
        conservation = {'delta_KE': 0, 'total_work': 0, 'conservation_error': 0}

    # --- Heating profile ---
    nx, ny = state['nx'], state['ny']
    dx, dy = state['dx'], state['dy']
    Ex = state.get('Ex', np.zeros((nx, ny)))
    Ey = state.get('Ey', np.zeros((nx, ny)))

    P_heat = compute_heating_profile(
        xe, ye, vxe, vye, vze, Ex, Ey,
        nx, ny, dx, dy, PC.m_e, PC.q_e
    )

    return {
        'eedf': eedf,
        'eedf_bins': eedf_bins,
        'mean_Te_eV': mean_Te_eV,
        'energy_conservation': conservation,
        'P_heat': P_heat,
        'KE_history': KE_history,
    }
