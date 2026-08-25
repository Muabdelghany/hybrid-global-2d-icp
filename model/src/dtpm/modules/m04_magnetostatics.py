"""
Module 04 — Magnetostatic Field Computation
=============================================
Computes static magnetic field from ICP coil currents using the exact
axisymmetric solution for circular current loops via elliptic integrals.

Physics:
- Each coil is a circular loop of radius `a` at axial position `z_coil`
- The grid is interpreted as (r, z) cylindrical coordinates
- B_r and B_z are computed from complete elliptic integrals K(k) and E(k):
    B_r(r,z) = (mu_0*I / 2*pi) * (z-zc) / (r * sqrt((a+r)^2+(z-zc)^2))
               * [-K(k^2) + (a^2+r^2+(z-zc)^2)/((a-r)^2+(z-zc)^2) * E(k^2)]
    B_z(r,z) = (mu_0*I / 2*pi) * 1/sqrt((a+r)^2+(z-zc)^2)
               * [K(k^2) + (a^2-r^2-(z-zc)^2)/((a-r)^2+(z-zc)^2) * E(k^2)]
  where k^2 = 4*a*r / ((a+r)^2 + (z-zc)^2)
- On-axis (r=0): B_r=0, B_z = mu_0*I*a^2 / (2*(a^2+(z-zc)^2)^(3/2))
- Fields from all coils are superposed

References:
    - Jackson, Classical Electrodynamics, 3rd ed., Section 5.5
    - Smythe, Static and Dynamic Electricity, Chapter 7

Implementation notes:
- Axisymmetric field from the exact elliptic-integral solution
- Fully vectorized over grid points, with no nested Python loops
"""

import os
import numpy as np
import logging
from scipy.special import ellipk, ellipe

logger = logging.getLogger(__name__)


def compute_biot_savart(coil_centers_idx, coil_radius_idx, nx, ny, dx, dy,
                        mu_0, I_peak, n_segments=36):
    """
    Compute static magnetic field using axisymmetric elliptic integral solution.

    The grid is interpreted as (r, z) cylindrical coordinates where:
    - x-axis → radial direction r
    - y-axis → axial direction z

    Each coil center (ic, jc) defines a circular loop at r_coil = ic*dx, z_coil = jc*dy
    with radius a = coil_radius_idx * dx.

    Args:
        coil_centers_idx: List of (i, j) coil center grid indices.
        coil_radius_idx: Coil radius in grid points.
        nx, ny: Grid dimensions.
        dx, dy: Grid spacing [m].
        mu_0: Vacuum permeability [H/m].
        I_peak: Peak coil current [A].
        n_segments: Unused (kept for API compatibility).

    Returns:
        Br, Bz: Magnetic field components [T] on (nx, ny) grid.
        B_mag: Magnetic field magnitude [T].
    """
    Br = np.zeros((nx, ny))
    Bz = np.zeros((nx, ny))

    # Create coordinate arrays (r, z) for the full grid
    r_grid = np.arange(nx) * dx  # radial positions [m]
    z_grid = np.arange(ny) * dy  # axial positions [m]
    R, Z = np.meshgrid(r_grid, z_grid, indexing='ij')  # (nx, ny)

    for (ic, jc) in coil_centers_idx:
        # Coil parameters in physical units
        a = coil_radius_idx * dx   # coil radius [m]
        z_coil = jc * dy           # axial position of coil [m]

        # Distance from coil plane
        dz = Z - z_coil  # (nx, ny)

        # Compute k^2 = 4*a*r / ((a+r)^2 + dz^2)
        r_plus_a = R + a
        r_minus_a = R - a
        denom_plus = r_plus_a**2 + dz**2   # (a+r)^2 + (z-zc)^2
        denom_minus = r_minus_a**2 + dz**2  # (a-r)^2 + (z-zc)^2

        # Avoid division by zero at the wire location
        denom_plus = np.maximum(denom_plus, (0.01 * dx)**2)
        denom_minus = np.maximum(denom_minus, (0.01 * dx)**2)

        k_sq = 4.0 * a * R / denom_plus
        # Clamp k^2 to [0, 1) to avoid singularity in elliptic integrals
        k_sq = np.clip(k_sq, 0.0, 1.0 - 1e-12)

        K_val = ellipk(k_sq)  # Complete elliptic integral of first kind
        E_val = ellipe(k_sq)  # Complete elliptic integral of second kind

        sqrt_denom = np.sqrt(denom_plus)
        prefactor = mu_0 * I_peak / (2.0 * np.pi)

        # ----- On-axis points (r ~ 0): use exact on-axis formula -----
        on_axis = R < 0.5 * dx  # points very close to r=0

        # ----- Off-axis: full elliptic integral solution -----
        # B_z = (mu_0*I / 2*pi) * 1/sqrt((a+r)^2+dz^2) * [K + (a^2-r^2-dz^2)/((a-r)^2+dz^2) * E]
        Bz_contrib = prefactor / sqrt_denom * (
            K_val + (a**2 - R**2 - dz**2) / denom_minus * E_val
        )

        # B_r = (mu_0*I / 2*pi) * dz / (r * sqrt((a+r)^2+dz^2))
        #        * [-K + (a^2+r^2+dz^2)/((a-r)^2+dz^2) * E]
        # Safe division by r (avoid r=0)
        r_safe = np.where(on_axis, 1.0, R)  # placeholder for on-axis
        Br_contrib = prefactor * dz / (r_safe * sqrt_denom) * (
            -K_val + (a**2 + R**2 + dz**2) / denom_minus * E_val
        )

        # On-axis: B_r = 0, B_z = mu_0*I*a^2 / (2*(a^2+dz^2)^(3/2))
        Bz_on_axis = mu_0 * I_peak * a**2 / (
            2.0 * np.maximum(a**2 + dz**2, (0.01 * dx)**2)**1.5
        )
        Br_contrib = np.where(on_axis, 0.0, Br_contrib)
        Bz_contrib = np.where(on_axis, Bz_on_axis, Bz_contrib)

        Br += Br_contrib
        Bz += Bz_contrib

    B_mag = np.sqrt(Br**2 + Bz**2)

    logger.info(f"Magnetostatic field computed: {len(coil_centers_idx)} coils, "
                f"|B|_max = {B_mag.max():.4e} T")
    logger.info(f"  Br range: [{Br.min():.4e}, {Br.max():.4e}] T")
    logger.info(f"  Bz range: [{Bz.min():.4e}, {Bz.max():.4e}] T")

    return Br, Bz, B_mag


def on_axis_bz_analytical(z_points, coil_z, coil_radius, mu_0, I_peak):
    """
    Analytical on-axis (r=0) B_z for a single circular loop.

    B_z(0, z) = mu_0 * I * a^2 / (2 * (a^2 + (z - z_c)^2)^(3/2))

    Args:
        z_points: Array of axial positions [m].
        coil_z: Axial position of coil center [m].
        coil_radius: Coil radius [m].
        mu_0: Vacuum permeability [H/m].
        I_peak: Peak current [A].

    Returns:
        Bz_analytical: Analytical B_z values [T].
    """
    a = coil_radius
    dz = z_points - coil_z
    return mu_0 * I_peak * a**2 / (2.0 * (a**2 + dz**2)**1.5)


def plot_magnetostatic_results(Bx, By, B_mag, nx, ny, dx, dy, save_folder):
    """Generate magnetostatic field plots."""
    from ..utils.plotting import (plot_results, plot_vector_field,
                                   plot_field_streamlines)

    os.makedirs(save_folder, exist_ok=True)
    x_mm = np.linspace(0, (nx - 1) * dx * 1e3, nx)
    y_mm = np.linspace(0, (ny - 1) * dy * 1e3, ny)

    # Magnitude contour
    plot_results(B_mag, 'B_static_magnitude', save_folder=save_folder,
                 xlabel='r [mm]', ylabel='z [mm]',
                 title='Static Magnetic Field Magnitude',
                 x_data=x_mm, y_data=y_mm, cmap='jet',
                 colorbar_label='|B| [T]')

    # Vector field
    plot_vector_field(Bx, By, 'B_static_vectors',
                      'Static Magnetic Field', x_mm, y_mm, dx, dy,
                      field_type='magnetostatic', save_folder=save_folder)

    # Streamlines
    plot_field_streamlines(Bx, By, 'B_static_streamlines',
                           'Magnetic Field Lines', x_mm, y_mm,
                           field_type='magnetostatic', save_folder=save_folder)


# --- Pipeline interface ---

def run(state, config):
    """Pipeline-compatible entry point for M04."""
    from ..core.units import PhysicalConstants as PC

    nx = state['nx']
    ny = state['ny']
    dx = state['dx']
    dy = state['dy']
    coil_centers_idx = state.get('coil_centers_idx', [])
    coil_radius_idx = state.get('coil_radius_idx', 1)
    I_peak = state['I_peak']

    Br, Bz, B_mag = compute_biot_savart(
        coil_centers_idx, coil_radius_idx,
        nx, ny, dx, dy, PC.mu_0, I_peak, n_segments=36
    )

    # Return as Bs_x (radial) and Bs_y (axial) for pipeline compatibility
    return {'Bs_x': Br, 'Bs_y': Bz, 'Bs_mag': B_mag}
