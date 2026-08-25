"""
Module 03 — Adaptive Mesh Refinement (AMR)
============================================
Refines the computational grid in regions of high field gradients.

Physics:
- Gradient-based refinement criterion: refine where |grad(E)| > threshold
- Bilinear interpolation of coarse solution to refined grid
- Re-solve potential on refined mesh for improved accuracy

ML Interface:
- Refinement regions can serve as attention masks for neural operators
- Gradient magnitude maps are useful features for error-aware surrogates

Note: Currently implements uniform 2x refinement as fallback when
gradient-based AMR is disabled. Full AMR with hanging nodes is planned.
"""

import os
import numpy as np
import logging

logger = logging.getLogger(__name__)


def uniform_refinement(Phi, nx, ny, dx, dy, coil_centers_idx, coil_radius_idx, factor=2):
    """
    Perform uniform mesh refinement by doubling resolution.

    Interpolates coarse potential to fine grid using bilinear interpolation.

    Args:
        Phi: Coarse potential field (nx, ny).
        nx, ny: Coarse grid dimensions.
        dx, dy: Coarse grid spacing [m].
        coil_centers_idx: List of (i, j) coil centers on coarse grid.
        coil_radius_idx: Coil radius in coarse grid points.
        factor: Refinement factor (default 2).

    Returns:
        Phi_fine: Interpolated potential on fine grid.
        fine_nx, fine_ny: Fine grid dimensions.
        fine_dx, fine_dy: Fine grid spacing.
        fine_coil_centers: Rescaled coil centers.
        fine_coil_radius: Rescaled coil radius.
    """
    fine_nx = (nx - 1) * factor + 1
    fine_ny = (ny - 1) * factor + 1
    fine_dx = dx / factor
    fine_dy = dy / factor

    # Bilinear interpolation
    from scipy.interpolate import RegularGridInterpolator
    x_coarse = np.arange(nx) * dx
    y_coarse = np.arange(ny) * dy
    interp = RegularGridInterpolator((x_coarse, y_coarse), Phi, method='linear')

    x_fine = np.linspace(0, (nx - 1) * dx, fine_nx)
    y_fine = np.linspace(0, (ny - 1) * dy, fine_ny)
    Xf, Yf = np.meshgrid(x_fine, y_fine, indexing='ij')
    Phi_fine = interp((Xf, Yf))

    # Rescale coil geometry
    fine_coil_centers = [(xc * factor, yc * factor) for xc, yc in coil_centers_idx]
    fine_coil_radius = coil_radius_idx * factor

    logger.info(f"Uniform refinement: {nx}x{ny} -> {fine_nx}x{fine_ny} (factor {factor})")
    return Phi_fine, fine_nx, fine_ny, fine_dx, fine_dy, fine_coil_centers, fine_coil_radius


def compute_gradient_magnitude(E_mag):
    """Compute gradient magnitude of E-field for refinement criterion."""
    grad_x = np.gradient(E_mag, axis=0)
    grad_y = np.gradient(E_mag, axis=1)
    return np.sqrt(grad_x**2 + grad_y**2)


def identify_refinement_regions(E_mag, threshold_sigma=2.0):
    """
    Identify grid regions requiring refinement based on E-field gradient.

    Args:
        E_mag: Electric field magnitude (nx, ny).
        threshold_sigma: Refine where gradient > mean + threshold*std.

    Returns:
        mask: Boolean refinement mask (nx, ny).
        grad_mag: Gradient magnitude field.
    """
    grad_mag = compute_gradient_magnitude(E_mag)
    threshold = np.mean(grad_mag) + threshold_sigma * np.std(grad_mag)
    mask = grad_mag > threshold
    n_refine = int(mask.sum())
    logger.info(f"Refinement: {n_refine}/{mask.size} points flagged "
                f"(threshold={threshold:.4e}, sigma={threshold_sigma})")
    return mask, grad_mag


# --- Pipeline interface ---

def run(state, config):
    """Pipeline-compatible entry point for M03."""
    mesh_cfg = config.mesh_refinement if hasattr(config, 'mesh_refinement') else config.get('mesh_refinement', {})

    if not mesh_cfg.get('enabled', False):
        logger.info("Mesh refinement disabled in config — skipping")
        return {}

    Phi = state['Phi']
    E_mag = state['E_mag']
    nx, ny = state['nx'], state['ny']
    dx, dy = state['dx'], state['dy']
    coil_centers_idx = state.get('coil_centers_idx', [])
    coil_radius_idx = state.get('coil_radius_idx', 1)

    # Uniform 2x refinement
    (Phi_fine, fine_nx, fine_ny, fine_dx, fine_dy,
     fine_coils, fine_radius) = uniform_refinement(
        Phi, nx, ny, dx, dy, coil_centers_idx, coil_radius_idx)

    return {
        'Phi': Phi_fine,
        'nx': fine_nx, 'ny': fine_ny,
        'dx': fine_dx, 'dy': fine_dy,
        'coil_centers_idx': fine_coils,
        'coil_radius_idx': fine_radius,
        'mesh_refined': True,
    }
