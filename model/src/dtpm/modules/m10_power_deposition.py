"""
Module 10 — Power Deposition (Self-Consistent Ohmic Integral)
=============================================================
Computes spatially resolved Ohmic heating from FDTD E-fields:

    P(r,z) = 0.5 * sigma(r,z) * |E_theta_rms(r,z)|^2
    P_abs  = integral( P(r,z) dV )  over the active plasma domain

where sigma = e^2 * ne / (m_e * nu_m) is the plasma conductivity and
nu_m is the electron-neutral momentum-transfer collision frequency.

Architecture note (Phase-1 self-consistent revision)
----------------------------------------------------
This module used to rescale the FDTD E-field magnitude every Picard
iteration so that the Ohmic integral matched a prescribed target
`eta_initial * P_rf`.  That produced a tautological `eta_computed == eta_initial`
and invalidated absolute-magnitude benchmarks (see
docs/CODE_REVIEW_ULTRAREVIEW.md sections A and B).

The rescaling has been removed.  The FDTD's E-field is now consumed
*unmodified*: its magnitude is set by the physical coil current I_peak
(computed by m01 from P_rf, R_coil, R_plasma) and its spatial shape by
the Maxwell solve.  The Ohmic integral therefore returns the physically
correct P_abs, and eta = P_abs / P_rf emerges as an observable, not a
dial.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_power_deposition(E_theta_rms, ne, Te, mesh, inside, config):
    """Compute spatially resolved Ohmic power deposition.

    Parameters
    ----------
    E_theta_rms : ndarray (Nr, Nz)
        RMS azimuthal electric field from FDTD [V/m].  Absolute magnitude
        is assumed physically correct (set by the FDTD coil-current drive).
    ne : ndarray (Nr, Nz)
        Electron density [m^-3].
    Te : ndarray (Nr, Nz) or float
        Electron temperature [eV].
    mesh : Mesh2D
    inside : ndarray (Nr, Nz), bool
    config : SimulationConfig

    Returns
    -------
    dict with P_rz, P_abs, eta_computed, sigma_rz.
    """
    from scipy.constants import e as eC, m_e, k as kB

    oper = config.operating if hasattr(config, 'operating') else config.get('operating', {})
    circ = config.circuit if hasattr(config, 'circuit') else config.get('circuit', {})

    p_mTorr = oper.get('pressure_mTorr', 10)
    Tgas = oper.get('Tgas', 313)
    P_rf = circ.get('source_power', 700)

    # Gas density
    p_Pa = p_mTorr * 0.133322
    ng = p_Pa / (kB * Tgas)

    Nr, Nz = mesh.Nr, mesh.Nz

    # Scalar Te handling
    if np.isscalar(Te):
        Te_arr = np.full((Nr, Nz), Te)
    else:
        Te_arr = Te.copy()
    Te_arr = np.clip(Te_arr, 0.3, 20.0)

    E_rms = E_theta_rms  # consumed as-is; absolute magnitude is physical

    # Elastic collision rate for SF6: k_el = 2.8e-7 * exp(-1.5/Te) [cm^3/s]
    el_rate = 2.8e-7 * np.exp(-1.5 / Te_arr) * 1e-6  # m^3/s
    nu_m = ng * el_rate  # momentum-transfer frequency [s^-1]

    # Plasma conductivity
    sigma_rz = np.zeros((Nr, Nz))
    active = inside & (ne > 1e8) & (nu_m > 0)
    sigma_rz[active] = eC**2 * ne[active] / (m_e * nu_m[active])

    # Ohmic deposition
    P_rz = 0.5 * sigma_rz * E_rms**2

    # Total absorbed power (observable)
    P_abs = float(np.sum(P_rz * mesh.vol * inside))
    eta_computed = P_abs / P_rf if P_rf > 0 else 0.0

    logger.info(
        f"Power deposition: P_abs={P_abs:.1f}W, eta={eta_computed:.3f}, "
        f"P_max={P_rz.max():.2e}W/m^3"
    )

    return {
        'P_rz': P_rz,
        'P_abs': P_abs,
        'eta_computed': eta_computed,
        'sigma_rz': sigma_rz,
    }


def run(state, config):
    """Pipeline-compatible entry point for M10."""
    E_theta_rms = state.get('E_theta_rms')
    ne = state.get('ne')
    Te = state.get('Te', 3.0)
    mesh = state['mesh']
    inside = state['inside']

    if E_theta_rms is None:
        logger.warning("M10: No E_theta_rms in state — skipping power deposition")
        return {}

    if ne is None:
        logger.warning("M10: No ne in state — using uniform 1e16 m^-3")
        ne = np.where(inside, 1e16, 0.0)

    return compute_power_deposition(E_theta_rms, ne, Te, mesh, inside, config)
