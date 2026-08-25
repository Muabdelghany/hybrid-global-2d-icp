"""
Module 12 — Capacitive Wafer-Bias Sheath (Lieberman reduced-order)
===================================================================
Adds a wafer-bias sheath source term to the Phase-1 Global–2D framework,
modelling the plasma-volume expansion observed when RF bias is applied to
the wafer electrode (Mettler 2025, Fig 4.13 / 4.17).

Physical picture
----------------
At the wafer, an RF-biased sheath accelerates positive ions toward the
wafer surface.  The ion energy (~V_dc) is deposited as kinetic energy on
the wafer and, via ion-neutral elastic collisions near the wafer,
transferred to the neutral gas.  A fraction f_ion_to_gas ~ 0.5 (Turner
2013) of the ion power becomes local neutral-gas heating, which reduces
the local gas density (pV = n k T at fixed pressure), increases the
electron mean free path, and effectively extends the ionisation zone
downward into the process chamber.  Mettler observed the [F] density at
the wafer rising by a factor 1.6 (90% SF6) to 2.15 (30% SF6 centre).

Reduced-order model
-------------------
Rather than solving a full capacitive-sheath + gas-energy PDE (that is
Phase-3 scope), this module (i) closes the bias-sheath power balance to
find V_dc and the ion heating power, and (ii) adds an EFFECTIVE volumetric
ionisation source in the process chamber that, when combined with the
ambipolar diffusion solver, reproduces Mettler's wafer-region [F]
enhancement.  A single free parameter lambda_exp scales the effective
expansion source; it is calibrated against Mettler Fig 4.17 90% SF6
bias-on centre [F] and blind-tested against the 30% SF6 branch.

Power balance
-------------
Step 1: Bohm ion flux at the sheath edge (just above the wafer):
    Gamma_i = 0.61 * n_e_wafer * sqrt( k_B * T_e / m_i )

Step 2: Sheath DC voltage from bias power balance:
    P_bias = e * V_dc * Gamma_i * A_wafer
    => V_dc = P_bias / ( e * Gamma_i * A_wafer )

Step 3: Neutral-gas heating power:
    P_ion_to_gas = f_ion_to_gas * P_bias

Step 4: Effective volumetric source in the process chamber:
    S_bias(r,z) = lambda_exp * P_ion_to_gas * shape_bias(r,z) / (eps_T * e)

    shape_bias(r,z) = exp(-(z_apt - z)/L_exp) * I(inside process chamber)

See docs/ASSUMED_PARAMETERS.md for lambda_exp / L_exp / f_ion_to_gas values.

This module is bypassed entirely when `config.bias.enabled = False` or
when `bias.P_bias_W = 0`.
"""
import logging
import numpy as np
from scipy.constants import k as kB, e as eC

logger = logging.getLogger(__name__)

# Dominant ion mass by composition regime (amu) [LIT: Lallement 2009]
# At SF6-rich plasma the dominant positive ion is SF5+ (~127 amu).
# At Ar-rich plasma the dominant positive ion is Ar+ (40 amu).
AMU_TO_KG = 1.66053906660e-27


def _effective_ion_mass_amu(frac_Ar, sf_dominant_amu=127.0, ar_amu=40.0):
    """Composition-weighted effective ion mass for the Bohm flux.

    At frac_Ar = 0.0 we use the SF5+ mass; at frac_Ar = 1.0 we use Ar+.
    Intermediate compositions use a linear weighting by the Ar fraction.
    """
    return (1.0 - frac_Ar) * sf_dominant_amu + frac_Ar * ar_amu


def _bohm_flux(ne_wafer, Te_wafer_eV, m_i_kg):
    """Bohm ion flux [m^-2 s^-1].

    Gamma_i = 0.61 * n_e_wafer * sqrt( k_B * T_e / m_i )
    with T_e in eV (converted to K via eV_to_K = e / kB).
    """
    T_e_K = Te_wafer_eV * eC / kB  # convert eV -> K
    u_B = np.sqrt(kB * T_e_K / max(m_i_kg, 1e-30))
    return 0.61 * ne_wafer * u_B


def _shape_bias_field(mesh, inside, reactor_geom, L_exp_m, R_wafer_m):
    """Build the normalised spatial shape function shape_bias(r,z).

    shape_bias is non-zero only inside the process chamber (z < L_proc +
    L_apt) and decays exponentially with (z_apt_top - z) / L_exp.  It is
    further restricted to r <= R_wafer_m (a cosine taper at the wafer
    edge rather than a hard cutoff to avoid discretisation artefacts).

    Returns
    -------
    shape : (Nr, Nz) array, normalised so that integral(shape * dV) = 1.
    """
    Nr, Nz = mesh.Nr, mesh.Nz
    L_proc = reactor_geom.get('L_proc', 0.050)
    L_apt = reactor_geom.get('L_apt', 0.002)
    z_apt_top = L_proc + L_apt

    rc = mesh.rc[:, None]     # (Nr, 1)
    zc = mesh.zc[None, :]     # (1, Nz)
    # Axial decay from the aperture downward toward the wafer
    dz_from_apt = np.clip(z_apt_top - zc, 0.0, None)
    # Radial cosine taper from r=0 to r=R_wafer
    r_norm = np.clip(rc / max(R_wafer_m, 1e-6), 0.0, 1.0)
    radial = 0.5 * (1.0 + np.cos(np.pi * r_norm))  # smooth cos taper
    axial = np.exp(-dz_from_apt / max(L_exp_m, 1e-6))

    shape = radial * axial
    # Mask to process-chamber region (below aperture top)
    proc_mask = inside & (zc < z_apt_top)
    shape = np.where(proc_mask, shape, 0.0)

    # Normalise so integral = 1
    norm = float(np.sum(shape * mesh.vol * inside))
    if norm > 1e-30:
        shape = shape / norm
    else:
        logger.warning("m12: shape_bias normalisation failed (norm=%.2e)", norm)
    return shape


def compute_bias_sheath(ne, Te, mesh, inside, config):
    """Compute the wafer-bias sheath power balance and volumetric source.

    Parameters
    ----------
    ne : ndarray (Nr, Nz)
        Electron density field [m^-3].  Wafer-local value is sampled at
        z index 0 along r for the Bohm flux.
    Te : ndarray (Nr, Nz) or float
        Electron temperature [eV].
    mesh, inside, config : standard Phase-1 objects.

    Returns
    -------
    dict with:
        enabled     : whether the bias model contributed (bool)
        V_dc        : time-averaged sheath DC voltage [V]
        Gamma_i     : wafer-averaged Bohm ion flux [m^-2 s^-1]
        P_bias      : input bias power [W]
        P_ion_to_gas: ion-heating power deposited as gas heating [W]
        P_rz_bias   : (Nr, Nz) additional power-deposition density [W/m^3]
                      used as the expansion source fed to the ambipolar solver.
        shape_bias  : (Nr, Nz) normalised spatial shape of P_rz_bias.
        lambda_exp  : calibration factor (for logging / sweep tracking).
    """
    bias = config.bias if hasattr(config, 'bias') else config.get('bias', {})
    rgeom = config.reactor_geometry if hasattr(config, 'reactor_geometry') else config.get('reactor_geometry', {})
    oper = config.operating if hasattr(config, 'operating') else config.get('operating', {})

    enabled = bool(bias.get('enabled', False))
    P_bias = float(bias.get('P_bias_W', 0.0))

    Nr, Nz = mesh.Nr, mesh.Nz
    result_inactive = {
        'enabled': False, 'V_dc': 0.0, 'Gamma_i': 0.0,
        'P_bias': 0.0, 'P_ion_to_gas': 0.0,
        'P_rz_bias': np.zeros((Nr, Nz)),
        'shape_bias': np.zeros((Nr, Nz)),
        'lambda_exp': 0.0,
    }
    if (not enabled) or (P_bias <= 0):
        logger.info("m12: bias sheath DISABLED (bias.enabled=%s, P_bias=%g W)",
                    enabled, P_bias)
        return result_inactive

    f_ion_to_gas = float(bias.get('f_ion_to_gas', 0.5))     # LIT: Turner 2013
    A_wafer = float(bias.get('A_wafer_eff', 0.01767))       # pi*(0.075)^2
    L_exp_m = float(bias.get('L_exp_mm', 30.0)) * 1e-3
    lambda_exp = float(bias.get('lambda_exp', 1.6))         # calibrated free param
    # Energy-conserving alternative closure (bias.closure = 'f_e_bias'):
    # a single electron-heating fraction <= 1 replaces lambda_exp*f_ion_to_gas.
    closure = str(bias.get('closure', 'lambda_exp'))
    f_e_bias = min(float(bias.get('f_e_bias', 0.5)), 1.0)
    m_ion_amu_dom = float(bias.get('m_ion_amu_dominant', 127.0))  # SF5+
    frac_Ar = float(oper.get('frac_Ar', 0.0))

    # Effective ion mass — composition-weighted
    m_i_amu = _effective_ion_mass_amu(frac_Ar, sf_dominant_amu=m_ion_amu_dom)
    m_i_kg = m_i_amu * AMU_TO_KG

    # Wafer-surface values: z index 0, averaged over inside cells
    wafer_row_mask = inside[:, 0]
    if not np.any(wafer_row_mask):
        logger.warning("m12: no inside cells at wafer row z=0 — skipping bias")
        return result_inactive

    ne_wafer = float(np.mean(ne[:, 0][wafer_row_mask]))
    if np.isscalar(Te):
        Te_wafer = float(Te)
    else:
        Te_wafer = float(np.mean(Te[:, 0][wafer_row_mask]))
    Te_wafer = max(Te_wafer, 0.3)  # floor to avoid sqrt(negative/0) drama

    # Bohm flux and sheath voltage
    Gamma_i = _bohm_flux(ne_wafer, Te_wafer, m_i_kg)
    # V_dc from power balance: P_bias = e * V_dc * Gamma_i * A_wafer
    denom = eC * max(Gamma_i, 1e-30) * A_wafer
    V_dc = P_bias / denom if denom > 0 else 0.0

    # Fraction of ion kinetic energy deposited as neutral heating
    P_ion_to_gas = f_ion_to_gas * P_bias

    # Effective expansion source in the process chamber
    R_wafer_m = float(np.sqrt(A_wafer / np.pi))
    shape_bias = _shape_bias_field(mesh, inside, rgeom, L_exp_m, R_wafer_m)

    # Total expansion source power (W).  lambda_exp is the calibration
    # knob: larger lambda_exp -> more effective plasma-glow expansion.
    if closure == 'f_e_bias':
        # energy-conserving: never more watts than the supply delivers
        P_bias_total = f_e_bias * P_bias
    else:
        P_bias_total = lambda_exp * P_ion_to_gas
    P_rz_bias = P_bias_total * shape_bias  # [W/m^3] array

    logger.info(
        "m12 bias: P_bias=%.1f W, Gamma_i=%.2e m^-2 s^-1, V_dc=%.1f V, "
        "P_ion_to_gas=%.1f W, lambda_exp=%.2f, eff. source=%.1f W",
        P_bias, Gamma_i, V_dc, P_ion_to_gas, lambda_exp, P_bias_total,
    )

    return {
        'enabled': True,
        'V_dc': float(V_dc),
        'Gamma_i': float(Gamma_i),
        'P_bias': float(P_bias),
        'P_ion_to_gas': float(P_ion_to_gas),
        'P_rz_bias': P_rz_bias,
        'shape_bias': shape_bias,
        'lambda_exp': float(lambda_exp),
        'closure': closure, 'f_e_bias': f_e_bias,
        'ne_wafer': ne_wafer,
        'Te_wafer': Te_wafer,
        'm_i_amu': m_i_amu,
        'R_wafer_m': R_wafer_m,
    }


def run(state, config):
    """Pipeline-compatible wrapper.

    Called from m11's Picard loop between the ne-diffusion and chemistry
    steps.  Returns a dict that m11 merges into state for downstream
    solvers.  When `bias.enabled = False` the function is still safe to
    call — it returns zeros.
    """
    mesh = state['mesh']
    inside = state['inside']
    ne = state.get('ne')
    Te = state.get('Te', 3.0)
    if ne is None:
        logger.warning("m12: no ne in state — skipping bias sheath")
        return {}
    return compute_bias_sheath(ne, Te, mesh, inside, config)
