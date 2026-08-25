"""
Ambipolar diffusion solver for ne(r,z).

Solves the steady-state electron continuity equation with an explicit
ionization source from the FDTD power deposition:

    div(D_a * grad(ne)) + S_iz(r,z) - nu_loss * ne = 0

where:
  S_iz(r,z) = P(r,z) / (eps_T * e)   — ionization source from EM heating
  nu_loss   = u_B * A_eff / V          — Bohm wall loss frequency
  D_a       = D_i * (1 + Te/Ti) * (1+alpha) / (1+alpha*Ti/Te)
                                       — electronegative ambipolar coefficient
                                         (Lieberman & Lichtenberg 2005 §10.3)

The electronegative correction factor (1+alpha)/(1+alpha*Ti/Te) reduces to
unity when alpha = 0, recovering the electropositive expression exactly.
This implementation is the L1 resolution documented in
docs/report/main.tex §5.x and is rigorous physics — no free parameters.
The derivation is set out in the manuscript, section 3.

Physics: electrons are PRODUCED at the skin depth (wall-peaked source)
but DIFFUSE inward via ambipolar transport. The resulting ne profile is
centre-peaked (Bessel-like) because diffusion smooths the source profile
and wall losses deplete ne at boundaries.

Robin BCs at walls: D_a * dne/dn = -u_B * ne (Bohm loss).
Neumann BC at axis: dne/dr = 0 (symmetry).
"""

import numpy as np
import logging
from scipy.sparse.linalg import spsolve
from scipy import sparse
from scipy.constants import e as eC, k as kB, pi, m_e
from scipy.special import j0

from .species_transport import build_diffusion_matrix, field_to_flat, flat_to_field

logger = logging.getLogger(__name__)


def prescribe_bessel_cosine(ne_avg, mesh, inside, R_icp, L_proc, L_apt, L_icp):
    """Prescribed Bessel-cosine ne(r,z) profile (Stage 10 warm-start).

    Used as the initial guess before the PDE solver converges.
    """
    Nr, Nz = mesh.Nr, mesh.Nz
    z_apt_top = L_proc + L_apt
    shape = np.zeros((Nr, Nz))

    for i in range(Nr):
        r = mesh.rc[i]
        for j in range(Nz):
            z = mesh.zc[j]
            if not inside[i, j]:
                continue
            if z >= z_apt_top and r <= R_icp:
                bessel = j0(2.405 * r / R_icp)
                cosine = np.cos(pi * (z - z_apt_top) / (2 * L_icp))
                shape[i, j] = max(bessel * cosine, 0.01)
            elif z >= L_proc:
                frac = (z - L_proc) / max(L_apt, 1e-3)
                shape[i, j] = 0.20 + 0.20 * 0.5 * (1 + np.cos(pi * (1 - frac)))
            elif r <= R_icp:
                dist = L_proc - z
                shape[i, j] = 0.20 * np.exp(-dist / 0.012) + 0.01
            else:
                dist = L_proc - z
                r_dist = r - R_icp
                shape[i, j] = 0.20 * np.exp(-dist / 0.012) * np.exp(-r_dist / 0.020) + 0.01

    shape_avg = np.sum(shape * mesh.vol * inside) / max(np.sum(mesh.vol * inside), 1e-30)
    if shape_avg > 0:
        shape *= ne_avg / shape_avg

    return shape


def solve_ne_ambipolar(Te, P_abs, mesh, inside, bc_type, ij_to_flat, flat_to_ij,
                       n_active, config, P_rz=None, ne_init=None, alpha=0.0,
                       nSF6_field=None, ec_clip=(80.0, 400.0)):
    """Solve the ionization-source diffusion equation for ne(r,z).

    Solves:  div(D_a * grad(ne)) + S_iz(r,z) - nu_loss * ne = 0

    In matrix form: (A - diag(nu_loss)) * ne = -S_iz
    This is a single linear solve — no iteration needed.

    Parameters
    ----------
    Te : ndarray (Nr, Nz)
        Electron temperature field [eV].
    P_abs : float
        Total absorbed power [W] (for normalisation check).
    mesh : Mesh2D
    inside, bc_type, ij_to_flat, flat_to_ij, n_active : geometry data
    config : SimulationConfig
    P_rz : ndarray (Nr, Nz) or None
        Spatially resolved power deposition [W/m^3] from FDTD.
        If None, uses a prescribed Gaussian profile in the ICP region.
    ne_init : ndarray or None
        Not used in this solver (kept for API compatibility).
    alpha : float, optional (default = 0.0)
        Electronegativity n_-/n_e from the 0D global model.  When alpha = 0
        the electropositive form is recovered exactly (backwards-compat).
        When alpha > 0 the Lieberman 2005 §10.3 electronegative ambipolar
        correction factor is applied: D_a^en = D_a^ep * (1+alpha)/(1+alpha*Ti/Te).
        This is L1 resolution — rigorous physics, not a calibration.

    Returns
    -------
    ne : ndarray (Nr, Nz)
        Electron density [m^-3].
    ne_avg : float
        Volume-averaged ne [m^-3].
    """
    from ..chemistry.sf6_rates import rates

    rgeom = config.reactor_geometry if hasattr(config, 'reactor_geometry') else config.get('reactor_geometry', {})
    oper = config.operating if hasattr(config, 'operating') else config.get('operating', {})

    R_icp = rgeom.get('R_icp', 0.038)
    L_proc = rgeom.get('L_proc', 0.050)
    L_apt = rgeom.get('L_apt', 0.002)
    L_icp = rgeom.get('L_icp', 0.1815)
    z_apt_top = L_proc + L_apt

    p_mTorr = oper.get('pressure_mTorr', 10)
    Tgas = oper.get('Tgas', 313)
    frac_Ar = oper.get('frac_Ar', 0.0)

    p_Pa = p_mTorr * 0.133322
    ng = p_Pa / (kB * Tgas)
    nSF6 = ng * (1 - frac_Ar)

    Nr, Nz = mesh.Nr, mesh.Nz

    # -- Ambipolar diffusion coefficient (L1 — Lieberman 2005 §10.3) --
    #
    # Electronegative form:
    #   D_a^en = D_i * (1 + T_e/T_i) * (1 + alpha) / (1 + alpha * T_i/T_e)
    #
    # The factor on the right is the electronegative correction; it reduces
    # to 1 when alpha = 0, recovering the electropositive expression
    # D_a^ep = D_i * (1 + T_e/T_i).  No free parameters are introduced.
    # The derivation and its provenance are in the manuscript, section 3.
    M_ion = 127.06 * 1.66054e-27  # SF5+ mass [kg]
    sigma_in = 5e-19  # ion-neutral cross-section [m^2]
    v_th_ion = np.sqrt(8 * kB * Tgas / (pi * M_ion))
    nu_in = ng * sigma_in * v_th_ion
    D_i_base = kB * Tgas / (M_ion * nu_in) if nu_in > 0 else 0.01
    Ti_eV = Tgas * kB / eC

    # Vectorised D_a with electronegative correction.
    # `alpha` may be a scalar (baseline L1 path) OR a 2D (Nr, Nz) field
    # (Path D — electronegative-ambipolar with spatial resolution). The
    # broadcast below handles both cases transparently.
    Te_safe = np.where(inside, np.maximum(Te, 0.5), 0.5)
    Ti_safe = max(Ti_eV, 0.01)
    D_a_ep = D_i_base * (1.0 + Te_safe / Ti_safe)
    alpha_arr = np.asarray(alpha)  # shape () for scalar, (Nr,Nz) for 2D
    correction = (1.0 + alpha_arr) / (1.0 + alpha_arr * Ti_safe / Te_safe)
    D_a = np.where(inside, D_a_ep * correction, 0.0)

    # Diagnostic scalar summary — works for both scalar and 2D alpha
    if alpha_arr.ndim == 0:
        alpha_log = float(alpha_arr)
    else:
        alpha_log = float(np.mean(alpha_arr[inside])) if np.any(inside) else 0.0
    Te_mean = float(np.mean(Te_safe[inside])) if np.any(inside) else 0.5
    corr_log = float(np.mean(correction[inside])) if np.any(inside) else 1.0
    logger.info(
        "Ambipolar D_a: alpha_mean=%.3f, Te_avg=%.2f eV, Ti=%.3f eV, "
        "correction_mean=%.3f, D_a_max=%.2e m^2/s",
        alpha_log, Te_mean, Ti_safe, corr_log,
        D_a.max() if D_a.size else 0.0,
    )

    # -- Bohm velocity for wall loss BC --
    Te_avg = np.mean(Te[inside]) if np.any(inside) else 3.0
    u_B = np.sqrt(eC * Te_avg / M_ion)

    # Robin BC: D_a * dne/dn = -u_B * ne at walls
    # In build_diffusion_matrix: h = gamma * v_th / 4
    # We need h = u_B, so set gamma = 1 and v_th = 4*u_B
    from ..core.geometry import (BC_QUARTZ, BC_WINDOW, BC_AL_SIDE, BC_AL_TOP,
                                  BC_WAFER, BC_SHOULDER, BC_AXIS)
    gamma_ne_map = {
        BC_QUARTZ: 1.0, BC_WINDOW: 1.0,
        BC_AL_SIDE: 1.0, BC_AL_TOP: 1.0,
        BC_WAFER: 1.0, BC_SHOULDER: 1.0,
        BC_AXIS: 0.0,
    }
    v_th_for_bc = 4.0 * u_B  # so that gamma*v_th/4 = u_B

    # Build diffusion matrix A (encodes diffusion + Robin wall loss)
    A = build_diffusion_matrix(mesh, inside, bc_type, ij_to_flat, flat_to_ij,
                                n_active, D_a, gamma_ne_map, v_th=v_th_for_bc)

    # -- Volume loss term --
    # nu_loss accounts for recombination losses in electronegative plasmas
    # In SF6: attachment rate ~ k_att * nSF6 * ne depletes electrons
    # Also ion-ion recombination: k_rec * n_pos * n_neg
    # Simplified: nu_loss ~ k_att * nSF6 (attachment frequency)
    nu_loss = np.zeros(n_active)
    for kk in range(n_active):
        i, j = flat_to_ij[kk]
        k = rates(max(Te[i, j], 0.5))
        # Attachment loss (dominant in SF6)
        # C6: use the LOCAL depleted nSF6 when the chemistry field is provided
        # (chemistry.depleted_attachment); the feed scalar ignores the 50-95%
        # depletion the 9-species solver computes and over-states this sink.
        if nSF6_field is None:
            _nsf6_loc = nSF6
        else:
            _nsf6_loc = max(float(nSF6_field[i, j]), 1e10)
        nu_loss[kk] = _nsf6_loc * k['att_SF6_total']

    # -- Ionization source S_iz(r,z) --
    # S_iz = P(r,z) / (eps_T * e) [electrons per m^3 per second]
    # This represents the rate at which NEW electrons are created by
    # the RF power deposition.
    S_iz = np.zeros(n_active)

    if P_rz is not None and np.any(P_rz > 0):
        # Use FDTD-derived power profile
        for kk in range(n_active):
            i, j = flat_to_ij[kk]
            k = rates(max(Te[i, j], 0.5))
            Riz = k['iz_SF6_total']
            Eloss = (16*k['iz18'] + 20*k['iz19'] + 20.5*k['iz20'] + 28*k['iz21']
                     + 37.5*k['iz22'] + 18*k['iz23'] + 29*k['iz24']
                     + 9.6*k['d1'] + 12.1*k['d2'] + 16*k['d3']
                     + 18.6*k['d4'] + 22.7*k['d5'])
            Ec = np.clip(Eloss / max(Riz, 1e-30), ec_clip[0], ec_clip[1])
            eiw = 0.5 * Te[i, j] * np.log(max(M_ion / (2*pi*m_e), 1))
            eps_T = Ec + eiw + 2 * Te[i, j]
            # S_iz = P / (eps_T * e) [m^-3 s^-1]
            S_iz[kk] = P_rz[i, j] / max(eps_T * eC, 1e-30)
    else:
        # Fallback: prescribed Gaussian source in the ICP region
        # Peaked at skin depth (~70% of R_icp) and upper part of ICP
        for kk in range(n_active):
            i, j = flat_to_ij[kk]
            r, z = mesh.rc[i], mesh.zc[j]
            if z >= z_apt_top and r <= R_icp:
                z_local = (z - z_apt_top) / L_icp
                r_norm = r / R_icp
                p_shape = (np.exp(-((r_norm - 0.7) / 0.25)**2)
                           * np.exp(-((z_local - 0.85) / 0.3)**2))
                k = rates(max(Te[i, j], 0.5))
                Riz = k['iz_SF6_total']
                Eloss = (16*k['iz18'] + 20*k['iz19'] + 20.5*k['iz20'] + 28*k['iz21']
                         + 37.5*k['iz22'] + 18*k['iz23'] + 29*k['iz24']
                         + 9.6*k['d1'] + 12.1*k['d2'] + 16*k['d3']
                         + 18.6*k['d4'] + 22.7*k['d5'])
                Ec = np.clip(Eloss / max(Riz, 1e-30), ec_clip[0], ec_clip[1])
                eiw = 0.5 * Te[i, j] * np.log(max(M_ion / (2*pi*m_e), 1))
                eps_T = Ec + eiw + 2 * Te[i, j]
                S_iz[kk] = p_shape * P_abs / max(eps_T * eC, 1e-30)
        # Normalise so total source matches P_abs
        S_total = 0.0
        for kk in range(n_active):
            i, j = flat_to_ij[kk]
            S_total += S_iz[kk] * mesh.vol[i, j]
        if S_total > 0:
            # We want integral(S_iz * eps_T * e * dV) = P_abs
            # Since eps_T varies, do a simple volume normalisation
            for kk in range(n_active):
                i, j = flat_to_ij[kk]
                k = rates(max(Te[i, j], 0.5))
                Riz = k['iz_SF6_total']
                Eloss = (16*k['iz18'] + 20*k['iz19'] + 20.5*k['iz20'] + 28*k['iz21']
                         + 37.5*k['iz22'] + 18*k['iz23'] + 29*k['iz24']
                         + 9.6*k['d1'] + 12.1*k['d2'] + 16*k['d3']
                         + 18.6*k['d4'] + 22.7*k['d5'])
                Ec = np.clip(Eloss / max(Riz, 1e-30), ec_clip[0], ec_clip[1])
                eiw = 0.5 * Te[i, j] * np.log(max(M_ion / (2*pi*m_e), 1))
                eps_T = Ec + eiw + 2 * Te[i, j]
                S_iz[kk] = P_rz[i, j] / max(eps_T * eC, 1e-30) if P_rz is not None else S_iz[kk]
            # Re-normalise fallback
            if P_rz is None:
                S_vol = sum(S_iz[kk] * mesh.vol[flat_to_ij[kk][0], flat_to_ij[kk][1]]
                            for kk in range(n_active))
                target_S = P_abs / (200 * eC)  # approximate eps_T ~ 200 eV
                if S_vol > 0:
                    S_iz *= target_S / S_vol

    # -- Solve: (A - diag(nu_loss)) * ne = -S_iz --
    loss_diag = sparse.diags(-nu_loss, format='csr')
    M = A + loss_diag
    ne_flat = np.asarray(spsolve(M, -S_iz), dtype=np.float64)
    ne_flat = np.maximum(ne_flat, 0.0)

    ne = flat_to_field(ne_flat, flat_to_ij, Nr, Nz, fill=0.0)

    # -- Diagnostics --
    ne_avg = np.sum(ne * mesh.vol * inside) / max(np.sum(mesh.vol * inside), 1e-30)
    ne_max = ne.max()

    # Check: where is ne peaked?
    i_max, j_max = np.unravel_index(np.argmax(ne), ne.shape)
    r_peak = mesh.rc[i_max] * 1e3 if i_max < Nr else 0
    z_peak = mesh.zc[j_max] * 1e3 if j_max < Nz else 0

    logger.info(f"Ambipolar diffusion: ne_avg={ne_avg:.2e}, ne_max={ne_max:.2e}, "
                f"peak at r={r_peak:.1f}mm z={z_peak:.1f}mm")

    return ne, float(ne_avg)
