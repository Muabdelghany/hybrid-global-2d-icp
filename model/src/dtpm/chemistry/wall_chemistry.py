"""
Boundary conditions for IPI ICP reactor — material-specific surface kinetics.

Based on Kokkoris et al., J. Phys. D 42, 055209 (2009) — Table 2.
Each wall surface has distinct adsorption, recombination, and
SF6 regeneration probabilities.
"""
from ..core.geometry import (
    BC_INTERIOR, BC_AXIS, BC_QUARTZ, BC_WINDOW,
    BC_AL_SIDE, BC_AL_TOP, BC_WAFER, BC_SHOULDER, BC_INACTIVE,
)

# ===================================================
# Surface kinetics parameters per material
# ===================================================
# Keys: s_F (F sticking), s_SFx (SFx sticking), p_fluor (fluorination),
#       p_wr (Eley-Rideal), p_FF (F+F recombination)

SURF_QUARTZ = dict(s_F=0.0015, s_SFx=0.0005, p_fluor=0.025, p_wr=0.007, p_FF=0.003)
SURF_AL     = dict(s_F=0.015,  s_SFx=0.008,  p_fluor=0.025, p_wr=0.010, p_FF=0.005)
SURF_SI     = dict(s_F=0.025,  s_SFx=0.001,  p_fluor=0.001, p_wr=0.001, p_FF=0.001)
SURF_WINDOW = dict(s_F=0.001,  s_SFx=0.0003, p_fluor=0.025, p_wr=0.005, p_FF=0.002)
SURF_NONE   = dict(s_F=0,      s_SFx=0,      p_fluor=0,     p_wr=0,     p_FF=0)

# Map boundary type -> surface kinetics
BC_TO_SURFACE = {
    BC_QUARTZ:   SURF_QUARTZ,
    BC_WINDOW:   SURF_WINDOW,
    BC_AL_SIDE:  SURF_AL,
    BC_AL_TOP:   SURF_AL,
    BC_WAFER:    SURF_SI,
    BC_SHOULDER: SURF_AL,
    BC_AXIS:     SURF_NONE,
    BC_INTERIOR: SURF_NONE,
    BC_INACTIVE: SURF_NONE,
}


def wall_sf6_regeneration(bc_type_ij, nF, nSF5, v_th_F, dr_or_dz):
    """Compute SF6 wall regeneration rate for a single cell.

    Parameters
    ----------
    bc_type_ij : int
        Boundary type of this cell.
    nF : float
        Local F atom density (m^-3).
    nSF5 : float
        Local SF5 density (m^-3).
    v_th_F : float
        F thermal speed (m/s).
    dr_or_dz : float
        Cell width perpendicular to the wall (m).

    Returns
    -------
    R_sf6 : float
        SF6 production rate (m^-3 s^-1).
    """
    surf = BC_TO_SURFACE.get(bc_type_ij, SURF_NONE)
    if surf['s_F'] == 0:
        return 0.0

    GF = nF * v_th_F / 4
    GS = nSF5 * v_th_F / 4

    denom_F = surf['s_F'] + surf['p_FF'] * surf['s_F'] + surf['p_fluor']
    theta_F = surf['s_F'] / max(denom_F, 1e-30)
    denom_S = surf['s_SFx'] + surf['p_fluor']
    theta_S = surf['s_SFx'] / max(denom_S, 1e-30)

    AV = 1.0 / max(dr_or_dz, 1e-6)

    R_fluor = surf['p_fluor'] * theta_S * GF * AV
    R_ER    = surf['p_wr']    * theta_F * GS * AV
    return R_fluor + R_ER


def wall_F_loss(bc_type_ij, nF, v_th_F, dr_or_dz):
    """Compute F atom wall loss rate for a single cell.

    Returns
    -------
    R_loss : float
        F loss rate (m^-3 s^-1).
    """
    surf = BC_TO_SURFACE.get(bc_type_ij, SURF_NONE)
    if surf['s_F'] == 0:
        return 0.0

    GF = nF * v_th_F / 4
    denom_F = surf['s_F'] + surf['p_FF'] * surf['s_F'] + surf['p_fluor']
    theta_F = surf['s_F'] / max(denom_F, 1e-30)

    AV = 1.0 / max(dr_or_dz, 1e-6)
    return (surf['s_F'] + surf['p_FF'] * theta_F) * GF * AV


def get_gamma_map(gamma_quartz=0.001, gamma_Al=0.18,
                  gamma_wafer=0.025, gamma_window=0.001):
    """Build Robin BC gamma map for the diffusion operator.

    Returns
    -------
    gamma_map : dict {BC_TYPE: gamma_value}
    """
    return {
        BC_QUARTZ:   gamma_quartz,
        BC_WINDOW:   gamma_window,
        BC_AL_SIDE:  gamma_Al,
        BC_AL_TOP:   gamma_Al,
        BC_WAFER:    gamma_wafer,
        BC_SHOULDER: gamma_Al,
        BC_AXIS:     0.0,
        BC_INTERIOR: 0.0,
    }
