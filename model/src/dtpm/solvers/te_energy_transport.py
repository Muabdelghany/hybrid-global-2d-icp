"""Electron energy transport solver: Te(r,z) from a conduction + convection PDE.

Motivation
----------
The legacy closure (`solve_Te_local_power_balance` in m11) solves a *local
algebraic* balance cell by cell::

    ne * nSF6 * k_iz(Te) * eps_T(Te) * e  =  P_rz(r,z)

That equation has two structural defects:

1. **No root where the power vanishes.**  The left side is strictly positive
   for any Te > 0, so as P_rz -> 0 (the whole downstream processing chamber)
   the bracket [0.5, 15] eV contains no root, `brentq` raises, and the cell
   silently keeps its initialisation value.  A gate
   (`ne < 1e10 or P_rz < 1e-3: continue`) skips those cells outright.  In the
   reference run only 27.6% of in-domain cells were ever solved; the other
   72.4% held the 3.0 eV initialisation and therefore responded to no input.

2. **Wall losses charged to interior cells.**  `eps_T = Ec + eiw + 2*Te`
   is the *global* energy cost per electron-ion pair: `eiw` (ion wall energy)
   and `2*Te` (electron wall energy) are surface channels.  They are correct in
   a 0D model where the walls are lumped in, but applying them in every
   interior cell charges cells for walls they never touch.

Both defects are structural rather than numerical: a purely local balance has no
mechanism to transport energy from where it is deposited to where it is lost, so
the cold, power-free region is unreachable.

Formulation
-----------
This module solves the steady electron energy equation instead::

    div( -kappa grad(Te) + (5/2) Te Gamma_e )  =  P_rz - P_loss(Te)

with, in eV-based units (divide the W/m^3 equation by the elementary charge)::

    div(kappa grad Te) - div((5/2) Te Gamma_e) + (P_rz - P_loss(Te))/e = 0

  * kappa   = 2.5 ne (Te*e) / (m_e nu_m)          [m^-1 s^-1]
              The coefficient is 5/2, NOT Braginskii's 3.16 ~ 3.2. Braginskii /
              Spitzer-Harm is derived for a FULLY IONISED plasma, where Coulomb
              collisions give nu ~ v^-3; this discharge is weakly ionised and
              electron-NEUTRAL dominated. Solving the Lorentz/two-term closure by
              direct Maxwellian quadrature gives, for kappa/(nT/m nu) and the
              matching convective coefficient c:
                  nu = const  (constant collision frequency) -> kappa=2.50, c=2.50
                  nu ~ v      (constant cross-section)       -> kappa=1.51, c=2.00
                  nu ~ v^-3   (Coulomb, fully ionised)       -> kappa=18.1, c=4.00
                                                  (e-e collisions -> 3.16 at Z=1)
              nu_m below is a Maxwellian-averaged RATE CONSTANT: one scalar per
              cell with no velocity dependence, i.e. the constant-nu case, whose
              exact coefficient is 5/2. That is also the c already used by the
              convective term, so (kappa, c) = (5/2, 5/2) is a consistent pair
              drawn from ONE closure -- (3.2, 5/2) silently mixed two.
              Using 3.2 over-predicted kappa by 28%, which compressed the Te
              spatial contrast by ~16% and under-predicted the peak Te by 0.23 eV.
  * nu_m    = ng * 2.8e-7 exp(-1.5/Te) * 1e-6     [s^-1]       same law as m10,
              so the conduction coefficient is consistent with the conductivity
              that sets the Ohmic deposition. Caveat inherited from that fit:
              kappa ~ Te exp(+1.5/Te) is non-monotonic below Te = 1.5 eV. The
              solved field here spans ~1.9-4.1 eV, so the inverted branch is not
              reached, but it is a property of the m10 rate fit, not of the closure.
  * Gamma_e = -D_a grad(ne)                       [m^-2 s^-1]  ambipolar electron
              flux, with D_a rebuilt exactly as in `solve_ne_ambipolar`.
  * P_loss  = sf6_rates.energy_loss_density(...)  the COMPLETE volumetric sink:
              SF6 ionisation (iz18-24), SF6 dissociation (d1-d5), SF6 vibrational
              (0.09 eV), SF5/SF3/F/F2/S channels, Ar ionisation/excitation (with
              the stepwise-ionisation correction) and per-species elastic drag.
              The wall terms now live in the boundary condition where they belong.

              This MUST be the complete expression.  An earlier version of this
              module summed a hand-copied 12-channel subset (iz18-24 + d1-d5),
              inherited from the legacy `Eloss` in m11.  That expression exists
              only to form the RATIO Ec = Eloss/Riz (energy per ionisation); as
              an absolute volumetric sink it is 4.03x too small, which starved
              the equation of 75% of its sink and drove Te to the 25 eV clamp.
              Measured on the reference run: subset = 37.3% of the deposited
              power, complete = 150.3%; a uniform Te closes the budget at 3.80 eV.
              energy_loss_density is the same expression global_model.py:214
              carries inline, i.e. the physics of the validated 0D model.

Boundary conditions
  * axis (i == 0): dTe/dr = 0, enforced structurally by the one-sided stencil.
  * walls: electron energy flux q.n = Gamma_e,wall * 2 Te, i.e. each escaping
    electron carries 2 Te.  Gamma_e,wall = 0.61 ne u_B (Bohm).  In Robin form
    q.n = h Te with h = 2 * 0.61 * ne * u_B  [m^-2 s^-1].

Because every in-domain cell is an unknown of one coupled linear system, no
initialisation can survive anywhere: the cold chamber temperature is *solved*
as the balance between conduction/convection in from the source and inelastic
plus wall losses out.

Numerics
--------
Cell-centred finite volume on the same mesh, mask and index maps as
`build_diffusion_matrix`, with identical arithmetic face averaging and
cylindrical r-weighting (the diffusion stencil here is verified against that
function cell-for-cell by `tests/test_te_energy_transport.py`).  Convection uses
first-order upwinding, which keeps the operator an M-matrix.  The nonlinear loss
is Newton-linearised about the previous iterate, L(T) ~ L(T*) + L'(T*)(T - T*);
L' > 0 so it enters as a stabilising diagonal sink.
"""

import logging

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.constants import e as eC, m_e, k as kB, pi

from ..chemistry.sf6_rates import energy_loss_density
from ..chemistry.rate_table import RateTable
from ..core.geometry import (BC_QUARTZ, BC_WINDOW, BC_AL_SIDE, BC_AL_TOP,
                             BC_WAFER, BC_SHOULDER, BC_AXIS)
from ..solvers.species_transport import flat_to_field

logger = logging.getLogger(__name__)

M_ION = 127.06 * 1.66054e-27   # SF5+ [kg], as in ambipolar_diffusion
SIGMA_IN = 5e-19               # ion-neutral cross-section [m^2]

TE_MIN, TE_MAX = 0.2, 25.0     # solution clamp [eV]

# Conduction coefficient in kappa = KAPPA_COEFF * ne * Te / (m_e * nu_m).
# 5/2 is the EXACT Lorentz value for a constant collision frequency, which is what
# nu_m (a Maxwellian-averaged rate constant) represents. It must equal the
# convective coefficient CONV_COEFF: both come from the same moment closure.
# Braginskii's 3.16 applies to fully ionised Coulomb plasmas and does NOT apply here.
KAPPA_COEFF = 2.5
CONV_COEFF = 2.5

# Species energy_loss_density consumes, in its argument order.
_LOSS_SPECIES = ('SF6', 'SF5', 'SF3', 'F', 'F2', 'S')


def _nu_m(Te, ng):
    """Momentum-transfer collision frequency [s^-1] (identical law to m10)."""
    return ng * 2.8e-7 * np.exp(-1.5 / np.maximum(Te, 0.05)) * 1e-6


def _loss(Te, ne, sp, nAr, nArm, tab):
    """Complete volumetric electron energy loss [eV m^-3 s^-1], evaluated per cell.

    Delegates to `sf6_rates.energy_loss_density` -- the canonical expression, the
    same one `global_model.py:214` carries inline -- with rate coefficients taken
    at the LOCAL Te of every cell via the interpolating RateTable (verified to
    ~2e-7 relative on this quantity).  Nothing here re-derives the physics.
    """
    k = tab(Te)
    return energy_loss_density(k, ne, Te,
                               sp['SF6'], sp['SF5'], sp['SF3'],
                               sp['F'], sp['F2'], sp['S'], nAr, nArm)


def _loss_and_derivative(Te, ne, sp, nAr, nArm, tab, dT=1e-3):
    """Loss L and dL/dTe, both [per eV].

    The derivative is a one-sided difference on Te at fixed composition, which is
    the correct Newton linearisation here: the species densities are outer-loop
    (Picard-lagged) quantities and are not functions of the Te being solved for.
    dL is clamped non-negative so it can only ever act as a stabilising diagonal
    sink; the loss is monotone increasing in Te over the physical range, so the
    clamp is inactive in practice and cannot flip the Newton step's sign.
    """
    L1 = _loss(Te, ne, sp, nAr, nArm, tab)
    L2 = _loss(np.clip(Te + dT, TE_MIN, TE_MAX + dT), ne, sp, nAr, nArm, tab)
    dL = (L2 - L1) / dT
    return L1, np.maximum(dL, 0.0)


def _ambipolar_Da(Te, inside, ng, Tgas, alpha):
    """Rebuild D_a^en exactly as `solve_ne_ambipolar` does (same constants)."""
    v_th_ion = np.sqrt(8 * kB * Tgas / (pi * M_ION))
    nu_in = ng * SIGMA_IN * v_th_ion
    D_i_base = kB * Tgas / (M_ION * nu_in) if nu_in > 0 else 0.01
    Ti_eV = Tgas * kB / eC

    Te_safe = np.where(inside, np.maximum(Te, 0.5), 0.5)
    Ti_safe = max(Ti_eV, 0.01)
    D_a_ep = D_i_base * (1.0 + Te_safe / Ti_safe)
    a = np.asarray(alpha)
    correction = (1.0 + a) / (1.0 + a * Ti_safe / Te_safe)
    return np.where(inside, D_a_ep * correction, 0.0)


def _build_operator(mesh, inside, bc_type, ij_to_flat, flat_to_ij, n_active,
                    kappa, Gam_r, Gam_z, h_wall, gamma_map):
    """Assemble  div(kappa grad T) - div((5/2) T Gamma)  per unit volume.

    Mirrors `build_diffusion_matrix` for the diffusive part (arithmetic face
    averaging, cylindrical r-weighting, L'Hopital on the axis, ghost-node
    elimination for the Robin wall term) and adds first-order upwind convection.

    `h_wall` is a 2D field [m^-2 s^-1] (unlike the scalar `gamma*v_th/4` of the
    density solver) because the electron energy flux 2 Te Gamma_e,wall carries
    the local ne through the Bohm flux.
    """
    Nr, Nz = mesh.Nr, mesh.Nz
    rows, cols, vals = [], [], []

    def add(k, kk, c):
        rows.append(k); cols.append(kk); vals.append(c)

    for k in range(n_active):
        i, j = flat_to_ij[k]
        ri, dri, dzj = mesh.rc[i], mesh.dr[i], mesh.dz[j]
        Kij = kappa[i, j]
        diag = 0.0

        # ------------------ radial ------------------
        if i == 0:
            # Axis: L'Hopital, (1/r) d(r K dT/dr) -> 2 K d2T/dr2.  Only the
            # outward neighbour appears, which *is* the dTe/dr = 0 condition.
            if i + 1 < Nr and inside[i + 1, j]:
                c = 2.0 * 0.5 * (Kij + kappa[i+1, j]) / (dri * mesh.drc[0])
                add(k, ij_to_flat[i+1, j], c); diag -= c
                # convection across the outer face (area/volume = 1/dri here)
                F = Gam_r[i+1, j] * 2.0 / dri   # outward normal flux weight
                if F > 0:
                    diag -= CONV_COEFF * F
                else:
                    add(k, ij_to_flat[i+1, j], -CONV_COEFF * F)
        else:
            # inner radial face at rf[i]; outward normal is -r_hat
            if inside[i-1, j]:
                c = 0.5 * (kappa[i-1, j] + Kij) * mesh.rf[i] / (ri * dri * mesh.drc[i-1])
                add(k, ij_to_flat[i-1, j], c); diag -= c
                F = -Gam_r[i, j] * mesh.rf[i] / (ri * dri)
                if F > 0:
                    diag -= CONV_COEFF * F
                else:
                    add(k, ij_to_flat[i-1, j], -CONV_COEFF * F)

            # outer radial face at rf[i+1]; outward normal is +r_hat
            if i < Nr - 1 and inside[i+1, j]:
                c = 0.5 * (Kij + kappa[i+1, j]) * mesh.rf[i+1] / (ri * dri * mesh.drc[i])
                add(k, ij_to_flat[i+1, j], c); diag -= c
                F = Gam_r[i+1, j] * mesh.rf[i+1] / (ri * dri)
                if F > 0:
                    diag -= CONV_COEFF * F
                else:
                    add(k, ij_to_flat[i+1, j], -CONV_COEFF * F)
            else:
                g = gamma_map.get(bc_type[i, j], 0.0)
                hij = h_wall[i, j]
                if g > 0 and hij > 0:
                    h = g * hij
                    drw = mesh.rf[min(i+1, Nr)] - ri
                    rf = mesh.rf[min(i+1, Nr)]
                    diag -= Kij * h / (Kij + h * drw) * rf / (ri * dri)

        # ------------------ axial ------------------
        # bottom face at zf[j]; outward normal is -z_hat
        if j > 0 and inside[i, j-1]:
            c = 0.5 * (kappa[i, j-1] + Kij) / (dzj * mesh.dzc[j-1])
            add(k, ij_to_flat[i, j-1], c); diag -= c
            F = -Gam_z[i, j] / dzj
            if F > 0:
                diag -= CONV_COEFF * F
            else:
                add(k, ij_to_flat[i, j-1], -CONV_COEFF * F)
        else:
            g = gamma_map.get(bc_type[i, j], 0.0)
            hij = h_wall[i, j]
            if g > 0 and hij > 0:
                h = g * hij
                dzw = mesh.zc[j] - mesh.zf[j]
                diag -= Kij * h / (Kij + h * dzw) / dzj

        # top face at zf[j+1]; outward normal is +z_hat
        if j < Nz - 1 and inside[i, j+1]:
            c = 0.5 * (Kij + kappa[i, j+1]) / (dzj * mesh.dzc[j])
            add(k, ij_to_flat[i, j+1], c); diag -= c
            F = Gam_z[i, j+1] / dzj
            if F > 0:
                diag -= CONV_COEFF * F
            else:
                add(k, ij_to_flat[i, j+1], -CONV_COEFF * F)
        else:
            g = gamma_map.get(bc_type[i, j], 0.0)
            hij = h_wall[i, j]
            if g > 0 and hij > 0:
                h = g * hij
                dzw = mesh.zf[min(j+1, Nz)] - mesh.zc[j]
                diag -= Kij * h / (Kij + h * dzw) / dzj

        add(k, k, diag)

    return sparse.csr_matrix((vals, (rows, cols)), shape=(n_active, n_active))


def _prep_species(species, inside, Nr, Nz):
    """Normalise the species argument to a dict of masked non-negative fields."""
    if not isinstance(species, dict):
        species = {'SF6': np.asarray(species)}
    sp = {}
    for name in _LOSS_SPECIES:
        f = species.get(name)
        sp[name] = (np.where(inside, np.maximum(np.asarray(f), 0.0), 0.0)
                    if f is not None else np.zeros((Nr, Nz)))
    return sp


def energy_balance(Te, ne, species, mesh, inside, P_rz, config,
                   nArm=0.0, rate_table=None):
    """Closing electron-energy balance for a GIVEN Te field (diagnostic only).

    Returns the same P_source_W / P_volloss_W / P_wall_W / sink_frac / Te_volavg
    dict that `solve_Te_energy_transport` reports, but evaluated on whatever Te is
    passed in. This is the single source of truth for the balance, so callers can
    evaluate it on the FINAL under-relaxed field that is actually saved and used
    downstream -- not only on the solver's raw (pre-Picard-damping) output. For a
    converged solve the two coincide; for a non-converged one (e.g. lxcat, where
    the raw solve clamps every iteration while the damped field lags) they differ,
    and the saved field is the honest one to report.
    """
    oper = config.operating if hasattr(config, 'operating') else config.get('operating', {})
    ng = (oper.get('pressure_mTorr', 10) * 0.133322) / (kB * oper.get('Tgas', 313))
    nAr = oper.get('frac_Ar', 0.0) * ng

    Nr, Nz = mesh.Nr, mesh.Nz
    ne_s = np.where(inside, np.maximum(ne, 1e8), 0.0)
    sp = _prep_species(species, inside, Nr, Nz)
    tab = rate_table if rate_table is not None else RateTable()

    Tc = np.clip(Te, TE_MIN, TE_MAX)
    P_src = float(np.sum(np.where(inside, P_rz, 0.0) * mesh.vol * inside))
    L_fin = np.where(inside, _loss(Tc, ne_s, sp, nAr, nArm, tab), 0.0)
    P_loss_W = float(np.sum(L_fin * eC * mesh.vol * inside))
    u_B_f = np.sqrt(eC * Tc / M_ION)
    q_w = 2.0 * Tc * (0.61 * ne_s * u_B_f) * eC
    P_wall_W = 0.0
    for i in range(Nr):
        for j in range(Nz):
            if not inside[i, j]:
                continue
            A = 0.0
            if i == Nr - 1 or not inside[i + 1, j]:
                A += 2 * pi * mesh.rf[i + 1] * mesh.dz[j]
            if j == 0 or not inside[i, j - 1]:
                A += 2 * pi * mesh.rc[i] * mesh.dr[i]
            if j == Nz - 1 or not inside[i, j + 1]:
                A += 2 * pi * mesh.rc[i] * mesh.dr[i]
            P_wall_W += q_w[i, j] * A

    Ti = Te[inside] if np.any(inside) else np.array([3.0])
    return {
        'Te_min': float(Ti.min()),
        'Te_max': float(Ti.max()),
        'Te_mean': float(Ti.mean()),
        'Te_volavg': float(np.sum(Te * mesh.vol * inside)
                           / max(np.sum(mesh.vol * inside), 1e-30)),
        'n_cells': int(np.sum(inside)),
        # RELATIVE tolerance: an exact >= TE_MAX-1e-6 test reads 0.0% for a field
        # pinned at 24.99982 eV -- the same asymptotic blind spot as the 3.0 eV
        # defect this module replaces.
        'frac_at_clamp': float(np.mean(Ti >= TE_MAX * (1.0 - 1e-3))),
        'frac_near_clamp_1pct': float(np.mean(Ti >= TE_MAX * 0.99)),
        'P_source_W': P_src,
        'P_volloss_W': P_loss_W,
        'P_wall_W': float(P_wall_W),
        'sink_frac': (P_loss_W + float(P_wall_W)) / P_src if P_src > 0 else float('nan'),
    }


def solve_Te_energy_transport(P_rz, ne, species, mesh, inside, config,
                              bc_type, ij_to_flat, flat_to_ij, n_active,
                              Te_init=None, alpha=0.0, nArm=0.0,
                              rate_table=None,
                              max_newton=40, tol=1e-3, relax=0.7):
    """Solve the steady electron energy equation for Te(r,z).

    Replacement for `solve_Te_local_power_balance`, with the geometry index maps
    and the local species composition additionally threaded through (both already
    sit in `state` / the previous Picard iterate).

    Parameters
    ----------
    species : dict of ndarray, or ndarray
        Local neutral densities keyed 'SF6','SF5','SF3','F','F2','S' (extra keys
        ignored).  For backwards compatibility a bare array is accepted and read
        as nSF6 with the other channels zeroed -- but note that discards most of
        the energy sink and is only for testing.
    nArm : float or ndarray
        Ar metastable density [m^-3], from the 0D warm start.  Contributes the
        (12 + 4.95) eV stepwise channel.

    Returns
    -------
    Te : ndarray (Nr, Nz)
        Electron temperature [eV].  Every in-domain cell is an unknown of the
        coupled system, so no cell retains an initialisation value.
    info : dict
        Convergence diagnostics, including the closing energy balance.
    """
    oper = config.operating if hasattr(config, 'operating') else config.get('operating', {})
    p_mTorr = oper.get('pressure_mTorr', 10)
    Tgas = oper.get('Tgas', 313)
    frac_Ar = oper.get('frac_Ar', 0.0)

    p_Pa = p_mTorr * 0.133322
    ng = p_Pa / (kB * Tgas)
    nAr = frac_Ar * ng

    Nr, Nz = mesh.Nr, mesh.Nz
    ne_s = np.where(inside, np.maximum(ne, 1e8), 0.0)

    if not isinstance(species, dict):
        logger.warning('solve_Te_energy_transport: bare array passed as `species`; '
                       'treating it as nSF6 and zeroing the other loss channels. '
                       'This under-counts the energy sink -- pass the species dict.')
        species = {'SF6': np.asarray(species)}
    sp = {}
    for name in _LOSS_SPECIES:
        f = species.get(name)
        sp[name] = (np.where(inside, np.maximum(np.asarray(f), 0.0), 0.0)
                    if f is not None else np.zeros((Nr, Nz)))

    tab = rate_table if rate_table is not None else RateTable()

    Te = np.full((Nr, Nz), 3.0) if Te_init is None else np.clip(Te_init.copy(), TE_MIN, TE_MAX)

    # Ambipolar electron flux Gamma_e = -D_a grad(ne), evaluated on faces and
    # held fixed through the Newton loop (ne is an outer-loop quantity).
    D_a = _ambipolar_Da(Te, inside, ng, Tgas, alpha)
    Gam_r = np.zeros((Nr + 1, Nz))
    Gam_z = np.zeros((Nr, Nz + 1))
    for i in range(1, Nr):
        for j in range(Nz):
            if inside[i, j] and inside[i-1, j]:
                Df = 0.5 * (D_a[i-1, j] + D_a[i, j])
                Gam_r[i, j] = -Df * (ne_s[i, j] - ne_s[i-1, j]) / mesh.drc[i-1]
    for i in range(Nr):
        for j in range(1, Nz):
            if inside[i, j] and inside[i, j-1]:
                Df = 0.5 * (D_a[i, j-1] + D_a[i, j])
                Gam_z[i, j] = -Df * (ne_s[i, j] - ne_s[i, j-1]) / mesh.dzc[j-1]

    gamma_map = {
        BC_QUARTZ: 1.0, BC_WINDOW: 1.0,
        BC_AL_SIDE: 1.0, BC_AL_TOP: 1.0,
        BC_WAFER: 1.0, BC_SHOULDER: 1.0,
        BC_AXIS: 0.0,
    }

    S_abs = np.where(inside, P_rz / eC, 0.0)   # [eV m^-3 s^-1]

    history = []
    converged = False
    for it in range(max_newton):
        # 5/2, not Braginskii's 3.2 -- see the module docstring. This is the
        # exact constant-nu Lorentz coefficient, matching nu_m's convention and
        # the 5/2 of the convective term (one closure, not two).
        kappa = np.where(inside,
                         KAPPA_COEFF * ne_s * (np.clip(Te, TE_MIN, TE_MAX) * eC)
                         / (m_e * _nu_m(Te, ng)), 0.0)

        # Bohm electron energy flux: q.n = 2 Te * Gamma_e,wall -> h = 2*0.61*ne*u_B
        u_B = np.sqrt(eC * np.clip(Te, TE_MIN, TE_MAX) / M_ION)
        h_wall = 2.0 * 0.61 * ne_s * u_B

        L, dL = _loss_and_derivative(Te, ne_s, sp, nAr, nArm, tab)
        L = np.where(inside, L, 0.0)
        dL = np.where(inside, dL, 0.0)

        A = _build_operator(mesh, inside, bc_type, ij_to_flat, flat_to_ij,
                            n_active, kappa, Gam_r, Gam_z, h_wall, gamma_map)

        dL_f = np.array([dL[flat_to_ij[k][0], flat_to_ij[k][1]] for k in range(n_active)])
        L_f = np.array([L[flat_to_ij[k][0], flat_to_ij[k][1]] for k in range(n_active)])
        S_f = np.array([S_abs[flat_to_ij[k][0], flat_to_ij[k][1]] for k in range(n_active)])
        T_f = np.array([Te[flat_to_ij[k][0], flat_to_ij[k][1]] for k in range(n_active)])

        M = A - sparse.diags(dL_f, format='csr')
        rhs = -S_f + L_f - dL_f * T_f

        T_new_f = np.asarray(spsolve(M, rhs), dtype=np.float64)
        if not np.all(np.isfinite(T_new_f)):
            logger.warning("Te energy solve produced non-finite values at Newton "
                           "iteration %d; keeping previous iterate.", it)
            break
        T_new_f = np.clip(T_new_f, TE_MIN, TE_MAX)

        Te_new = flat_to_field(T_new_f, flat_to_ij, Nr, Nz, fill=0.0)
        Te_relaxed = np.where(inside, (1 - relax) * Te + relax * Te_new, 0.0)

        delta = (np.max(np.abs(Te_relaxed[inside] - Te[inside]))
                 if np.any(inside) else 0.0)
        Te = Te_relaxed
        history.append(float(delta))
        if delta < tol:
            converged = True
            break

    Te = np.where(inside, np.clip(Te, TE_MIN, TE_MAX), 0.0)

    # ---- closing energy balance (the instrument that catches a starved sink) ----
    # A previous version of this module summed an incomplete channel subset; its
    # sink was 4x too small and Te ran to the clamp.  Reporting the balance makes
    # that failure mode self-evident instead of silent: if `sink_frac` is far from
    # ~1 the equation being solved is not the intended one.  Computed via the shared
    # `energy_balance` so the m11 caller can re-evaluate it on the final saved field.
    info = {
        'converged': bool(converged),
        'newton_iters': len(history),
        'last_delta': history[-1] if history else 0.0,
        'history': history,
    }
    info.update(energy_balance(Te, ne, species, mesh, inside, P_rz, config,
                               nArm=nArm, rate_table=tab))
    logger.info(
        "Te energy transport: converged=%s in %d Newton iters (delta=%.2e), "
        "Te in [%.3f, %.3f] eV, volavg=%.3f eV over %d cells, %.1f%% at clamp",
        info['converged'], info['newton_iters'], info['last_delta'],
        info['Te_min'], info['Te_max'], info['Te_volavg'], info['n_cells'],
        100 * info['frac_at_clamp'],
    )
    logger.info(
        "Te energy balance: source=%.1f W, volumetric loss=%.1f W, wall=%.1f W, "
        "sinks/source=%.3f", info['P_source_W'], info['P_volloss_W'],
        info['P_wall_W'], info['sink_frac'],
    )
    if info['frac_at_clamp'] > 0.01:
        logger.warning("Te energy transport: %.1f%% of cells are AT the %.1f eV clamp "
                       "-- the sink is probably under-counted; check sink_frac=%.3f",
                       100 * info['frac_at_clamp'], TE_MAX, info['sink_frac'])
    return Te, info
