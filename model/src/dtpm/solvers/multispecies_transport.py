"""
9-Species Multi-Species Transport Solver for IPI ICP Reactor.

Solves the full Lallement SF6/Ar neutral chemistry (9 species) on the
masked cylindrical (r,z) domain with material-specific wall recombination.

Species: SF6, SF5, SF4, SF3, SF2, SF, S, F, F2

Each species satisfies a steady-state diffusion-reaction equation:
    div(D_s * grad(n_s)) + S_s(n_all, ne, Te) - L_s * n_s = 0

Ported from Stage 10 TELMultiSpeciesSolver, adapted to use the Phase 1
framework's standalone functions (no TELSolver inheritance).
"""

import numpy as np
import logging
from scipy.sparse import diags as sp_diags
from scipy.sparse.linalg import spsolve
from scipy.ndimage import gaussian_filter
from scipy.constants import k as kB, pi, atomic_mass as AMU_KG

from ..chemistry.sf6_chemistry import (
    SPECIES, M, WALL_GAMMA, compute_rates, compute_diffusion_coefficients,
)
from ..chemistry.rate_table import RateTable, CellRates
from ..core.geometry import (
    BC_QUARTZ, BC_WINDOW, BC_AL_SIDE, BC_AL_TOP, BC_WAFER, BC_SHOULDER,
)
from .species_transport import build_diffusion_matrix, field_to_flat, flat_to_field

logger = logging.getLogger(__name__)


def _gamma_map_for_species(species_name, gamma_Al_calibrated=0.18):
    """Build bc_type -> gamma dict for a given species using Kokkoris values."""
    bc_to_mat = {
        BC_QUARTZ: 'quartz', BC_WINDOW: 'window',
        BC_AL_SIDE: 'aluminium', BC_AL_TOP: 'aluminium',
        BC_SHOULDER: 'aluminium', BC_WAFER: 'silicon',
    }
    gmap = {}
    for bc_val, mat in bc_to_mat.items():
        g = WALL_GAMMA.get(mat, {}).get(species_name, 0.001)
        if species_name == 'F' and mat == 'aluminium':
            g = gamma_Al_calibrated
        gmap[bc_val] = g
    return gmap


def solve_multispecies_transport(mesh, inside, bc_type, ij_to_flat, flat_to_ij,
                                 n_active, ne, Te, config,
                                 n_iter=60, w=0.08, verbose=True):
    """Solve the full 9-species neutral chemistry on the masked reactor domain.

    Uses the self-consistent ne(r,z) and Te(r,z) from the Picard loop.

    Parameters
    ----------
    mesh : Mesh2D
    inside, bc_type, ij_to_flat, flat_to_ij, n_active : geometry data
    ne : ndarray (Nr, Nz) — electron density [m^-3]
    Te : ndarray (Nr, Nz) — electron temperature [eV] (scalar or field)
    config : SimulationConfig
    n_iter : int — outer iterations
    w : float — under-relaxation

    Returns
    -------
    dict with 'fields' (all 9 species 2D arrays), 'ions' (charged species),
    'F_drop_pct', and diagnostics.
    """
    rgeom = config.reactor_geometry if hasattr(config, 'reactor_geometry') else config.get('reactor_geometry', {})
    oper = config.operating if hasattr(config, 'operating') else config.get('operating', {})
    wc = config.wall_chemistry if hasattr(config, 'wall_chemistry') else config.get('wall_chemistry', {})

    p_mTorr = oper.get('pressure_mTorr', 10)
    Tgas = oper.get('Tgas', 313)
    frac_Ar = oper.get('frac_Ar', 0.0)
    Q_sccm = oper.get('Q_sccm', 100)
    gamma_Al = wc.get('gamma_Al', 0.18)

    p_Pa = p_mTorr * 0.133322
    ng = p_Pa / (kB * Tgas)
    ng_cm3 = ng * 1e-6
    nSF6_feed = ng * (1 - frac_Ar)

    Nr, Nz = mesh.Nr, mesh.Nz
    N = n_active

    # Residence time
    V_total = np.sum(mesh.vol[inside])
    Q_tp = Q_sccm * 1e-6 / 60 * 1.01325e5 * (Tgas / 273.15)
    tau_R = p_Pa * V_total / Q_tp if Q_tp > 0 else 1e10

    # Diffusion coefficients for all species
    D_all = compute_diffusion_coefficients(Tgas, p_mTorr)

    # Build diffusion matrices for all 9 species (with material-specific wall gamma)
    A_mats = {}
    for sp in SPECIES:
        D_field = np.where(inside, D_all[sp], 0.0)
        gamma_map = _gamma_map_for_species(sp, gamma_Al)
        v_th_sp = np.sqrt(8 * kB * Tgas / (pi * M[sp] * AMU_KG))
        A_mats[sp] = build_diffusion_matrix(
            mesh, inside, bc_type, ij_to_flat, flat_to_ij,
            n_active, D_field, gamma_map, v_th=v_th_sp)

    # Average Te for rate computation
    Te_field = Te if isinstance(Te, np.ndarray) and Te.ndim == 2 else np.full((Nr, Nz), float(Te))
    Te_avg = np.mean(Te_field[inside & (Te_field > 0)]) if np.any(inside & (Te_field > 0)) else 2.5

    # ---- rate coefficients: per-cell (local Te) or the legacy domain scalar ----
    # Legacy behaviour collapses Te(r,z) to Te_avg and uses ONE k dict everywhere,
    # so the nine neutral fields see no spatial variation in their rates at all.
    # With per_cell_rates, `k` becomes a view returning the coefficient at the
    # current cell index; the source/loss expressions below are unchanged.
    _chem_cfg = (config.chemistry if hasattr(config, 'chemistry')
                 else config.get('chemistry', {})) or {}
    _per_cell = bool(_chem_cfg.get('per_cell_rates', False))
    _reconciled = bool(_chem_cfg.get('reconciled_rates', False))
    if _per_cell:
        _rate_tab = RateTable(fn=lambda T: compute_rates(T, ng_cm3, frac_Ar, reconciled=_reconciled))
        _Te_flat = np.array([Te_field[flat_to_ij[idx][0], flat_to_ij[idx][1]]
                             for idx in range(N)], dtype=np.float64)
        k = CellRates(_rate_tab(_Te_flat))
        logger.info("Multi-species: PER-CELL rates over Te in [%.3f, %.3f] eV "
                    "(%d cells); legacy scalar would have used Te_avg=%.3f eV",
                    float(_Te_flat.min()), float(_Te_flat.max()), N, float(Te_avg))
    else:
        k = compute_rates(Te_avg, ng_cm3, frac_Ar, reconciled=_reconciled)

    # Initialise all species
    fields = {}
    fields['SF6'] = np.where(inside, nSF6_feed * 0.5, 0.0)
    fields['F']   = np.where(inside, nSF6_feed * 0.1, 0.0)
    for sp in ['SF5', 'SF4', 'SF3', 'SF2', 'SF', 'S', 'F2']:
        fields[sp] = np.where(inside, nSF6_feed * 0.01, 0.0)

    if verbose:
        logger.info(f"Multi-species transport: {N} active cells, {n_iter} iterations, "
                    f"Te_avg={Te_avg:.2f} eV")

    # Iteration loop
    for it in range(n_iter):
        # Update rates every 5 iterations for self-consistency.
        # NOTE: this is a no-op -- Te_field is an argument to this function and is
        # never reassigned inside it, so this recomputes an identical k. Retained
        # unchanged on the legacy path for bit-exact reproduction; skipped on the
        # per-cell path, where k is a view that already tracks the local Te.
        if (not _per_cell) and it > 0 and it % 5 == 0:
            Te_avg = np.mean(Te_field[inside & (Te_field > 0)])
            k = compute_rates(Te_avg, ng_cm3, frac_Ar, reconciled=_reconciled)

        for sp in SPECIES:
            rhs = np.zeros(N)
            loss_diag = np.zeros(N)

            for idx in range(N):
                if _per_cell:
                    k._i = idx          # resolve k['...'] at THIS cell's Te
                i, j = flat_to_ij[idx]
                ne_l = ne[i, j]
                n6 = fields['SF6'][i, j]
                nFl = fields['F'][i, j]

                if sp == 'F':
                    src = (ne_l * n6 * (k['d1'] + 2*k['d2'] + 3*k['d3'] + 2*k['d4'] + 3*k['d5'])
                           + ne_l * fields['SF5'][i, j] * k['d7']
                           + ne_l * fields['SF4'][i, j] * k['d8']
                           + ne_l * fields['SF3'][i, j] * k['d9']
                           + ne_l * fields['SF2'][i, j] * k['d10']
                           + ne_l * fields['SF'][i, j] * k['d11']
                           + 2 * ne_l * fields['F2'][i, j] * k['d6'])
                    loss = k['nr42'] * fields['SF5'][i, j] + 1 / tau_R

                elif sp == 'SF6':
                    src = nSF6_feed / tau_R + k['nr42'] * fields['SF5'][i, j] * nFl
                    loss = ne_l * k['k_e_SF6'] + 1 / tau_R

                elif sp == 'SF5':
                    src = (k['d1'] * ne_l * n6
                           + k['nr41'] * fields['SF4'][i, j] * nFl)
                    loss = ((k['d7'] + k['iz25'] + k['iz26']) * ne_l
                            + k['nr42'] * nFl + 1 / tau_R)

                elif sp == 'SF4':
                    src = (k['d2'] * ne_l * n6
                           + k['d7'] * ne_l * fields['SF5'][i, j]
                           + k['nr45'] * fields['SF5'][i, j]**2)
                    loss = k['d8'] * ne_l + k['nr41'] * nFl + 1 / tau_R

                elif sp == 'SF3':
                    src = (k['d3'] * ne_l * n6
                           + k['d8'] * ne_l * fields['SF4'][i, j])
                    loss = (k['d9'] + k['iz27']) * ne_l + k['nr40'] * nFl + 1 / tau_R

                elif sp == 'SF2':
                    src = (k['d4'] * ne_l * n6
                           + k['d9'] * ne_l * fields['SF3'][i, j])
                    loss = k['d10'] * ne_l + k['nr39'] * nFl + 1 / tau_R

                elif sp == 'SF':
                    src = (k['d5'] * ne_l * n6
                           + k['d10'] * ne_l * fields['SF2'][i, j])
                    loss = k['d11'] * ne_l + k['nr38'] * nFl + 1 / tau_R

                elif sp == 'S':
                    src = k['d11'] * ne_l * fields['SF'][i, j]
                    loss = k['iz29'] * ne_l + k['nr37'] * nFl + 1 / tau_R

                elif sp == 'F2':
                    src = ne_l * n6 * (k['d4'] + k['d5'])
                    loss = k['d6'] * ne_l + 1 / tau_R

                else:
                    src = 0
                    loss = 1 / tau_R

                rhs[idx] = -src
                loss_diag[idx] = -loss

            A_total = A_mats[sp] + sp_diags(loss_diag, 0, shape=(N, N))
            try:
                x = spsolve(A_total, rhs)
                x = np.maximum(x, 0)
                new = flat_to_field(x, flat_to_ij, Nr, Nz, fill=0.0)
                fields[sp] = (1 - w) * fields[sp] + w * new
                fields[sp] = np.clip(fields[sp], 0, ng * 10)
            except Exception:
                pass

        if verbose and (it % 10 == 0 or it == n_iter - 1):
            Fw = fields['F'][:, 0]
            active_wafer = inside[:, 0]
            Fw_active = Fw[active_wafer]
            if len(Fw_active) > 1:
                Fc = max(Fw_active[0], 1e-10)
                Fe = Fw_active[-1]
                drop = (1 - Fe / Fc) * 100
            else:
                drop = 0
            sf6d = (1 - np.mean(fields['SF6'][inside]) / nSF6_feed) * 100
            logger.info(f"  iter {it:3d}: [F] drop={drop:.1f}%  SF6 depl={sf6d:.0f}%")

    # Charged species from quasi-neutrality
    k_rec = 1.5e-9 * 1e-6  # m^3/s
    R_att = (k['at30'] + k['at31']) * fields['SF6']
    alpha = (-1 + np.sqrt(np.clip(1 + 4 * R_att / (k_rec * ne + 1e-30), 0, 1e12))) / 2
    alpha = np.clip(alpha, 0, 500)
    n_neg = alpha * ne
    n_pos = ne + n_neg

    # Smooth charged species for visualization
    def smooth(f):
        f2 = f.copy()
        f2[~inside] = 0
        for _ in range(3):
            fs = gaussian_filter(f2, 2.5)
            ms = gaussian_filter(inside.astype(float), 2.5)
            f2 = np.where(ms > 0.05, fs / ms, 0)
            f2[~inside] = 0
        return f2

    ne_s = smooth(ne)
    np_s = smooth(n_pos)
    nn_s = smooth(n_neg)
    ions = {
        'ne': ne_s, 'n+': np_s, 'n-': nn_s,
        'alpha': np.clip(nn_s / (ne_s + 1e-20), 0, 500),
        'SF5+': np_s * 0.65, 'SF3+': np_s * 0.20,
        'SF4+': np_s * 0.10, 'F+': np_s * 0.05,
        'F-': nn_s * 0.55, 'SF6-': nn_s * 0.25,
        'SF5-': nn_s * 0.15, 'SF4-': nn_s * 0.05,
    }

    # F drop
    Fw = fields['F'][:, 0]
    active_wafer = inside[:, 0]
    Fw_active = Fw[active_wafer]
    if len(Fw_active) > 1:
        drop = (1 - Fw_active[-1] / max(Fw_active[0], 1e-10)) * 100
    else:
        drop = 0

    logger.info(f"Multi-species complete: [F] drop = {drop:.1f}%")

    return {
        'fields': fields,
        'ions': ions,
        'ne': ne,
        'Te_avg': Te_avg,
        'per_cell_rates': _per_cell,
        'F_drop_pct': drop,
        'nF': fields['F'],
        'nSF6': fields['SF6'],
    }
