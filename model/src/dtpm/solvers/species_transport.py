"""
Species transport solver on masked cylindrical (r,z) domain.

Extracted from Stage 10 TELSolver: sparse-matrix PDE solver for
neutral species (F, SF6, etc.) with Robin BCs and material-specific
wall chemistry.

The diffusion equation in cylindrical (r,z) coordinates:

    (1/r) d/dr(r D dF/dr) + d/dz(D dF/dz) + S - L*F = 0

is discretized on the active cells only. Inactive (solid) cells are
excluded from the linear system.
"""
import numpy as np
import logging
from scipy.sparse.linalg import spsolve
from scipy import sparse

from ..core.geometry import (
    BC_INTERIOR, BC_AXIS, BC_QUARTZ, BC_WINDOW,
    BC_AL_SIDE, BC_AL_TOP, BC_WAFER, BC_SHOULDER,
)
from ..chemistry.sf6_rates import rates
from ..chemistry.wall_chemistry import wall_sf6_regeneration, wall_F_loss

logger = logging.getLogger(__name__)


def field_to_flat(field_2d, flat_to_ij):
    """Extract active-cell values from 2D field into flat vector."""
    return np.array([field_2d[i, j] for i, j in flat_to_ij], dtype=np.float64)


def flat_to_field(flat, flat_to_ij, Nr, Nz, fill=np.nan):
    """Scatter flat vector back to 2D field."""
    f = np.full((Nr, Nz), fill, dtype=np.float64)
    for k, (i, j) in enumerate(flat_to_ij):
        f[i, j] = float(flat[k])
    return f


def build_diffusion_matrix(mesh, inside, bc_type, ij_to_flat, flat_to_ij,
                           n_active, D_field, gamma_map, v_th=None):
    """Assemble sparse diffusion operator for active cells only.

    Solves: div(D * grad(n)) = ... with Robin BCs at walls.

    Parameters
    ----------
    mesh : Mesh2D
    inside : ndarray (Nr, Nz), bool
    bc_type : ndarray (Nr, Nz), int
    ij_to_flat, flat_to_ij : index maps from build_index_maps()
    n_active : int
    D_field : ndarray (Nr, Nz)
        Diffusion coefficient field [m^2/s].
    gamma_map : dict {BC_TYPE: gamma_value}
        Robin BC coefficients per boundary type.
    v_th : float or None
        Thermal speed for Robin BC (m/s). Required if gamma_map has non-zero entries.

    Returns
    -------
    A : sparse CSR matrix (n_active x n_active)
    """
    Nr, Nz = mesh.Nr, mesh.Nz
    rows, cols, vals = [], [], []

    for k in range(n_active):
        i, j = flat_to_ij[k]
        ri = mesh.rc[i]
        dri = mesh.dr[i]
        dzj = mesh.dz[j]
        Dij = D_field[i, j]
        diag = 0.0

        # -- Radial direction --
        if i == 0:
            # Axis: L'Hopital rule for (1/r)*d(r*D*dF/dr) -> 2*D*d2F/dr2
            if i + 1 < Nr and inside[i + 1, j]:
                c = 2.0 * 0.5 * (Dij + D_field[i+1, j]) / (dri * mesh.drc[0])
                rows.append(k); cols.append(ij_to_flat[i+1, j]); vals.append(c)
                diag -= c
        else:
            # Inner radial neighbour
            if inside[i-1, j]:
                c = 0.5 * (D_field[i-1, j] + Dij) * mesh.rf[i] / (ri * dri * mesh.drc[i-1])
                rows.append(k); cols.append(ij_to_flat[i-1, j]); vals.append(c)
                diag -= c
            # Outer radial neighbour
            if i < Nr - 1 and inside[i+1, j]:
                c = 0.5 * (Dij + D_field[i+1, j]) * mesh.rf[i+1] / (ri * dri * mesh.drc[i])
                rows.append(k); cols.append(ij_to_flat[i+1, j]); vals.append(c)
                diag -= c
            else:
                # Wall Robin BC at outer radial boundary
                g = gamma_map.get(bc_type[i, j], 0.0)
                if g > 0 and v_th is not None:
                    h = g * v_th / 4
                    drw = mesh.rf[min(i+1, Nr)] - ri
                    rf = mesh.rf[min(i+1, Nr)]
                    diag -= Dij * h / (Dij + h * drw) * rf / (ri * dri)

        # -- Axial direction --
        # Bottom neighbour
        if j > 0 and inside[i, j-1]:
            c = 0.5 * (D_field[i, j-1] + Dij) / (dzj * mesh.dzc[j-1])
            rows.append(k); cols.append(ij_to_flat[i, j-1]); vals.append(c)
            diag -= c
        else:
            bc = BC_WAFER if bc_type[i, j] == BC_WAFER else bc_type[i, j]
            g = gamma_map.get(bc, 0.0)
            if g > 0 and v_th is not None:
                h = g * v_th / 4
                dzw = mesh.zc[j] - mesh.zf[j]
                diag -= Dij * h / (Dij + h * dzw) / dzj

        # Top neighbour
        if j < Nz - 1 and inside[i, j+1]:
            c = 0.5 * (Dij + D_field[i, j+1]) / (dzj * mesh.dzc[j])
            rows.append(k); cols.append(ij_to_flat[i, j+1]); vals.append(c)
            diag -= c
        else:
            bc = bc_type[i, j]
            if bc == BC_WINDOW:
                g = gamma_map.get(BC_WINDOW, 0.0)
            elif bc == BC_AL_TOP:
                g = gamma_map.get(BC_AL_TOP, 0.0)
            else:
                g = gamma_map.get(bc, 0.0)
            if g > 0 and v_th is not None:
                h = g * v_th / 4
                dzw = mesh.zf[min(j+1, Nz)] - mesh.zc[j]
                diag -= Dij * h / (Dij + h * dzw) / dzj

        rows.append(k); cols.append(k); vals.append(diag)

    return sparse.coo_matrix((vals, (rows, cols)), shape=(n_active, n_active)).tocsr()


def solve_species_transport(mesh, inside, bc_type, ij_to_flat, flat_to_ij,
                            n_active, ne, Te, config,
                            nSF6_init=None, nF_init=None,
                            n_iter=60, w=0.12, verbose=False):
    """Solve the coupled F/SF6 transport on the masked reactor domain.

    This is the inner chemistry loop — given ne(r,z) and Te(r,z),
    solve for the steady-state nF(r,z) and nSF6(r,z).

    Parameters
    ----------
    mesh : Mesh2D
    inside, bc_type, ij_to_flat, flat_to_ij, n_active : geometry data
    ne : ndarray (Nr, Nz) — electron density [m^-3]
    Te : ndarray (Nr, Nz) — electron temperature [eV]
    config : SimulationConfig
    n_iter : int — number of inner iterations
    w : float — under-relaxation factor

    Returns
    -------
    dict with nF, nSF6, nSF5, F_drop_pct
    """
    from scipy.constants import k as kB, pi

    rgeom = config.reactor_geometry if hasattr(config, 'reactor_geometry') else config.get('reactor_geometry', {})
    oper = config.operating if hasattr(config, 'operating') else config.get('operating', {})
    wc = config.wall_chemistry if hasattr(config, 'wall_chemistry') else config.get('wall_chemistry', {})

    R_icp = rgeom.get('R_icp', 0.038)
    R_proc = rgeom.get('R_proc', 0.105)
    L_proc = rgeom.get('L_proc', 0.050)
    L_apt = rgeom.get('L_apt', 0.002)
    L_icp = rgeom.get('L_icp', 0.1815)
    z_apt_bot = L_proc
    z_apt_top = L_proc + L_apt

    p_mTorr = oper.get('pressure_mTorr', 10)
    Tgas = oper.get('Tgas', 313)
    frac_Ar = oper.get('frac_Ar', 0.0)
    Q_sccm = oper.get('Q_sccm', 100)

    p_Pa = p_mTorr * 0.133322
    ng = p_Pa / (kB * Tgas)
    nSF6_feed = ng * (1 - frac_Ar)
    nAr = ng * frac_Ar

    D_F_val = 1.6    # m^2/s
    D_SF6_val = 0.5  # m^2/s
    v_th_F = np.sqrt(8 * kB * Tgas / (pi * 19.0 * 1.66054e-27))

    V_total = pi * R_icp**2 * L_icp + pi * R_proc**2 * L_proc
    Q_tp = Q_sccm * 1e-6 / 60 * 1.01325e5 * (Tgas / 273.15)
    tau_R = p_Pa * V_total / Q_tp if Q_tp > 0 else 1e10

    Nr, Nz = mesh.Nr, mesh.Nz

    # Gamma map from config
    from ..chemistry.wall_chemistry import get_gamma_map
    gamma_map = get_gamma_map(
        wc.get('gamma_quartz', 0.001), wc.get('gamma_Al', 0.18),
        wc.get('gamma_wafer', 0.025), wc.get('gamma_window', 0.001))

    # Build diffusion operators (constant D fields)
    D_F = np.where(inside, D_F_val, 0.0)
    D_SF6 = np.where(inside, D_SF6_val, 0.0)
    A_F = build_diffusion_matrix(mesh, inside, bc_type, ij_to_flat, flat_to_ij,
                                  n_active, D_F, gamma_map, v_th_F)
    A_SF6 = build_diffusion_matrix(mesh, inside, bc_type, ij_to_flat, flat_to_ij,
                                    n_active, D_SF6, {}, v_th_F)

    # Initial fields
    nSF6 = nSF6_init if nSF6_init is not None else np.where(inside, nSF6_feed * 0.5, 0.0)
    nF = nF_init if nF_init is not None else np.where(inside, 1e18, 0.0)
    nSF5 = np.where(inside, nSF6 * 0.03, 0.0)

    # Penning parameters
    k_Penn = 2e-10 * 1e-6
    k_Penn_d = 3e-10 * 1e-6
    k_quench = 5e-10 * 1e-6
    k_exc = 3.5e-11 * 1e-6
    k_izm = 6e-11 * 1e-6

    for it in range(n_iter):
        # -- SF6 transport --
        src_s = np.zeros(n_active)
        loss_s = np.zeros(n_active)
        for kk in range(n_active):
            i, j = flat_to_ij[kk]
            kc = rates(Te[i, j])
            kd = (kc['d1'] + kc['d2'] + kc['d3'] + kc['d4'] + kc['d5']
                  + kc['iz_SF6_total'] + kc['att_SF6_total'])
            rep = 1.0 / tau_R
            z = mesh.zc[j]
            if z > z_apt_top + 0.7 * L_icp:
                rep += ((z - z_apt_top - 0.7 * L_icp)
                        / (0.3 * L_icp) / (tau_R * 0.2))
            bc = bc_type[i, j]
            dr_dz = mesh.dr[i] if bc in (BC_QUARTZ, BC_AL_SIDE, BC_SHOULDER) else mesh.dz[j]
            Rw = wall_sf6_regeneration(bc, nF[i, j], nSF5[i, j], v_th_F, dr_dz)
            src_s[kk] = nSF6_feed * rep + Rw
            loss_s[kk] = -(ne[i, j] * kd + rep)

        x = np.asarray(spsolve(A_SF6 + sparse.diags(loss_s), -src_s), dtype=np.float64)
        nSF6_new = flat_to_field(np.clip(x, 0.01 * nSF6_feed, nSF6_feed),
                                  flat_to_ij, Nr, Nz, fill=0.0)
        nSF6 = (1 - w) * nSF6 + w * nSF6_new

        # -- F transport --
        f_src = np.zeros(n_active)
        f_loss = np.zeros(n_active)
        for kk in range(n_active):
            i, j = flat_to_ij[kk]
            kc = rates(Te[i, j])
            RF = ne[i, j] * nSF6[i, j] * (
                kc['d1'] + kc['d2']*2 + kc['d3']*3 + kc['d4']*2
                + kc['d5']*3 + kc.get('d7', 0) + kc.get('d8', 0))
            if frac_Ar > 0.01:
                nArm = (k_exc * ne[i, j] * nAr
                        / max(k_izm * ne[i, j] + k_quench * nSF6_feed + 1e-20, 1e-30))
                RF += (k_Penn + k_Penn_d) * nArm * nSF6[i, j]
            f_src[kk] = RF
            bc = bc_type[i, j]
            dr_dz = mesh.dr[i] if bc in (BC_QUARTZ, BC_AL_SIDE, BC_SHOULDER) else mesh.dz[j]
            Rl = wall_F_loss(bc, nF[i, j], v_th_F, dr_dz)
            f_loss[kk] = -Rl / max(nF[i, j], 1e6)

        pump_wall = np.full(n_active, -1.0 / tau_R) + f_loss
        y = np.asarray(spsolve(A_F + sparse.diags(pump_wall), -f_src), dtype=np.float64)
        nF_new = flat_to_field(np.maximum(y, 0.0), flat_to_ij, Nr, Nz, fill=0.0)
        nF = (1 - w) * nF + w * nF_new

        nSF5 = np.where(inside, nSF6 * 0.03, 0.0)

    # Compute [F] drop
    rw = mesh.rc[inside[:, 0]]
    Fw = nF[:, 0][inside[:, 0]]
    Fc = Fw[0] if len(Fw) > 0 else 0
    Fe = Fw[-1] if len(Fw) > 0 else 0
    drop = (1 - Fe / max(Fc, 1e6)) * 100 if Fc > 1e6 else 0

    logger.info(f"Species transport: {n_iter} iterations, [F] drop = {drop:.1f}%")

    return {
        'nF': nF, 'nSF6': nSF6, 'nSF5': nSF5,
        'F_wafer': Fw, 'r_wafer': rw,
        'F_drop_pct': drop,
    }
