"""
Module 06c — Cylindrical FDTD for ICP Reactor
==============================================
Axisymmetric TE-mode FDTD solver for the azimuthal electric field
E_theta driven by the ICP coil current.

Physics:
- Cylindrical (r, z) coordinates with azimuthal symmetry (d/d_theta = 0)
- TE mode: E_theta, H_r, H_z
- Yee-grid staggering:
    E_theta lives at cell centres (rc[i], zc[j])
    H_r lives at (rc[i], zf[j])   — staggered in z
    H_z lives at (rf[i], zc[j])   — staggered in r
- Maxwell's curl equations:
    dH_r/dt = +(1/mu_0) * dE_theta/dz
    dH_z/dt = -(1/mu_0) * (1/r) * d(r * E_theta)/dr
    dE_theta/dt = (1/eps) * (dH_r/dz - dH_z/dr) - sigma*E/eps - J_coil/eps
- Axis: E_theta(r=0) = 0 by symmetry; H_z uses L'Hopital
- PEC at metal walls: E_theta = 0
- Mur 1st-order ABC at open boundaries
- Dielectric eps_r in quartz region
- Plasma conductivity sigma for lossy medium (Ohmic heating)

Source normalisation:
- The coil carries current I_peak [A] in a ring at radius R_coil.
- In the FDTD cell, J_theta = I / (dr * dz) [A/m^2].
- A Gaussian-modulated sinusoidal envelope ramps up over ~1.5 RF periods.
"""

import numpy as np
import logging

from ..core.geometry import BC_QUARTZ, BC_AL_SIDE, BC_AL_TOP, BC_WAFER, BC_SHOULDER, BC_WINDOW

logger = logging.getLogger(__name__)


def run_fdtd_cylindrical(mesh, inside, bc_type, coil_positions,
                         source_frequency, I_peak,
                         n_rf_cycles=2.0, eps_r_quartz=3.8,
                         sigma_plasma=None,
                         R_icp=0.038, quartz_wall_thickness=0.0005,
                         z_apt_top=0.052,
                         em_active=None, eps_r_map=None):
    """Run axisymmetric TE-mode FDTD on the IPI ICP reactor geometry.

    All field updates are vectorised with numpy — no Python loops in the
    time-stepping hot path except for the coil source injection (O(N_coils)).

    Parameters
    ----------
    mesh : Mesh2D
        Cylindrical mesh with rc, zc, rf, zf, dr, dz, drc, dzc arrays.
    inside : ndarray (Nr, Nz), bool
        Geometry mask.
    bc_type : ndarray (Nr, Nz), int
        Boundary type classification.
    coil_positions : list of (i, j)
        Grid indices of coil source positions (inside gas volume).
    source_frequency : float
        RF source frequency [Hz].
    I_peak : float
        Peak coil current [A].
    n_rf_cycles : float
        Number of RF cycles to simulate.
    eps_r_quartz : float
        Relative permittivity of quartz dielectric.
    sigma_plasma : ndarray (Nr, Nz) or None
        Plasma conductivity [S/m]. If None, vacuum everywhere.

    Returns
    -------
    dict with E_theta, E_theta_rms, H_r, H_z, B_r, B_z, dt_fdtd, etc.
    """
    mu_0 = 4e-7 * np.pi
    eps_0 = 8.854187817e-12
    c = 1.0 / np.sqrt(mu_0 * eps_0)

    Nr, Nz = mesh.Nr, mesh.Nz
    omega = 2.0 * np.pi * source_frequency

    # -- Spatially varying permittivity --
    if eps_r_map is not None:
        eps_r = eps_r_map.copy()
    else:
        eps_r = np.ones((Nr, Nz))
        eps_r[bc_type == BC_QUARTZ] = eps_r_quartz
        R_outer = R_icp + quartz_wall_thickness
        for i in range(Nr):
            r = mesh.rc[i]
            if R_icp <= r <= R_outer:
                for j in range(Nz):
                    if mesh.zc[j] >= z_apt_top:
                        eps_r[i, j] = eps_r_quartz
    eps_field = eps_0 * eps_r

    # -- EM-active mask --
    # If em_active is provided, use it (extends beyond gas volume into quartz/coil).
    # Otherwise fall back to gas-only mask.
    if em_active is None:
        em_active = inside

    # -- CFL-stable time step --
    dr_min = mesh.dr.min()
    dz_min = mesh.dz.min()
    dt = 0.5 * min(dr_min, dz_min) / c
    T_rf = 1.0 / source_frequency
    n_steps = max(int(n_rf_cycles * T_rf / dt), 100)
    cfl = c * dt / min(dr_min, dz_min)

    logger.info(f"Cylindrical FDTD: {Nr}x{Nz}, {n_steps} steps, "
                f"dt={dt:.4e}s, CFL={cfl:.3f}, T_sim={n_steps*dt*1e9:.2f}ns")

    # -- Initialize fields --
    E_theta = np.zeros((Nr, Nz))
    H_r = np.zeros((Nr, Nz))  # lives at (rc[i], zf[j]), j = 0..Nz-1
    H_z = np.zeros((Nr + 1, Nz))  # lives at (rf[i], zc[j]), i = 0..Nr

    # Conductivity
    if sigma_plasma is None:
        sigma_plasma = np.zeros((Nr, Nz))

    # -- Pre-compute update coefficients (vectorised) --
    # Lossy medium coefficients for E_theta update:
    #   E^{n+1} = Ca * E^n + Cb * (curl H)
    #   Ca = (1 - sigma*dt/(2*eps)) / (1 + sigma*dt/(2*eps))
    #   Cb = (dt/eps) / (1 + sigma*dt/(2*eps))
    denom = 1.0 + sigma_plasma * dt / (2.0 * eps_field)
    Ca = (1.0 - sigma_plasma * dt / (2.0 * eps_field)) / denom
    Cb = (dt / eps_field) / denom

    # -- PEC mask: metal walls and EM-inactive cells --
    # Cells outside em_active are PEC (zero E-field).
    # Metal walls (Al, wafer, shoulder) are also PEC.
    pec_mask = ~em_active | np.isin(bc_type, [BC_AL_SIDE, BC_AL_TOP, BC_WAFER, BC_SHOULDER])

    # -- Source: current density J_theta = I / (cell area) --
    # For each coil position, compute J = I / (dr_i * dz_j)
    coil_J_factor = []  # list of (i, j, area_factor)
    for (ci, cj) in coil_positions:
        if 0 < ci < Nr and 0 < cj < Nz and em_active[ci, cj] and not pec_mask[ci, cj]:
            cell_area = mesh.dr[ci] * mesh.dz[cj]
            coil_J_factor.append((ci, cj, 1.0 / cell_area))

    if not coil_J_factor:
        logger.warning("No valid coil source positions found inside active domain!")

    # -- Source envelope --
    # For steady-state CW ICP operation we use a linear ramp-up over the
    # first two RF cycles (to avoid starting transients) and then a
    # CONSTANT-amplitude sinusoidal drive for the remainder of the run.
    # The prior Gaussian-pulsed envelope (t_center=1.5T, t_width=0.5T) was
    # intended for *transient* resonator studies; it turns the source off
    # after ~3 RF cycles which makes the steady-state RMS accumulation
    # capture a free-decaying cavity mode instead of the driven amplitude.
    # That was the systematic ~1000x undershoot seen in the prior pipeline
    # (see docs/CODE_REVIEW_ULTRAREVIEW.md).
    T_rf_steps = max(int(T_rf / dt), 1)
    ramp_end = 2 * T_rf_steps   # linear ramp over 2 RF cycles

    # -- RMS accumulation over last RF cycle --
    E2_accum = np.zeros((Nr, Nz))
    n_accum = 0
    accum_start = max(n_steps - int(T_rf / dt), n_steps // 2)

    # -- Pre-compute geometric coefficients for vectorised updates --
    # H_r update: dH_r/dt = (1/mu_0) * dE_theta/dz
    #   H_r[i, j] += (dt/mu_0) * (E_theta[i, j+1] - E_theta[i, j]) / dzc[j]
    # (dzc = distance between z cell centres = mesh.dzc, length Nz-1)
    dt_over_mu0 = dt / mu_0
    inv_dzc = 1.0 / mesh.dzc  # shape (Nz-1,)

    # H_z update: dH_z/dt = -(1/mu_0) * (1/r_face) * d(r*E_theta)/dr
    # At face i (between cells i-1 and i):
    #   H_z[i, j] -= (dt/mu_0) / rf[i] * (rc[i]*E[i,j] - rc[i-1]*E[i-1,j]) / drc[i-1]
    # For i=0: axis boundary, H_z[0,:] uses L'Hopital -> 2*E[0,:]/dr[0]
    #   But E_theta[0,:]=0 by symmetry, so at the first interior face:
    #   H_z[1,:] uses (rc[1]*E[1] - rc[0]*E[0]) / drc[0] with rf[1]
    # For i=Nr: outer boundary (Mur ABC handles this)

    # rc broadcasted for r*E products
    rc_2d = mesh.rc[:, None]  # (Nr, 1)

    # Mur ABC storage
    E_bnd_rN = np.zeros(Nz)
    E_adj_rN = np.zeros(Nz)
    coeff_mur_r = (c * dt - mesh.dr[-1]) / (c * dt + mesh.dr[-1])

    # -- Time-stepping loop --
    for n in range(n_steps):
        # Save boundary for Mur ABC at r = R_proc
        E_bnd_rN[:] = E_theta[-1, :]
        E_adj_rN[:] = E_theta[-2, :]

        # -- 1. Update H_r (vectorised) --
        # H_r[i, j] += (dt/mu0) * (E_theta[i, j+1] - E_theta[i, j]) / dzc[j]
        # j ranges 0..Nz-2 (H_r at face j+1/2 between cell j and j+1)
        H_r[:, :-1] += dt_over_mu0 * (E_theta[:, 1:] - E_theta[:, :-1]) * inv_dzc[None, :]

        # -- 2. Update H_z (vectorised) --
        # H_z lives at faces rf[i], i = 0..Nr
        # Interior faces i = 1..Nr-1:
        #   H_z[i,j] -= (dt/mu0) / rf[i] * (rc[i]*E[i,j] - rc[i-1]*E[i-1,j]) / drc[i-1]
        rE = rc_2d * E_theta  # (Nr, Nz): r_i * E_theta[i,j]
        # Difference at interior faces: rE[i] - rE[i-1] for i=1..Nr-1
        d_rE = rE[1:, :] - rE[:-1, :]  # shape (Nr-1, Nz)
        inv_drc = 1.0 / mesh.drc  # shape (Nr-1,)
        inv_rf_interior = 1.0 / mesh.rf[1:-1]  # shape (Nr-1,) — faces 1..Nr-1
        H_z[1:-1, :] -= dt_over_mu0 * (inv_rf_interior[:, None] *
                                         d_rE * inv_drc[:, None])
        # Axis face (i=0): L'Hopital: (1/r)*d(rE)/dr -> 2*dE/dr at r=0
        # Since E_theta[0,:]=0: d(rE)/dr|_0 = rc[0]*E[0]/... but E[0]=0,
        # so use: lim_{r->0} (1/r)*d(rE)/dr = 2*dE/dr = 2*(E[1]-E[0])/dr[0] = 2*E[1]/dr[0]
        # But H_z[0] is at rf[0]=0, and this is ON the axis.
        # The correct axis treatment: H_z at r=0 is actually well-defined,
        # use 2*E_theta[0,:]/dr[0] but E_theta[0,:]=0, so H_z[0,:] ~ 2*E[1]/dr[0]
        # Actually H_z[0] at rf[0]=0 is not needed; H_z[1] at rf[1] is the first interior face.
        # The axis H_z[0] only appears in the E_theta update for i=0, but E_theta[0]=0 always.
        # So we can safely skip H_z[0] update — it doesn't affect any active E_theta cell.

        # -- 3. Update E_theta (vectorised) --
        # dE/dt = Ca*E + Cb*(dH_r/dz - dH_z/dr)
        # dH_r/dz at cell (i,j) = (H_r[i,j] - H_r[i,j-1]) / dz[j]
        # dH_z/dr at cell (i,j) = (H_z[i+1,j] - H_z[i,j]) / dr[i]
        # Skip i=0 (axis, E=0) and apply PEC mask after

        # dH_r/dz: H_r[i,j] is at face j+1/2, so dH_r/dz at cell j uses faces j-1/2 and j+1/2
        # = (H_r[i, j] - H_r[i, j-1]) / dz[j]  for j >= 1
        # For j=0: H_r at j=-1/2 doesn't exist — use PEC (wafer) -> H_r boundary = 0 or extrapolate
        dHr_dz = np.zeros((Nr, Nz))
        dHr_dz[:, 1:] = (H_r[:, 1:] - H_r[:, :-1]) / mesh.dz[1:][None, :]
        dHr_dz[:, 0] = H_r[:, 0] / mesh.dz[0]  # assume H_r at j=-1/2 = 0 (PEC wafer)

        # dH_z/dr: H_z[i,j] is at face i+1/2, so dH_z/dr at cell i uses faces i-1/2 and i+1/2
        # = (H_z[i+1,j] - H_z[i,j]) / dr[i]
        dHz_dr = (H_z[1:, :] - H_z[:-1, :]) / mesh.dr[:, None]  # shape (Nr, Nz)

        # Curl = dH_r/dz - dH_z/dr
        curl_H = dHr_dz - dHz_dr

        # Update E_theta everywhere, then enforce BCs
        E_theta_new = Ca * E_theta + Cb * curl_H

        # -- 4. Source injection --
        # Linear ramp-up over the first 2 RF cycles, then constant amplitude
        # for the remainder — appropriate for a CW-driven steady-state ICP.
        ramp = min(1.0, n / max(ramp_end, 1))
        J_signal = I_peak * np.sin(omega * n * dt) * ramp
        for (ci, cj, area_inv) in coil_J_factor:
            # J_theta = I_peak / cell_area [A/m^2]
            # dE/dt += -J/eps => E -= dt * J / eps, but using Cb coefficient:
            E_theta_new[ci, cj] -= Cb[ci, cj] * J_signal * area_inv

        # -- 5. Apply boundary conditions --
        # Axis symmetry
        E_theta_new[0, :] = 0.0
        # PEC walls
        E_theta_new[pec_mask] = 0.0
        # Mur ABC at r = R_proc (outer radial boundary)
        E_theta_new[-1, :] = E_adj_rN + coeff_mur_r * (E_theta_new[-2, :] - E_bnd_rN)

        E_theta = E_theta_new

        # -- 6. Accumulate RMS over last RF cycle --
        if n >= accum_start:
            E2_accum += E_theta**2
            n_accum += 1

    # -- Post-processing --
    E_theta_rms = np.sqrt(E2_accum / max(n_accum, 1))
    B_r = mu_0 * H_r
    B_z = mu_0 * H_z[:-1, :]  # trim extra face to match (Nr, Nz)

    E_max = np.abs(E_theta).max()
    E_rms_max = E_theta_rms.max()
    logger.info(f"Cylindrical FDTD complete: |E_theta|_max={E_max:.4e} V/m, "
                f"|E_theta_rms|_max={E_rms_max:.4e} V/m")

    return {
        'E_theta': E_theta,
        'E_theta_rms': E_theta_rms,
        'H_r': H_r,
        'H_z': H_z[:-1, :],  # trim to (Nr, Nz) for consistency
        'B_r': B_r,
        'B_z': B_z,
        'dt_fdtd': dt,
        'cfl': cfl,
        'n_steps_fdtd': n_steps,
    }


def run(state, config):
    """Pipeline-compatible entry point for cylindrical FDTD (M06c)."""
    from ..core.geometry import compute_coil_positions_cylindrical, build_em_active_mask

    mesh = state['mesh']
    inside = state['inside']
    bc_type = state['bc_type']

    circ = config.circuit if hasattr(config, 'circuit') else config.get('circuit', {})
    sim = config.simulation if hasattr(config, 'simulation') else config.get('simulation', {})
    rgeom = config.reactor_geometry if hasattr(config, 'reactor_geometry') else config.get('reactor_geometry', {})
    geom = config.geometry if hasattr(config, 'geometry') else config.get('geometry', {})

    coil_positions = compute_coil_positions_cylindrical(config, mesh)
    I_peak = state.get('I_peak', 5.0)
    sigma_plasma = state.get('sigma_plasma', None)

    n_rf_cycles = sim.get('fdtd_rf_cycles', 2.0)
    R_icp = rgeom.get('R_icp', 0.038)
    qz_thick = rgeom.get('quartz_wall_thickness', 0.0005)
    z_apt_top = rgeom.get('L_proc', 0.050) + rgeom.get('L_apt', 0.002)
    z_top = z_apt_top + rgeom.get('L_icp', 0.1815)

    # Build EM-active mask extending through quartz wall to coil
    R_coil = geom.get('coil_r_position', 0.0405)
    coil_wire_r = geom.get('coil_radius', 0.001)
    R_outer = R_coil + coil_wire_r + 0.001

    em_active, eps_r_map = build_em_active_mask(
        mesh, inside, bc_type, R_icp, R_outer, z_apt_top, z_top)

    # Store coil positions for plotting
    state['coil_positions_cyl'] = coil_positions

    fields = run_fdtd_cylindrical(
        mesh=mesh,
        inside=inside,
        bc_type=bc_type,
        coil_positions=coil_positions,
        source_frequency=circ['source_frequency'],
        I_peak=I_peak,
        n_rf_cycles=n_rf_cycles,
        sigma_plasma=sigma_plasma,
        R_icp=R_icp,
        quartz_wall_thickness=qz_thick,
        z_apt_top=z_apt_top,
        em_active=em_active,
        eps_r_map=eps_r_map,
    )

    return fields
