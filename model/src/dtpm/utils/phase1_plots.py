"""
Plotting for the Phase 1 merged simulation.

Generates publication-quality figures for:
1. Reactor geometry and mesh
2. FDTD electromagnetic fields (E_theta, H_r, H_z)
3. Power deposition P(r,z)
4. Electron density ne(r,z) and temperature Te(r,z)
5. Neutral species profiles (nF, nSF6)
6. Wafer-level radial profiles
7. Picard convergence history
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Rectangle
import logging

logger = logging.getLogger(__name__)


def _setup_style():
    """Set publication-quality plot style."""
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 200,
        'savefig.bbox': 'tight',
    })


def _make_contour(ax, mesh, field, inside, title, cmap='viridis',
                  log_scale=False, vmin=None, vmax=None, units=''):
    """Helper: contour plot on the masked (r,z) domain."""
    R, Z = np.meshgrid(mesh.rc * 1e3, mesh.zc * 1e3, indexing='ij')
    data = field.copy()
    data[~inside] = np.nan

    if log_scale:
        data = np.where(data > 0, data, np.nan)
        with np.errstate(invalid='ignore'):
            norm = LogNorm(vmin=vmin or np.nanmin(data[data > 0]),
                          vmax=vmax or np.nanmax(data))
        im = ax.pcolormesh(R, Z, data, cmap=cmap, norm=norm, shading='auto')
    else:
        if vmin is None:
            vmin = np.nanmin(data)
        if vmax is None:
            vmax = np.nanmax(data)
        im = ax.pcolormesh(R, Z, data, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')

    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    if units:
        cb.set_label(units)
    ax.set_xlabel('r [mm]')
    ax.set_ylabel('z [mm]')
    ax.set_title(title)
    ax.set_aspect('equal')
    return im


def plot_geometry(mesh, inside, bc_type, coil_positions, config, save_dir):
    """Plot reactor geometry with boundary types and coil positions."""
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    rgeom = config.reactor_geometry if hasattr(config, 'reactor_geometry') else config.get('reactor_geometry', {})
    R_icp = rgeom.get('R_icp', 0.038) * 1e3
    L_proc = rgeom.get('L_proc', 0.050) * 1e3
    L_apt = rgeom.get('L_apt', 0.002) * 1e3

    # Panel 1: Geometry mask (active vs inactive)
    R, Z = np.meshgrid(mesh.rc * 1e3, mesh.zc * 1e3, indexing='ij')
    ax1.pcolormesh(R, Z, inside.astype(float), cmap='Blues', shading='auto')
    # Mark coil positions
    for (ci, cj) in coil_positions:
        ax1.plot(mesh.rc[ci] * 1e3, mesh.zc[cj] * 1e3, 'ro', ms=6, zorder=5)
    # Geometry lines
    ax1.axhline(L_proc, color='gray', ls='--', lw=0.8, label='Aperture')
    ax1.axhline(L_proc + L_apt, color='gray', ls='--', lw=0.8)
    ax1.axvline(R_icp, color='orange', ls='--', lw=0.8, label=f'R_icp={R_icp:.0f}mm')
    ax1.set_xlabel('r [mm]')
    ax1.set_ylabel('z [mm]')
    ax1.set_title('(a) Reactor Geometry — Active Cells')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_aspect('equal')

    # Panel 2: Boundary type map
    bc_plot = bc_type.astype(float).copy()
    bc_plot[bc_type < 0] = -1
    im = ax2.pcolormesh(R, Z, bc_plot, cmap='tab10', vmin=-1, vmax=7, shading='auto')
    cb = plt.colorbar(im, ax=ax2, shrink=0.85)
    bc_labels = {-1: 'Inactive', 0: 'Interior', 1: 'Axis', 2: 'Quartz',
                 3: 'Window', 4: 'Al side', 5: 'Al top', 6: 'Wafer', 7: 'Shoulder'}
    cb.set_ticks(list(bc_labels.keys()))
    cb.set_ticklabels(list(bc_labels.values()))
    ax2.set_xlabel('r [mm]')
    ax2.set_ylabel('z [mm]')
    ax2.set_title('(b) Boundary Types')
    ax2.set_aspect('equal')

    fig.suptitle('IPI ICP Reactor Geometry', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'fig01_geometry.png'))
    plt.close(fig)
    logger.info("  Fig 01: Geometry saved")


def plot_em_fields(state, mesh, inside, config, save_dir):
    """Plot FDTD electromagnetic fields."""
    _setup_style()

    E_theta = state.get('E_theta')
    E_rms = state.get('E_theta_rms')
    H_r = state.get('H_r')
    H_z = state.get('H_z')
    B_r = state.get('B_r')
    B_z = state.get('B_z')

    if E_theta is None:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # E_theta instantaneous
    _make_contour(axes[0, 0], mesh, E_theta, inside,
                  '(a) E_theta (instantaneous)', 'RdBu_r', units='V/m')

    # E_theta RMS
    if E_rms is not None:
        _make_contour(axes[0, 1], mesh, E_rms, inside,
                      '(b) E_theta RMS', 'hot', units='V/m')

    # |E_theta_rms|^2 (proportional to power)
    if E_rms is not None:
        _make_contour(axes[0, 2], mesh, E_rms**2, inside,
                      '(c) |E_theta_rms|^2', 'inferno', units='V^2/m^2')

    # B_r
    if B_r is not None:
        Nr, Nz = mesh.Nr, mesh.Nz
        B_r_plot = B_r[:Nr, :Nz] if B_r.shape[0] >= Nr else B_r
        _make_contour(axes[1, 0], mesh, B_r_plot, inside,
                      '(d) B_r', 'RdBu_r', units='T')

    # B_z
    if B_z is not None:
        B_z_plot = B_z[:Nr, :Nz] if B_z.shape[0] >= Nr else B_z
        _make_contour(axes[1, 1], mesh, B_z_plot, inside,
                      '(e) B_z', 'RdBu_r', units='T')

    # |B| magnitude
    if B_r is not None and B_z is not None:
        B_mag = np.sqrt(B_r_plot**2 + B_z_plot**2)
        _make_contour(axes[1, 2], mesh, B_mag, inside,
                      '(f) |B| magnitude', 'magma', units='T')

    fig.suptitle('FDTD Electromagnetic Fields (Cylindrical TE Mode)', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'fig02_em_fields.png'))
    plt.close(fig)
    logger.info("  Fig 02: EM fields saved")


def plot_power_deposition(state, mesh, inside, config, save_dir):
    """Plot power deposition and plasma conductivity."""
    _setup_style()

    P_rz = state.get('P_rz')
    sigma_rz = state.get('sigma_rz')
    if P_rz is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # P(r,z) linear
    _make_contour(axes[0], mesh, P_rz, inside,
                  '(a) Power Deposition P(r,z)', 'hot', units='W/m^3')

    # P(r,z) log
    P_pos = np.where(P_rz > 0, P_rz, 1e-10)
    _make_contour(axes[1], mesh, P_pos, inside,
                  '(b) Power Deposition (log scale)', 'hot',
                  log_scale=True, units='W/m^3')

    # sigma(r,z)
    if sigma_rz is not None:
        _make_contour(axes[2], mesh, sigma_rz, inside,
                      '(c) Plasma Conductivity sigma(r,z)', 'plasma',
                      log_scale=True, units='S/m')

    P_abs = state.get('P_abs', 0)
    eta = state.get('eta_computed', 0)
    fig.suptitle(f'Power Deposition — P_abs = {P_abs:.1f} W, eta = {eta:.3f}',
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'fig03_power_deposition.png'))
    plt.close(fig)
    logger.info("  Fig 03: Power deposition saved")


def plot_plasma_state(state, mesh, inside, config, save_dir):
    """Plot ne(r,z) and Te(r,z)."""
    _setup_style()

    ne = state.get('ne')
    Te = state.get('Te')
    if ne is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ne linear
    _make_contour(axes[0], mesh, ne, inside,
                  '(a) Electron Density ne(r,z)', 'viridis', units='m^-3')

    # ne log
    _make_contour(axes[1], mesh, np.maximum(ne, 1e8), inside,
                  '(b) ne (log scale)', 'viridis', log_scale=True, units='m^-3')

    # Te
    if Te is not None:
        _make_contour(axes[2], mesh, Te, inside,
                      '(c) Electron Temperature Te(r,z)', 'plasma', units='eV')

    ne_avg = state.get('ne_avg', 0)
    fig.suptitle(f'Plasma State — ne_avg = {ne_avg:.2e} m^-3', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'fig04_plasma_state.png'))
    plt.close(fig)
    logger.info("  Fig 04: Plasma state saved")


def plot_species(state, mesh, inside, config, save_dir):
    """Plot neutral species nF(r,z) and nSF6(r,z)."""
    _setup_style()

    nF = state.get('nF')
    nSF6 = state.get('nSF6')
    if nF is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # nF linear
    _make_contour(axes[0, 0], mesh, nF, inside,
                  '(a) Fluorine [F] Density', 'Reds', units='m^-3')

    # nF log
    _make_contour(axes[0, 1], mesh, np.maximum(nF, 1e10), inside,
                  '(b) [F] (log scale)', 'Reds', log_scale=True, units='m^-3')

    # nSF6
    if nSF6 is not None:
        _make_contour(axes[1, 0], mesh, nSF6, inside,
                      '(c) SF6 Density', 'Blues', units='m^-3')
        _make_contour(axes[1, 1], mesh, np.maximum(nSF6, 1e10), inside,
                      '(d) SF6 (log scale)', 'Blues', log_scale=True, units='m^-3')

    F_drop = state.get('F_drop_pct', 0)
    fig.suptitle(f'Neutral Species — [F] drop = {F_drop:.1f}%', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'fig05_species.png'))
    plt.close(fig)
    logger.info("  Fig 05: Species profiles saved")


def plot_wafer_profiles(state, mesh, inside, config, save_dir):
    """Plot radial profiles at the wafer surface."""
    _setup_style()

    nF = state.get('nF')
    ne = state.get('ne')
    Te = state.get('Te')
    nSF6 = state.get('nSF6')

    if nF is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Wafer profiles (z = 0, j = 0)
    r_mm = mesh.rc[inside[:, 0]] * 1e3

    # [F] at wafer
    F_wafer = nF[:, 0][inside[:, 0]]
    axes[0, 0].plot(r_mm, F_wafer * 1e-6, 'r-o', lw=2, ms=3)
    axes[0, 0].set_xlabel('r [mm]')
    axes[0, 0].set_ylabel('[F] (cm^-3)')
    axes[0, 0].set_title('(a) [F] Radial Profile at Wafer')
    axes[0, 0].grid(True, alpha=0.3)
    if len(F_wafer) > 1:
        Fc, Fe = F_wafer[0], F_wafer[-1]
        drop = (1 - Fe / max(Fc, 1e6)) * 100 if Fc > 1e6 else 0
        axes[0, 0].axhline(Fc * 1e-6, color='gray', ls=':', lw=0.8)
        axes[0, 0].text(0.95, 0.95, f'Drop: {drop:.1f}%',
                        transform=axes[0, 0].transAxes, ha='right', va='top',
                        fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))

    # [F] normalised
    if len(F_wafer) > 0 and F_wafer[0] > 0:
        axes[0, 1].plot(r_mm, F_wafer / F_wafer[0], 'r-o', lw=2, ms=3)
        axes[0, 1].set_xlabel('r [mm]')
        axes[0, 1].set_ylabel('[F] / [F]_center')
        axes[0, 1].set_title('(b) Normalised [F] at Wafer')
        axes[0, 1].set_ylim(0, 1.2)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(0.26, color='green', ls='--', lw=1.5, label='74% drop (Mettler)')
        axes[0, 1].legend()

    # ne at ICP midplane
    if ne is not None:
        j_mid_icp = np.argmin(np.abs(mesh.zc - (mesh.L * 0.6)))
        r_icp = mesh.rc * 1e3
        ne_mid = ne[:, j_mid_icp]
        axes[1, 0].plot(r_icp, ne_mid * 1e-6, 'b-o', lw=2, ms=2)
        axes[1, 0].set_xlabel('r [mm]')
        axes[1, 0].set_ylabel('ne (cm^-3)')
        axes[1, 0].set_title(f'(c) ne Radial Profile at z={mesh.zc[j_mid_icp]*1e3:.0f}mm')
        axes[1, 0].grid(True, alpha=0.3)

    # SF6 at wafer
    if nSF6 is not None:
        SF6_wafer = nSF6[:, 0][inside[:, 0]]
        axes[1, 1].plot(r_mm, SF6_wafer * 1e-6, 'b-o', lw=2, ms=3)
        axes[1, 1].set_xlabel('r [mm]')
        axes[1, 1].set_ylabel('[SF6] (cm^-3)')
        axes[1, 1].set_title('(d) SF6 Radial Profile at Wafer')
        axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle('Wafer-Level Radial Profiles', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'fig06_wafer_profiles.png'))
    plt.close(fig)
    logger.info("  Fig 06: Wafer profiles saved")


def plot_convergence(state, save_dir):
    """Plot Picard convergence history."""
    _setup_style()

    history = state.get('convergence_history')
    if not history:
        return

    iters = [h['iter'] for h in history]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # ne_avg
    axes[0, 0].plot(iters, [h['ne_avg'] * 1e-6 for h in history], 'b-o', lw=2, ms=4)
    axes[0, 0].set_ylabel('ne_avg (cm^-3)')
    axes[0, 0].set_title('(a) ne_avg')
    axes[0, 0].grid(True, alpha=0.3)

    # Te_avg
    axes[0, 1].plot(iters, [h['Te_avg'] for h in history], 'r-o', lw=2, ms=4)
    axes[0, 1].set_ylabel('Te_avg (eV)')
    axes[0, 1].set_title('(b) Te_avg')
    axes[0, 1].grid(True, alpha=0.3)

    # eta
    axes[0, 2].plot(iters, [h['eta'] for h in history], 'g-o', lw=2, ms=4)
    axes[0, 2].set_ylabel('eta')
    axes[0, 2].set_title('(c) Coupling Efficiency')
    axes[0, 2].grid(True, alpha=0.3)

    # P_abs
    axes[1, 0].plot(iters, [h['P_abs'] for h in history], 'k-o', lw=2, ms=4)
    axes[1, 0].set_ylabel('P_abs (W)')
    axes[1, 0].set_xlabel('Picard Iteration')
    axes[1, 0].set_title('(d) Absorbed Power')
    axes[1, 0].grid(True, alpha=0.3)

    # F_drop
    axes[1, 1].plot(iters, [h['F_drop_pct'] for h in history], 'm-o', lw=2, ms=4)
    axes[1, 1].axhline(74, color='green', ls='--', lw=1.5, label='Mettler (74%)')
    axes[1, 1].set_ylabel('[F] drop (%)')
    axes[1, 1].set_xlabel('Picard Iteration')
    axes[1, 1].set_title('(e) [F] Centre-to-Edge Drop')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # ne relative change
    axes[1, 2].semilogy(iters, [h['rel_change'] for h in history], 'b-o', lw=2, ms=4)
    axes[1, 2].axhline(0.02, color='red', ls='--', lw=1, label='Convergence tol')
    axes[1, 2].set_ylabel('||ne_new - ne_old|| / ||ne_old||')
    axes[1, 2].set_xlabel('Picard Iteration')
    axes[1, 2].set_title('(f) Picard Convergence')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    fig.suptitle('Picard Iteration Convergence', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'fig07_convergence.png'))
    plt.close(fig)
    logger.info("  Fig 07: Convergence history saved")


def plot_summary(state, mesh, inside, config, save_dir):
    """Single-page summary figure."""
    _setup_style()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    ne = state.get('ne')
    Te = state.get('Te')
    nF = state.get('nF')
    P_rz = state.get('P_rz')
    E_rms = state.get('E_theta_rms')

    # E_theta RMS
    if E_rms is not None:
        _make_contour(axes[0, 0], mesh, E_rms, inside,
                      'E_theta RMS [V/m]', 'hot', units='V/m')

    # P(r,z)
    if P_rz is not None:
        _make_contour(axes[0, 1], mesh, P_rz, inside,
                      'Power Deposition [W/m^3]', 'hot', units='W/m^3')

    # ne
    if ne is not None:
        _make_contour(axes[0, 2], mesh, np.maximum(ne, 1e8), inside,
                      'ne [m^-3]', 'viridis', log_scale=True, units='m^-3')

    # Te
    if Te is not None:
        _make_contour(axes[1, 0], mesh, Te, inside,
                      'Te [eV]', 'plasma', units='eV')

    # nF
    if nF is not None:
        _make_contour(axes[1, 1], mesh, nF, inside,
                      '[F] [m^-3]', 'Reds', units='m^-3')

    # Wafer [F] profile
    if nF is not None:
        r_mm = mesh.rc[inside[:, 0]] * 1e3
        F_w = nF[:, 0][inside[:, 0]]
        axes[1, 2].plot(r_mm, F_w * 1e-6, 'r-o', lw=2, ms=3)
        if len(F_w) > 0 and F_w[0] > 0:
            axes[1, 2].axhline(F_w[0] * 0.26 * 1e-6, color='green', ls='--',
                               label='74% drop level')
        axes[1, 2].set_xlabel('r [mm]')
        axes[1, 2].set_ylabel('[F] at wafer (cm^-3)')
        axes[1, 2].set_title('Wafer [F] Profile')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

    F_drop = state.get('F_drop_pct', 0)
    eta = state.get('eta_computed', 0)
    P_abs = state.get('P_abs', 0)
    ne_avg = state.get('ne_avg', 0)
    P_rf = config.circuit.get('source_power', 700)
    f_MHz = config.circuit.get('source_frequency', 40e6) / 1e6

    fig.suptitle(
        f'Phase 1 Summary: P_rf={P_rf}W, f={f_MHz:.0f}MHz, '
        f'eta={eta:.3f}, [F] drop={F_drop:.1f}%, ne_avg={ne_avg:.1e} m^-3',
        fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'fig00_summary.png'))
    plt.close(fig)
    logger.info("  Fig 00: Summary saved")


def generate_all_plots(state, mesh, inside, bc_type, config, save_dir):
    """Generate all Phase 1 figures."""
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Generating plots in {save_dir}")

    coil_pos = state.get('coil_positions_cyl', [])

    plot_geometry(mesh, inside, bc_type, coil_pos, config, save_dir)
    plot_em_fields(state, mesh, inside, config, save_dir)
    plot_power_deposition(state, mesh, inside, config, save_dir)
    plot_plasma_state(state, mesh, inside, config, save_dir)
    plot_species(state, mesh, inside, config, save_dir)
    plot_wafer_profiles(state, mesh, inside, config, save_dir)
    plot_convergence(state, save_dir)
    plot_summary(state, mesh, inside, config, save_dir)

    logger.info(f"All plots saved to {save_dir}")
