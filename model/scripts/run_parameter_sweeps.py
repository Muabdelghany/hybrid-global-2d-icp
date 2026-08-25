#!/usr/bin/env python3
"""
Run the parameter sweeps.

Power sweep: 200, 300, 400, 500, 600, 700, 800, 900, 1000 W at 10 mTorr
Pressure sweep: 5, 7, 10, 15, 20, 30 mTorr at 700 W
Sensitivity sweep: ±50% variation in gamma_Al, eta, D_F, pressure

Each run saves all 9 neutral species + charged species + ne/Te/P_rz.
Also runs 0D Lallement model at each point for benchmark comparison.

Total: ~25 runs × 55s = ~23 min
"""

import os
import sys
import numpy as np
import yaml
import tempfile
import time
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from dtpm.core import SimulationConfig, Mesh2D, build_geometry_mask, build_index_maps
from dtpm.core.grid import Grid
from dtpm.core.geometry import compute_coil_positions_cylindrical
from dtpm.modules import m01_circuit, m06_fdtd_cylindrical, m11_plasma_chemistry
from dtpm.chemistry.global_model import solve_0D

import logging
logging.getLogger('dtpm').setLevel(logging.WARNING)


def run_simulation(config_overrides, config_path):
    """Run a single simulation with config overrides, return all data."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Stage-3 fix-flag injection: if DTPM_FIX_FLAGS points to a JSON file (or
    # holds inline JSON), merge those dotted-key overrides for any key the
    # caller did NOT set explicitly. One site covers every sweep driver (they
    # all funnel through run_simulation); explicit caller switches still win.
    _ff = os.environ.get('DTPM_FIX_FLAGS', '').strip()
    if _ff:
        import json as _json
        try:
            _flags = _json.loads(open(_ff).read()) if os.path.exists(_ff) else _json.loads(_ff)
        except Exception as _e:
            raise RuntimeError(f'DTPM_FIX_FLAGS could not be parsed: {_e}')
        _added = {k: v for k, v in _flags.items() if k not in config_overrides}
        config_overrides = dict(config_overrides); config_overrides.update(_added)
        print(f'[flags] merged {len(_added)}/{len(_flags)} fix flags from DTPM_FIX_FLAGS')

    # Apply overrides (auto-creates missing nested sections so that new
    # config namespaces like `chemistry.use_boltzmann_rates` can be set
    # even when the default_config.yaml does not declare them).
    for key_path, value in config_overrides.items():
        keys = key_path.split('.')
        d = cfg
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(cfg, tmp, default_flow_style=False)
    tmp.close()

    config = SimulationConfig(tmp.name)
    rgeom = config.reactor_geometry
    L_total = rgeom['L_proc'] + rgeom['L_apt'] + rgeom['L_icp']
    mesh = Mesh2D(R=rgeom['R_proc'], L=L_total, Nr=rgeom['Nr'], Nz=rgeom['Nz'],
                  beta_r=rgeom['beta_r'], beta_z=rgeom['beta_z'])
    inside, bc = build_geometry_mask(mesh, rgeom['R_icp'], rgeom['R_proc'],
                                     rgeom['L_proc'], rgeom['L_proc']+rgeom['L_apt'], L_total,
                                     R_wafer=rgeom.get('R_wafer', 0.075))
    ijf, fij, na = build_index_maps(inside)
    grid = Grid(config)

    state = {
        'mesh': mesh, 'inside': inside, 'bc_type': bc,
        'ij_to_flat': ijf, 'flat_to_ij': fij, 'n_active': na,
        'nx': grid.nx, 'ny': grid.ny, 'dx': grid.dx, 'dy': grid.dy,
        'coil_centers_idx': compute_coil_positions_cylindrical(config, mesh),
        'coil_radius': config.geometry.get('coil_radius', 1e-3),
        'coil_radius_idx': 1,
    }

    state.update(m01_circuit.run(state, config))
    state.update(m06_fdtd_cylindrical.run(state, config))
    state.update(m11_plasma_chemistry.run(state, config))

    os.unlink(tmp.name)

    # Also run 0D model for benchmark with the 2D's self-consistent eta
    # (the 0D solver has no spatial E-field so it must be given eta as input;
    #  we use the converged value from m11 for a like-for-like comparison).
    oper = config.operating
    eta_2D = float(state.get('eta_computed',
                             state.get('eta_circuit', 0.43)))
    result_0D = solve_0D(
        P_rf=config.circuit['source_power'],
        p_mTorr=oper.get('pressure_mTorr', 10),
        frac_Ar=oper.get('frac_Ar', 0.0),
        Q_sccm=oper.get('Q_sccm', 100),
        Tgas=oper.get('Tgas', 313),
        eta=eta_2D,
        R_icp=rgeom.get('R_icp', 0.038),
        L_icp=rgeom.get('L_icp', 0.150),
    )

    return state, mesh, inside, rgeom, config, result_0D


def save_sweep_point(state, mesh, inside, result_0D, config, out_dir, label):
    """Save all data for one sweep point."""
    # Provenance: record the fix-flag state in EVERY summary so attribution never
    # depends on job-log archaeology (tail-truncated logs hid the overrides once).
    _coup = config.coupling if hasattr(config, 'coupling') else {}
    _chem = config.chemistry if hasattr(config, 'chemistry') else {}
    _bias_sec = config.bias if hasattr(config, 'bias') else {}
    if not _bias_sec and hasattr(config, 'get'):
        _bias_sec = config.get('bias', {}) or {}
    _wallc = config.wall_chemistry if hasattr(config, 'wall_chemistry') else {}
    if not _wallc and hasattr(config, 'get'):
        _wallc = config.get('wall_chemistry', {}) or {}
    fix_provenance = {
        'te_closure': _coup.get('te_closure', 'local_balance'),
        'outer_loop': bool(_coup.get('outer_loop', False)),
        'per_cell_rates': bool(_chem.get('per_cell_rates', False)),
        'depleted_attachment': bool(_chem.get('depleted_attachment', False)),
        'ec_clip_wide': bool(_chem.get('ec_clip_wide', False)),
        'ne_cap_0D': float(_chem.get('ne_cap_0D', 1e19)),
        'reconciled_rates': bool(_chem.get('reconciled_rates', False)),
        'alpha_fixed': _chem.get('alpha_fixed', None),
        # Read the bias section exactly as m12 reads it, so the recorded provenance
        # and the values the model used cannot diverge.
        'bias_closure': str(_bias_sec.get('closure', 'lambda_exp')),
        'f_e_bias': float(_bias_sec.get('f_e_bias', 0.5)),
        'gamma_Al': float(_wallc.get('gamma_Al', -1.0)),
        'bias_in_te': bool(_coup.get('bias_in_te', False)),
    }
    os.makedirs(out_dir, exist_ok=True)

    # Save all species fields
    species = state.get('species_fields', {})
    for sp_name, sp_field in species.items():
        np.save(os.path.join(out_dir, f'n{sp_name}.npy'), sp_field)

    # Save ne, Te, P_rz
    for name in ['ne', 'Te', 'P_rz', 'E_theta_rms']:
        if name in state and hasattr(state[name], 'shape'):
            np.save(os.path.join(out_dir, f'{name}.npy'), state[name])

    # Save charged species
    ions = state.get('ions', {})
    for ion_name, ion_field in ions.items():
        if hasattr(ion_field, 'shape'):
            np.save(os.path.join(out_dir, f'ion_{ion_name}.npy'), ion_field)

    # Compute and save summary metrics
    nF = species.get('F', state.get('nF', np.zeros((mesh.Nr, mesh.Nz))))
    Fw = nF[:, 0][inside[:, 0]]
    drop = (1 - Fw[-1] / max(Fw[0], 1e-10)) * 100 if len(Fw) > 1 else 0

    Nr, Nz = mesh.Nr, mesh.Nz
    R_icp = config.reactor_geometry.get('R_icp', 0.038)
    L_proc = config.reactor_geometry.get('L_proc', 0.050)
    L_apt = config.reactor_geometry.get('L_apt', 0.002)
    z_apt_top = L_proc + L_apt

    icp_mask = (inside &
                (np.outer(mesh.rc, np.ones(Nz)) <= R_icp) &
                (np.outer(np.ones(Nr), mesh.zc) >= z_apt_top))
    proc_mask = inside & (np.outer(np.ones(Nr), mesh.zc) < L_proc)

    ne = state.get('ne', np.zeros((Nr, Nz)))

    summary = {
        'label': label,
        'power': config.circuit['source_power'],
        'pressure': config.operating.get('pressure_mTorr', 10),
        'F_drop_pct': float(drop),
        'fix_provenance': fix_provenance,
        'ne_avg_icp': float(np.sum(ne[icp_mask] * mesh.vol[icp_mask]) / max(np.sum(mesh.vol[icp_mask]), 1e-30)) * 1e-6,
        'Te_avg': float(state.get('convergence_history', [{}])[-1].get('Te_avg', 0)) if state.get('convergence_history') else 0,
        'eta': float(state.get('eta_computed', 0.43)),
    }

    # Volume-averaged species densities (ICP and processing)
    for sp_name in ['SF6', 'SF5', 'SF4', 'SF3', 'SF2', 'SF', 'S', 'F', 'F2']:
        field = species.get(sp_name, np.zeros((Nr, Nz)))
        summary[f'{sp_name}_icp'] = float(np.sum(field[icp_mask] * mesh.vol[icp_mask]) / max(np.sum(mesh.vol[icp_mask]), 1e-30) * 1e-6)
        summary[f'{sp_name}_proc'] = float(np.sum(field[proc_mask] * mesh.vol[proc_mask]) / max(np.sum(mesh.vol[proc_mask]), 1e-30) * 1e-6)

    # 0D Lallement reference values
    summary['0D_Te'] = float(result_0D.get('Te', 0))
    summary['0D_ne'] = float(result_0D.get('ne', 0)) * 1e-6
    summary['0D_alpha'] = float(result_0D.get('alpha', 0))
    for sp_name in ['SF6', 'SF5', 'SF4', 'SF3', 'SF2', 'SF', 'S', 'F', 'F2']:
        key = f'n_{sp_name}' if sp_name != 'SF' else 'n_SF'
        summary[f'0D_{sp_name}'] = float(result_0D.get(key, 0)) * 1e-6

    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    config_path = os.path.join(PROJECT_ROOT, 'config', 'default_config.yaml')
    sweep_base = os.path.join(PROJECT_ROOT, 'results', 'sweeps')
    os.makedirs(sweep_base, exist_ok=True)

    t0 = time.time()

    # ===================================================
    # Power sweep: 200-1000 W at 10 mTorr
    # ===================================================
    powers = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
    print(f"{'='*60}")
    print(f"  Power Sweep: {powers} W at 10 mTorr")
    print(f"{'='*60}")

    power_summaries = []
    for i, p in enumerate(powers):
        print(f"  [{i+1}/{len(powers)}] P = {p} W...", end=' ', flush=True)
        t1 = time.time()
        state, mesh, inside, rgeom, config, r0D = run_simulation(
            {'circuit.source_power': p}, config_path)
        out_dir = os.path.join(sweep_base, 'power', f'P{p:04d}W')
        summary = save_sweep_point(state, mesh, inside, r0D, config, out_dir, f'{p}W')
        power_summaries.append(summary)
        print(f"drop={summary['F_drop_pct']:.1f}%, ne={summary['ne_avg_icp']:.1e} ({time.time()-t1:.0f}s)")

    # Save power sweep index
    with open(os.path.join(sweep_base, 'power', 'index.json'), 'w') as f:
        json.dump(power_summaries, f, indent=2)

    # ===================================================
    # Pressure sweep: 5-30 mTorr at 700 W
    # ===================================================
    pressures = [5, 7, 10, 15, 20, 30]
    print(f"\n{'='*60}")
    print(f"  Pressure Sweep: {pressures} mTorr at 700 W")
    print(f"{'='*60}")

    pressure_summaries = []
    for i, p in enumerate(pressures):
        print(f"  [{i+1}/{len(pressures)}] p = {p} mTorr...", end=' ', flush=True)
        t1 = time.time()
        state, mesh, inside, rgeom, config, r0D = run_simulation(
            {'operating.pressure_mTorr': p}, config_path)
        out_dir = os.path.join(sweep_base, 'pressure', f'p{p:02d}mTorr')
        summary = save_sweep_point(state, mesh, inside, r0D, config, out_dir, f'{p}mTorr')
        pressure_summaries.append(summary)
        print(f"drop={summary['F_drop_pct']:.1f}%, ne={summary['ne_avg_icp']:.1e} ({time.time()-t1:.0f}s)")

    with open(os.path.join(sweep_base, 'pressure', 'index.json'), 'w') as f:
        json.dump(pressure_summaries, f, indent=2)

    # ===================================================
    # Sensitivity sweep: ±50% variation at 700W, 10mTorr
    # ===================================================
    print(f"\n{'='*60}")
    print(f"  Sensitivity Sweep")
    print(f"{'='*60}")

    sensitivity_params = [
        ('gamma_Al', 'wall_chemistry.gamma_Al', [0.078, 0.103, 0.129, 0.155, 0.181, 0.206, 0.233]),
        ('eta', 'operating.eta_initial', [0.2, 0.3, 0.4, 0.43, 0.5, 0.6, 0.8]),
        ('pressure', 'operating.pressure_mTorr', [5, 7, 10, 12, 15, 18, 20]),
        ('gamma_wafer', 'wall_chemistry.gamma_wafer', [0.01, 0.015, 0.025, 0.035, 0.04, 0.05, 0.06]),
    ]

    sensitivity_results = {}
    for param_name, param_path, values in sensitivity_params:
        print(f"  {param_name}: {values}")
        results = []
        for v in values:
            state, mesh, inside, rgeom, config, r0D = run_simulation(
                {param_path: v}, config_path)
            species = state.get('species_fields', {})
            nF = species.get('F', state.get('nF', np.zeros((mesh.Nr, mesh.Nz))))
            Fw = nF[:, 0][inside[:, 0]]
            drop = (1 - Fw[-1] / max(Fw[0], 1e-10)) * 100 if len(Fw) > 1 else 0
            results.append({'value': v, 'F_drop': float(drop)})
            print(f"    {param_name}={v}: drop={drop:.1f}%")
        sensitivity_results[param_name] = results

    with open(os.path.join(sweep_base, 'sensitivity.json'), 'w') as f:
        json.dump(sensitivity_results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  All sweeps complete in {elapsed/60:.1f} min")
    print(f"  Power: {len(powers)} points")
    print(f"  Pressure: {len(pressures)} points")
    print(f"  Sensitivity: {sum(len(v) for _, _, v in sensitivity_params)} points")
    print(f"  Data saved to: {sweep_base}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
