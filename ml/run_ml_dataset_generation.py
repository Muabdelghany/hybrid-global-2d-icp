#!/usr/bin/env python3
"""Generate the ML training dataset on the production solver.

Training grid (220 operating points):
  * P_rf  in {200, 300, 400, ..., 1200} W           (11 points)
  * p     in {3, 5, 10, 15, 20} mTorr               (5 points)
  * x_Ar  in {0.0, 0.1, 0.3, 0.5}                    (4 points)
  * bias  = ON, 200 W                                (fixed)
  * gamma_Al = 0.155 (calibrated), lambda_exp = 3.20 (inert under the
    f_e_bias closure), R_coil = 0.8

Two rate modes are generated (to reproduce main.pdf's legacy-vs-LXCat split):

  --mode legacy  : Maxwellian Arrhenius (use_boltzmann_rates=False, default)
  --mode lxcat   : Tier-2 PINN Biagi-v10.6 (use_boltzmann_rates=True)

Output: results/ml_dataset/<mode>/<case_id>/{summary.json, nF.npy, nSF6.npy, ...}
        results/ml_dataset/<mode>/dataset_manifest.json

Parallelism: multiprocessing.Pool(8) by default. Set ML_DATASET_SERIAL=1 for debug.
"""
import argparse
import json
import os
import sys
import time
from itertools import product
from multiprocessing import Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PACKAGING NOTE: this is the production copy (md5 05e77aa61919223bf98aef0eda956631,
# the production tree's scripts/ directory), the script that generated the deployed
# 220-case dataset. Only PROJECT_ROOT is changed: the solver lives in model/ here.
# Override with DTPM_MODEL_ROOT for a solver tree elsewhere.
#
# The production closure is NOT set here -- it is injected via DTPM_FIX_FLAGS, which
# run_parameter_sweeps.run_simulation() merges. Point it at model/production_flags.json or the
# dataset will be generated with the the earlier baseline defaults. See cluster/ml/00_dataset_gen.sbatch.
PROJECT_ROOT = os.environ.get(
    'DTPM_MODEL_ROOT',
    os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'model')))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))


CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'default_config.yaml')
OUTPUT_BASE = os.path.join(PROJECT_ROOT, 'results', 'ml_dataset')


POWERS = [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
PRESSURES = [3.0, 5.0, 10.0, 15.0, 20.0]
X_AR_VALUES = [0.0, 0.1, 0.3, 0.5]
BIAS_ENABLED = True
P_BIAS_W = 200
LAMBDA_EXP = 3.20
R_COIL = 0.8
GAMMA_AL = 0.155   # the calibrated value (was 0.18)


def case_id(power, pressure, x_Ar):
    return f"P{power:04d}W_p{int(pressure):02d}mT_xAr{int(round(x_Ar*100)):03d}"


def run_one(task):
    mode, power, pressure, x_Ar = task
    tag = case_id(power, pressure, x_Ar)
    out_dir = os.path.join(OUTPUT_BASE, mode, tag)
    os.makedirs(out_dir, exist_ok=True)

    from run_parameter_sweeps import run_simulation, save_sweep_point

    overrides = {
        'circuit.source_power': power,
        'circuit.R_coil': R_COIL,
        'operating.pressure_mTorr': pressure,
        'operating.frac_Ar': x_Ar,
        'bias.enabled': BIAS_ENABLED,
        'bias.P_bias_W': P_BIAS_W,
        'bias.lambda_exp': LAMBDA_EXP,
        'wall_chemistry.gamma_Al': GAMMA_AL,
        'chemistry.use_boltzmann_rates': (mode == 'lxcat'),
    }
    t0 = time.time()
    try:
        state, mesh, inside, rgeom, config, r0D = run_simulation(overrides, CONFIG_PATH)
        elapsed = time.time() - t0
        s = save_sweep_point(state, mesh, inside, r0D, config, out_dir, tag)
        s['case_id'] = tag
        s['mode'] = mode
        s['P_rf_W'] = int(power)
        s['p_mTorr'] = float(pressure)
        s['x_Ar'] = float(x_Ar)
        s['bias_enabled'] = BIAS_ENABLED
        s['P_bias_W'] = P_BIAS_W
        s['gamma_Al'] = GAMMA_AL
        s['lambda_exp'] = LAMBDA_EXP
        s['R_coil'] = R_COIL
        s['nF_centre_wafer_cm3'] = float(state['nF'][0, 0]) * 1e-6
        s['eta_computed'] = float(state.get('eta_computed', 0.0))
        s['I_peak_final'] = float(state.get('I_peak_final', 0.0))
        s['R_plasma_final'] = float(state.get('R_plasma_final', 0.0))
        s['V_peak_final'] = float(state.get('V_peak_final', 0.0))
        s['P_abs_final'] = float(state.get('P_abs_final', state.get('P_abs', 0.0)))
        s['converged'] = True
        s['elapsed_sec'] = elapsed
        with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
            json.dump(s, f, indent=2)
        return {
            'case_id': tag, 'mode': mode, 'status': 'ok',
            'P_rf_W': power, 'p_mTorr': pressure, 'x_Ar': x_Ar,
            'nF_centre_wafer_cm3': s['nF_centre_wafer_cm3'],
            'F_drop_pct': float(s['F_drop_pct']),
            'eta': s['eta_computed'],
            'elapsed_sec': elapsed,
        }
    except Exception as e:
        with open(os.path.join(out_dir, 'FAILED.txt'), 'w') as f:
            f.write(f"{type(e).__name__}: {e}\n")
        return {'case_id': tag, 'mode': mode, 'status': 'failed',
                'P_rf_W': power, 'p_mTorr': pressure, 'x_Ar': x_Ar,
                'error': f"{type(e).__name__}: {e}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['legacy', 'lxcat'], required=True,
                        help="Rate-coefficient mode")
    parser.add_argument('--workers', type=int, default=None,
                        help="Override default pool size")
    args = parser.parse_args()

    mode = args.mode
    os.makedirs(os.path.join(OUTPUT_BASE, mode), exist_ok=True)

    tasks = [(mode, p, pp, xa)
             for p, pp, xa in product(POWERS, PRESSURES, X_AR_VALUES)]
    n_workers = (args.workers if args.workers is not None
                 else (1 if os.environ.get('ML_DATASET_SERIAL') == '1'
                       else min(8, len(tasks))))
    print(f"# ML dataset generation — mode={mode}")
    print(f"# {len(tasks)} runs "
          f"({len(POWERS)}P x {len(PRESSURES)}p x {len(X_AR_VALUES)}xAr), "
          f"{n_workers} workers")
    print(f"# Locked: gamma_Al={GAMMA_AL}, lambda_exp={LAMBDA_EXP}, "
          f"R_coil={R_COIL}, bias={'ON' if BIAS_ENABLED else 'OFF'} "
          f"@ {P_BIAS_W}W")

    t0 = time.time()
    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            results = pool.map(run_one, tasks)
    else:
        results = [run_one(t) for t in tasks]
    wall = time.time() - t0

    ok_runs = [r for r in results if r.get('status') == 'ok']
    failed_runs = [r for r in results if r.get('status') != 'ok']

    manifest = {
        'mode': mode,
        'grid': {
            'P_rf_W': POWERS,
            'p_mTorr': PRESSURES,
            'x_Ar': X_AR_VALUES,
        },
        'locked_parameters': {
            'gamma_Al': GAMMA_AL,
            'lambda_exp': LAMBDA_EXP,  # inert under bias.closure=f_e_bias (see fix_provenance)
            'R_coil': R_COIL,
            'bias_enabled': BIAS_ENABLED,
            'P_bias_W': P_BIAS_W,
            'use_boltzmann_rates': (mode == 'lxcat'),
        },
        'n_total': len(tasks),
        'n_ok': len(ok_runs),
        'n_failed': len(failed_runs),
        'wall_clock_sec': wall,
        'failed_cases': [r['case_id'] for r in failed_runs],
        'runs': sorted(ok_runs, key=lambda r: (r['P_rf_W'], r['p_mTorr'], r['x_Ar'])),
    }
    manifest_path = os.path.join(OUTPUT_BASE, mode, 'dataset_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n# Dataset generation complete — {len(ok_runs)}/{len(tasks)} ok, "
          f"{len(failed_runs)} failed")
    print(f"# Wall-clock: {wall:.1f}s ({wall/60:.1f} min)")
    print(f"# Manifest: {manifest_path}")
    if failed_runs:
        print(f"# Failed cases ({len(failed_runs)}):")
        for r in failed_runs[:10]:
            print(f"    {r['case_id']} - {r.get('error', 'unknown')}")


if __name__ == '__main__':
    main()
