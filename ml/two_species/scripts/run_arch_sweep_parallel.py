#!/usr/bin/env python3
"""Parallel wrapper around arch_sweep.py — runs the 7 experiments on a
ProcessPoolExecutor so they can share the 8-core CPU.

Each experiment internally runs n_runs=3 seeds sequentially (inside a worker);
concurrency is between experiments, not between seeds. That keeps the
per-worker memory footprint predictable (~1.5 GB torch + data).

Usage:
    ML_DATASET_MODE=legacy python scripts/ml/run_arch_sweep_parallel.py --workers 4
    ML_DATASET_MODE=lxcat  python scripts/ml/run_arch_sweep_parallel.py --workers 4

Results land under results/ml_arch_sweep_<mode>/experiment_table.json (same
schema as the serial arch_sweep.py), so the downstream train_ensemble.py
reads them unchanged.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Experiment registry — pickle-friendly top-level factories
# (avoid lambdas; those don't cross process boundaries).

def _make_E0(sweep):  return sweep.BaselineMLP(n_in=5)
def _make_E1(sweep):  return sweep.BaselineWithBias(n_in=5)
def _make_E2(sweep):  return sweep.WiderDeeperMLP(n_in=5)
def _make_E3(sweep):  return sweep.SeparateHeadsMLP(n_in=5)
def _make_E4(sweep):  return sweep.ResidualMLP(n_in=5)
def _make_E5(sweep):  return sweep.BaselineWithBias(n_in=10)
def _make_E6(sweep):  return sweep.BaselineWithBias(n_in=5)


# (name, make_fn, n_ep, use_physics_reg, use_huber, enhanced)
EXPERIMENTS = [
    ('E0_baseline',          _make_E0, 1500, False, False, False),
    ('E1_v4_recipe',         _make_E1, 2000, True,  False, False),
    ('E2_wider_deeper',      _make_E2, 2000, True,  False, False),
    ('E3_separate_heads',    _make_E3, 2000, True,  False, False),
    ('E4_residual',          _make_E4, 2000, True,  False, False),
    ('E5_enhanced_features', _make_E5, 2000, True,  False, True),
    ('E6_huber_loss',        _make_E6, 2000, True,  True,  False),
]


def _worker(args):
    """Top-level worker: runs one experiment (3 seeds internally)."""
    name, make_fn_name, n_ep, use_physics_reg, use_huber, enhanced = args
    # Import inside worker so each process has its own torch + data
    import arch_sweep as sweep
    # Resolve the factory via its top-level name (pickle-compatible)
    make_fn = globals()[make_fn_name]

    # Force CPU inside workers — MPS GPU can't be shared across processes
    import torch
    dev = torch.device('cpu')

    # Load datasets inside worker (each process has its own tensors)
    train_std, val_std, _meta, _vi = sweep.load_dataset(enhanced_features=False)
    if enhanced:
        train_enh, val_enh, _, _ = sweep.load_dataset(enhanced_features=True)
        train_data, val_data = train_enh, val_enh
    else:
        train_data, val_data = train_std, val_std

    def _factory():
        return make_fn(sweep)

    summary = sweep.run_experiment(
        name, _factory, dev, train_data, val_data,
        n_ep=n_ep, use_physics_reg=use_physics_reg,
        use_huber=use_huber, enhanced=enhanced, n_runs=3,
    )
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4,
                    help="Parallel experiment workers (default 4 for an 8-core CPU)")
    ap.add_argument('--experiments', default=None,
                    help="Comma-separated subset (e.g. E0,E3) — default all 7")
    args = ap.parse_args()

    import arch_sweep as sweep
    os.makedirs(sweep.OUT, exist_ok=True)

    if args.experiments:
        wanted = set(args.experiments.split(','))
        jobs = [e for e in EXPERIMENTS if e[0].split('_')[0] in wanted or e[0] in wanted]
    else:
        jobs = list(EXPERIMENTS)

    # Translate factory function refs into their __name__ strings for pickling
    jobs_pickled = [(j[0], j[1].__name__, j[2], j[3], j[4], j[5]) for j in jobs]

    print(f"{'='*72}")
    print(f"  Parallel arch sweep — mode={os.environ.get('ML_DATASET_MODE', 'legacy')}")
    print(f"  {len(jobs_pickled)} experiments, {args.workers} workers")
    print(f"  Output: {sweep.OUT}")
    print(f"{'='*72}", flush=True)

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_worker, j): j[0] for j in jobs_pickled}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                summary = fut.result()
                results.append(summary)
                print(f"  [OK] {name}: nF={summary['nF_rmse_mean']:.5f} "
                      f"nSF6={summary['nSF6_rmse_mean']:.5f}", flush=True)
            except Exception as e:
                print(f"  [FAIL] {name}: {type(e).__name__}: {e}", flush=True)
                results.append({'name': name, 'error': f"{type(e).__name__}: {e}"})

    wall = time.time() - t0

    # Aggregate into experiment_table.json (same schema as serial arch_sweep.py)
    ok_results = [r for r in results if 'error' not in r]
    # Preserve the canonical experiment order
    order = {name: i for i, (name, *_) in enumerate(EXPERIMENTS)}
    ok_results.sort(key=lambda r: order.get(r['name'], 999))

    table = {
        'experiments': ok_results,
        'reference': {
            'surrogate_lxcat_v3': {'nF_rmse': 0.0112, 'nSF6_rmse': 0.0081},
            'surrogate_v4': {'nF_rmse': 0.0029, 'nSF6_rmse': 0.0027},
        },
        'wall_clock_sec': wall,
        'workers': args.workers,
        'mode': os.environ.get('ML_DATASET_MODE', 'legacy'),
    }
    table_path = os.path.join(sweep.OUT, 'experiment_table.json')
    with open(table_path, 'w') as f:
        json.dump(table, f, indent=2)

    print(f"\n  Wall-clock: {wall:.1f}s ({wall/60:.1f} min)")
    print(f"  Experiment table: {table_path}")
    for r in ok_results:
        print(f"    {r['name']:<22s}  nF={r['nF_rmse_mean']:.5f}  "
              f"nSF6={r['nSF6_rmse_mean']:.5f}")


if __name__ == '__main__':
    main()
