#!/usr/bin/env python3
"""Parallel wrapper around ablation_study.py — dispatches the 5 config variants
across a ProcessPoolExecutor for concurrent CPU training.

Usage:
    ML_DATASET_MODE=legacy python scripts/ml/run_ablation_parallel.py --workers 4
    ML_DATASET_MODE=lxcat  python scripts/ml/run_ablation_parallel.py --workers 4

Output: results/ml_ablation_<mode>/ablation_results.{json,md}
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


EXPERIMENTS = [
    {'name': 'none',         'bias': False, 'reg': False, 'epochs': 1500},
    {'name': 'bias_only',    'bias': True,  'reg': False, 'epochs': 1500},
    {'name': 'reg_only',     'bias': False, 'reg': True,  'epochs': 1500},
    {'name': 'epochs_only',  'bias': False, 'reg': False, 'epochs': 2000},
    {'name': 'all_three',    'bias': True,  'reg': True,  'epochs': 2000},
]

SEEDS = [42, 179, 316]


def _worker(exp):
    """Top-level worker: runs one ablation config over 3 seeds."""
    import ablation_study as abl
    import torch
    import numpy as np

    dev = torch.device('cpu')
    train_d, val_d = abl.load_dataset()
    Xt, Yt = abl.to_tensors(train_d, dev)
    Xv, Yv = abl.to_tensors(val_d, dev)

    name = exp['name']
    runs = []
    t0 = time.time()
    for seed in SEEDS:
        m = abl.train_and_eval(
            exp['bias'], exp['reg'], exp['epochs'], seed,
            Xt, Yt, Xv, Yv, train_d, dev,
        )
        runs.append(m)
    dt = time.time() - t0

    nF_vals = [r['nF']['rmse'] for r in runs]
    nSF6_vals = [r['nSF6']['rmse'] for r in runs]
    return {
        'name': name,
        'bias_init': exp['bias'],
        'physics_reg': exp['reg'],
        'epochs': exp['epochs'],
        'nF_rmse_mean': float(np.mean(nF_vals)),
        'nF_rmse_std': float(np.std(nF_vals)),
        'nSF6_rmse_mean': float(np.mean(nSF6_vals)),
        'nSF6_rmse_std': float(np.std(nSF6_vals)),
        'time_s': dt,
        'runs': runs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    import ablation_study as abl
    os.makedirs(abl.OUT, exist_ok=True)

    print(f"{'='*72}")
    print(f"  Parallel ablation — mode={os.environ.get('ML_DATASET_MODE', 'legacy')}")
    print(f"  {len(EXPERIMENTS)} configs x {len(SEEDS)} seeds, {args.workers} workers")
    print(f"  Output: {abl.OUT}")
    print(f"{'='*72}", flush=True)

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_worker, e): e['name'] for e in EXPERIMENTS}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                summary = fut.result()
                results.append(summary)
                print(f"  [OK] {name}: nF={summary['nF_rmse_mean']:.5f}+/-"
                      f"{summary['nF_rmse_std']:.5f}  "
                      f"nSF6={summary['nSF6_rmse_mean']:.5f}", flush=True)
            except Exception as e:
                print(f"  [FAIL] {name}: {type(e).__name__}: {e}", flush=True)

    wall = time.time() - t0
    # Preserve canonical order for the markdown table / improvement analysis
    order = {e['name']: i for i, e in enumerate(EXPERIMENTS)}
    results.sort(key=lambda r: order.get(r['name'], 999))

    with open(os.path.join(abl.OUT, 'ablation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Write markdown summary (mirrors the serial script)
    md = "# Ablation Study: Training Recipe Components\n\n"
    md += f"Dataset: ml_dataset / {os.environ.get('ML_DATASET_MODE', 'legacy')}\n"
    md += "Architecture: Fourier + 4x128 GELU MLP\n"
    md += "3 seeds per experiment\n\n"
    md += "| Experiment | Bias Init | Phys Reg | Epochs | nF RMSE | nSF6 RMSE | Time |\n"
    md += "|---|---|---|---|---|---|---|\n"
    for r in results:
        md += (f"| {r['name']} | {'Y' if r['bias_init'] else 'N'} | "
               f"{'Y' if r['physics_reg'] else 'N'} | {r['epochs']} | "
               f"{r['nF_rmse_mean']:.5f}+/-{r['nF_rmse_std']:.5f} | "
               f"{r['nSF6_rmse_mean']:.5f}+/-{r['nSF6_rmse_std']:.5f} | "
               f"{r['time_s']:.0f}s |\n")
    if results:
        base = results[0]['nF_rmse_mean']
        md += "\n## Relative improvement vs baseline (none)\n\n"
        for r in results[1:]:
            improv = (base - r['nF_rmse_mean']) / base * 100
            md += f"- **{r['name']}**: {improv:+.1f}% nF improvement\n"
    md += f"\n\nWall-clock: {wall:.1f} s\n"
    with open(os.path.join(abl.OUT, 'ablation_results.md'), 'w') as f:
        f.write(md)

    print(f"\n  Wall-clock: {wall:.1f}s ({wall/60:.1f} min)")
    print(f"  JSON: {abl.OUT}/ablation_results.json")
    print(f"  MD:   {abl.OUT}/ablation_results.md")


if __name__ == '__main__':
    main()
