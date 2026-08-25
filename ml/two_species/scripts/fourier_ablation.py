#!/usr/bin/env python3
"""R2.2(a) Fourier-vs-raw ablation on the winning E3 (separate-heads) architecture.

Runs E3 WITH random Fourier features (reproduction) and WITHOUT (raw 5 inputs into the
same shared-trunk + separate-heads network), under the identical v4 recipe (bias init,
physics reg, 2000 epochs, 3 seeds) and the same leakage-free grouped train/val split.
Uses the production arch_sweep machinery so the numbers are directly comparable to the
published architecture sweep (E3_separate_heads nF-RMSE = 0.00210).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/ml
import torch
import torch.nn as nn
import arch_sweep as A


class SepHeadsNoFourier(nn.Module):
    """E3 SeparateHeadsMLP but fed the RAW 5 inputs (no random Fourier features)."""
    def __init__(self, n_in=5, drop=0.05):
        super().__init__()
        ni, nh = n_in, 128
        trunk = [nn.Linear(ni, nh), nn.GELU(), nn.Dropout(drop)]
        for _ in range(2):
            trunk += [nn.Linear(nh, nh), nn.GELU(), nn.Dropout(drop)]
        self.trunk = nn.Sequential(*trunk)
        self.proj = nn.Linear(ni, nh)
        self.head_nF = nn.Sequential(nn.Linear(nh, 64), nn.GELU(), nn.Linear(64, 1))
        self.head_nSF6 = nn.Sequential(nn.Linear(nh, 64), nn.GELU(), nn.Linear(64, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.head_nF[-1].bias.fill_(19.8)
            self.head_nSF6[-1].bias.fill_(20.0)

    def forward(self, x):
        h = self.trunk(x) + self.proj(x)
        return torch.cat([self.head_nF(h), self.head_nSF6(h)], dim=-1)


def main():
    dev, dev_name = A.select_device()
    print(f"Device: {dev} ({dev_name})", flush=True)
    train_std, val_std, meta, vi = A.load_dataset(enhanced_features=False)
    out = {}
    out['E3_with_fourier'] = A.run_experiment(
        'E3_with_fourier', lambda: A.SeparateHeadsMLP(n_in=5),
        dev, train_std, val_std, n_ep=2000, use_physics_reg=True, n_runs=3)
    out['E3_no_fourier'] = A.run_experiment(
        'E3_no_fourier', lambda: SepHeadsNoFourier(n_in=5),
        dev, train_std, val_std, n_ep=2000, use_physics_reg=True, n_runs=3)
    od = os.path.join('results', 'ml_ablation_fourier')
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, 'fourier_ablation.json'), 'w'), indent=2)
    print("RESULTS: E3_with_fourier nF=%.5f+/-%.5f | E3_no_fourier nF=%.5f+/-%.5f" % (
        out['E3_with_fourier']['nF_rmse_mean'], out['E3_with_fourier']['nF_rmse_std'],
        out['E3_no_fourier']['nF_rmse_mean'], out['E3_no_fourier']['nF_rmse_std']), flush=True)


if __name__ == '__main__':
    main()
