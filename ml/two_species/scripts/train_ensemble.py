"""
Train the production LXCat surrogate: 5-model ensemble using the WINNING
architecture from the architecture sweep.

Reads winner from results/lxcat_architecture_upgrade/experiment_table.json.
Refuses to run if the sweep hasn't finished.

Saved as surrogate_lxcat_v4_arch/ — does NOT overwrite any existing models.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
os.environ['MPLBACKEND'] = 'Agg'

R_PROC, Z_TOP = 0.105, 0.234

# Mode-selectable dataset + matching arch-sweep results path
from ml_dataset_loader import load_dataset as _load_dataset_impl
_ML_MODE = os.environ.get('ML_DATASET_MODE', 'legacy')
RESULTS = os.path.join(REPO, 'results', f'ml_production_ensemble_{_ML_MODE}')
SWEEP_TABLE = os.path.join(REPO, 'results', f'ml_arch_sweep_{_ML_MODE}', 'experiment_table.json')


def select_device():
    if torch.cuda.is_available(): return torch.device('cuda'), "CUDA"
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return torch.device('mps'), "MPS"
    return torch.device('cpu'), "CPU"


# ===============================================================
# All candidate architectures (must match lxcat_arch_upgrade.py)
# ===============================================================

class BaselineWithBias(nn.Module):
    """E1/E6: Fourier + 4x128 GELU + bias init."""
    def __init__(self, n_in=5, nf=64, nh=128, nl=4, fs=3.0, drop=0.05):
        super().__init__()
        B = torch.randn(n_in, nf) * fs
        self.register_buffer('B', B)
        ni = 2 * nf
        layers = [nn.Linear(ni, nh), nn.GELU(), nn.Dropout(drop)]
        for _ in range(nl - 1):
            layers += [nn.Linear(nh, nh), nn.GELU(), nn.Dropout(drop)]
        self.bb = nn.Sequential(*layers)
        self.proj = nn.Linear(ni, nh)
        self.head = nn.Linear(nh, 2)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None: nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([19.8, 20.0]))

    def forward(self, x):
        p = x @ self.B
        e = torch.cat([torch.sin(p), torch.cos(p)], -1)
        return self.head(self.bb(e) + self.proj(e))


class WiderDeeperMLP(nn.Module):
    """E2: 6x256 + Fourier + bias init."""
    def __init__(self, n_in=5, nf=64, fs=3.0, drop=0.05):
        super().__init__()
        B = torch.randn(n_in, nf) * fs
        self.register_buffer('B', B)
        ni = 2 * nf
        nh, nl = 256, 6
        layers = [nn.Linear(ni, nh), nn.GELU(), nn.Dropout(drop)]
        for _ in range(nl - 1):
            layers += [nn.Linear(nh, nh), nn.GELU(), nn.Dropout(drop)]
        self.bb = nn.Sequential(*layers)
        self.proj = nn.Linear(ni, nh)
        self.head = nn.Linear(nh, 2)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None: nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([19.8, 20.0]))

    def forward(self, x):
        p = x @ self.B
        e = torch.cat([torch.sin(p), torch.cos(p)], -1)
        return self.head(self.bb(e) + self.proj(e))


class SeparateHeadsMLP(nn.Module):
    """E3: shared Fourier trunk + separate per-species heads."""
    def __init__(self, n_in=5, nf=64, fs=3.0, drop=0.05):
        super().__init__()
        B = torch.randn(n_in, nf) * fs
        self.register_buffer('B', B)
        ni = 2 * nf
        nh = 128
        trunk = [nn.Linear(ni, nh), nn.GELU(), nn.Dropout(drop)]
        for _ in range(2):
            trunk += [nn.Linear(nh, nh), nn.GELU(), nn.Dropout(drop)]
        self.trunk = nn.Sequential(*trunk)
        self.proj = nn.Linear(ni, nh)
        self.head_nF = nn.Sequential(
            nn.Linear(nh, 64), nn.GELU(), nn.Linear(64, 1))
        self.head_nSF6 = nn.Sequential(
            nn.Linear(nh, 64), nn.GELU(), nn.Linear(64, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None: nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.head_nF[-1].bias.fill_(19.8)
            self.head_nSF6[-1].bias.fill_(20.0)

    def forward(self, x):
        p = x @ self.B
        e = torch.cat([torch.sin(p), torch.cos(p)], -1)
        h = self.trunk(e) + self.proj(e)
        return torch.cat([self.head_nF(h), self.head_nSF6(h)], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, dim, drop=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim, dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


class ResidualMLP(nn.Module):
    """E4: Fourier + residual blocks."""
    def __init__(self, n_in=5, nf=64, fs=3.0, drop=0.05):
        super().__init__()
        B = torch.randn(n_in, nf) * fs
        self.register_buffer('B', B)
        ni = 2 * nf
        nh = 128
        self.proj_in = nn.Linear(ni, nh)
        self.blocks = nn.Sequential(*[ResidualBlock(nh, drop) for _ in range(6)])
        self.head = nn.Linear(nh, 2)
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.head:
                nn.init.xavier_normal_(m.weight, gain=0.5)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([19.8, 20.0]))

    def forward(self, x):
        p = x @ self.B
        e = torch.cat([torch.sin(p), torch.cos(p)], -1)
        h = self.proj_in(e)
        h = self.blocks(h)
        return self.head(h)


# Map experiment names to architecture constructors
ARCH_MAP = {
    'E0_baseline':        lambda: BaselineWithBias(n_in=5),  # no bias in E0, but close enough
    'E1_v4_recipe':       lambda: BaselineWithBias(n_in=5),
    'E2_wider_deeper':    lambda: WiderDeeperMLP(n_in=5),
    'E3_separate_heads':  lambda: SeparateHeadsMLP(n_in=5),
    'E4_residual':        lambda: ResidualMLP(n_in=5),
    'E5_enhanced_features': lambda: BaselineWithBias(n_in=10),
    'E6_huber_loss':      lambda: BaselineWithBias(n_in=5),
}

ARCH_NAMES = {
    'E0_baseline':        'BaselineWithBias (Fourier + 4x128 GELU)',
    'E1_v4_recipe':       'BaselineWithBias (Fourier + 4x128 GELU)',
    'E2_wider_deeper':    'WiderDeeperMLP (Fourier + 6x256 GELU)',
    'E3_separate_heads':  'SeparateHeadsMLP (shared trunk + per-species heads)',
    'E4_residual':        'ResidualMLP (Fourier + 6 residual blocks)',
    'E5_enhanced_features': 'BaselineWithBias (10 inputs, enhanced features)',
    'E6_huber_loss':      'BaselineWithBias (Fourier + 4x128 GELU, Huber loss)',
}

# E5 uses enhanced features — need different data loading
ENHANCED_FEATURES = {'E5_enhanced_features'}


# ===============================================================
# Physics regularization
# ===============================================================

def reg_smoothness(model, xb, device):
    xg = xb.detach().clone().requires_grad_(True)
    pred = model(xg)
    L = torch.tensor(0.0, device=device)
    for c in range(2):
        g = torch.autograd.grad(pred[:, c].sum(), xg, create_graph=True)[0]
        L = L + (g[:, 0] ** 2 + g[:, 1] ** 2).mean()
    return L


def reg_bounded_density(pred, y_std):
    lo, hi = 17.0, 21.5
    below = torch.relu(lo - pred).pow(2).mean()
    above = torch.relu(pred - hi).pow(2).mean()
    return (below + above) / y_std.mean() ** 2


def reg_wafer_smoothness(model, data, device, enhanced=False):
    r = torch.linspace(0.001, 0.95, 30, device=device)
    z = torch.full((30,), 0.01, device=device)
    idx = np.random.randint(0, len(data['P']), 1)[0]
    P = torch.full((30,), data['P'][idx] / 1200, device=device)
    p = torch.full((30,), data['p'][idx] / 20, device=device)
    Ar = torch.full((30,), data['Ar'][idx], device=device)
    cols = [r, z, P, p, Ar]
    if enhanced and 'Pp' in data:
        cols += [
            torch.full((30,), data['Pp'][idx] / (1200 * 20), device=device),
            torch.full((30,), data['PAr'][idx] / 1200, device=device),
            torch.full((30,), data['pAr'][idx] / 20, device=device),
            torch.full((30,), data['logp'][idx] / np.log(20), device=device),
            torch.full((30,), data['inv_p'][idx] * 3, device=device),
        ]
    x = torch.stack(cols, dim=-1)
    pred = model(x)
    d2 = pred[2:] - 2 * pred[1:-1] + pred[:-2]
    return d2.pow(2).mean()


# ===============================================================
# Data loading
# ===============================================================

def load_dataset(val_frac=0.15, enhanced=False):
    """Delegates to the dataset adapter in ml_dataset_loader. The mode is taken
    from the ML_DATASET_MODE environment variable."""
    return _load_dataset_impl(mode=_ML_MODE, val_frac=val_frac, enhanced_features=enhanced)


def to_tensors(data, device, enhanced=False):
    cols = [data['r'] / R_PROC, data['z'] / Z_TOP,
            data['P'] / 1200, data['p'] / 20, data['Ar']]
    if enhanced:
        cols += [
            data['Pp'] / (1200 * 20),
            data['PAr'] / 1200,
            data['pAr'] / 20,
            data['logp'] / np.log(20),
            data['inv_p'] * 3,
        ]
    X = np.column_stack(cols)
    Y = np.column_stack([data['lnF'], data['lnSF6']])
    return (torch.tensor(X, dtype=torch.float32, device=device),
            torch.tensor(Y, dtype=torch.float32, device=device))


# ===============================================================
# Main
# ===============================================================

def main():
    # Step 1: Read winner from sweep results
    if not os.path.exists(SWEEP_TABLE):
        print("ERROR: Architecture sweep results not found.", flush=True)
        print(f"  Expected: {SWEEP_TABLE}", flush=True)
        print("  Run lxcat_arch_upgrade.py first.", flush=True)
        sys.exit(1)

    with open(SWEEP_TABLE) as f:
        sweep = json.load(f)

    experiments = sweep.get('experiments', [])
    if not experiments:
        print("ERROR: No experiments found in sweep table.", flush=True)
        sys.exit(1)

    # Select winner: lowest nF RMSE mean, unless an explicit override is set in the table.
    # Override exists for cases where the lowest-mean experiment has materially worse seed
    # variance than a near-tie alternative (see e.g. legacy 2026-04-25 recovery, where
    # E4 won by 2% on mean but had 35% larger sigma than E3).
    if 'winner_override' in sweep:
        winner_name = sweep['winner_override']
        winner = next((e for e in experiments if e['name'] == winner_name), None)
        if winner is None:
            print(f"ERROR: winner_override '{winner_name}' not found among experiments", flush=True)
            sys.exit(1)
    else:
        winner = min(experiments, key=lambda e: e['nF_rmse_mean'])
        winner_name = winner['name']

    if winner_name not in ARCH_MAP:
        print(f"ERROR: Unknown architecture '{winner_name}'", flush=True)
        print(f"  Known: {list(ARCH_MAP.keys())}", flush=True)
        sys.exit(1)

    enhanced = winner_name in ENHANCED_FEATURES
    arch_fn = ARCH_MAP[winner_name]
    arch_label = ARCH_NAMES[winner_name]

    dev, dev_name = select_device()
    n_ens = 5
    n_ep = 2000
    os.makedirs(RESULTS, exist_ok=True)

    print(f"{'=' * 60}", flush=True)
    print(f"  Production Ensemble Training", flush=True)
    print(f"  Winner: {winner_name}", flush=True)
    print(f"  Architecture: {arch_label}", flush=True)
    print(f"  nF RMSE from sweep: {winner['nF_rmse_mean']:.5f}+/-{winner['nF_rmse_std']:.5f}", flush=True)
    print(f"  Device: {dev} ({dev_name})", flush=True)
    print(f"{'=' * 60}", flush=True)

    # Check for existing complete ensemble
    existing = [i for i in range(n_ens)
                if os.path.exists(os.path.join(RESULTS, f'model_{i}.pt'))]
    if len(existing) == n_ens and os.path.exists(os.path.join(RESULTS, 'summary.json')):
        print(f"  All {n_ens} models already exist. Skipping.", flush=True)
        return

    train_d, val_d, meta, vi = load_dataset(enhanced=enhanced)
    Xt, Yt = to_tensors(train_d, dev, enhanced=enhanced)
    Xv, Yv = to_tensors(val_d, dev, enhanced=enhanced)
    ys = Yt.std(0, keepdim=True).clamp(min=1e-3)
    sw = torch.tensor([1.0, 1.5], device=dev)

    print(f"  Train: {Xt.shape[0]} pts, Val: {Xv.shape[0]} pts", flush=True)

    models = []
    vals_list = []
    t0 = time.time()

    for i in range(n_ens):
        ckpt = os.path.join(RESULTS, f'model_{i}.pt')
        if os.path.exists(ckpt):
            print(f"  === MODEL {i}/{n_ens} EXISTS — loading ===", flush=True)
            m = arch_fn().to(dev)
            m.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
            m.eval()
            with torch.no_grad():
                vp = m(Xv)
                vl = (sw * ((vp - Yv) / ys).pow(2)).mean().item()
            models.append(m)
            vals_list.append(vl)
            continue

        print(f"  === STARTING MODEL {i}/{n_ens} ===", flush=True)
        torch.manual_seed(42 + i * 137)
        np.random.seed(42 + i * 137)
        m = arch_fn().to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep, eta_min=1e-6)
        n = Xt.shape[0]
        best_v, best_s = float('inf'), None

        for ep in range(n_ep):
            m.train()
            pm = torch.randperm(n, device=dev)
            el, nb = 0, 0
            for s in range(0, n, 4096):
                idx = pm[s:s + 4096]
                xb, yb = Xt[idx], Yt[idx]
                pred = m(xb)
                Ld = (sw * ((pred - yb) / ys).pow(2)).mean()
                loss = Ld
                if nb % 4 == 0:
                    loss = loss + 5e-4 * reg_smoothness(m, xb, dev)
                    loss = loss + 1e-3 * reg_bounded_density(pred, ys)
                if nb % 8 == 0:
                    loss = loss + 2e-4 * reg_wafer_smoothness(m, train_d, dev, enhanced=enhanced)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                el += Ld.item()
                nb += 1
            sch.step()
            m.eval()
            with torch.no_grad():
                vp = m(Xv)
                vl = (sw * ((vp - Yv) / ys).pow(2)).mean().item()
            if vl < best_v:
                best_v = vl
                best_s = {k: v.cpu().clone() for k, v in m.state_dict().items()}
            if ep % 400 == 0:
                print(f"    M{i} [{ep:5d}] t={el / nb:.5f} v={vl:.5f}", flush=True)

        if best_s:
            m.load_state_dict(best_s)
        torch.save({k: v.cpu() for k, v in m.state_dict().items()}, ckpt)
        models.append(m)
        vals_list.append(best_v)
        print(f"  === FINISHED MODEL {i}/{n_ens} | best val = {best_v:.6f} ===", flush=True)

    tt = time.time() - t0
    print(f"  === ALL MODELS COMPLETE ({tt:.0f}s) ===", flush=True)

    # Evaluate ensemble
    if dev.type in ('mps', 'cuda'):
        Xv_eval = Xv.cpu()
        Yv_np = Yv.cpu().numpy()
        models_eval = [m.cpu() for m in models]
    else:
        Xv_eval = Xv
        Yv_np = Yv.numpy()
        models_eval = models

    preds = []
    for m in models_eval:
        m.eval()
        with torch.no_grad():
            preds.append(m(Xv_eval).numpy())
    preds_arr = np.stack(preds)
    pm_mean = preds_arr.mean(0)
    pm_std = preds_arr.std(0)

    metrics = {}
    for c, nm in enumerate(['nF', 'nSF6']):
        err = pm_mean[:, c] - Yv_np[:, c]
        metrics[nm] = {
            'rmse': float(np.sqrt((err ** 2).mean())),
            'mae': float(np.abs(err).mean()),
            'max_err': float(np.abs(err).max()),
            'ens_spread': float(pm_std[:, c].mean()),
        }

    summary = {
        'label': f'surrogate_{_ML_MODE}_v4_arch',
        'winner_experiment': winner_name,
        'architecture': arch_label,
        'training_recipe': 'v4 (physics reg + bias init + 2000 epochs)',
        'device': str(dev),
        'n_cases': len(meta),
        'n_ensemble': n_ens,
        'n_epochs': n_ep,
        'train_time_s': tt,
        'ensemble_vals': [float(v) for v in vals_list],
        'ens_mean': float(np.mean(vals_list)),
        'ens_std': float(np.std(vals_list)),
        'metrics': metrics,
        'rate_source': _ML_MODE,
        'selection': {
            'method': 'Lowest nF RMSE mean across 3 seeds in architecture sweep',
            'sweep_nF_rmse': winner['nF_rmse_mean'],
            'sweep_nF_std': winner['nF_rmse_std'],
            'all_candidates': [{'name': e['name'], 'nF_rmse': e['nF_rmse_mean']} for e in experiments],
        },
        'comparison': {
            'vs_surrogate_v4': {
                'v4_nF_rmse': 0.0029,
                'v4_nSF6_rmse': 0.0027,
                'this_nF_rmse': metrics['nF']['rmse'],
                'this_nSF6_rmse': metrics['nSF6']['rmse'],
                'nF_ratio': metrics['nF']['rmse'] / 0.0029,
                'nSF6_ratio': metrics['nSF6']['rmse'] / 0.0027,
            },
            'vs_surrogate_lxcat_v3': {
                'v3_nF_rmse': 0.0112,
                'v3_nSF6_rmse': 0.0081,
                'this_nF_rmse': metrics['nF']['rmse'],
                'this_nSF6_rmse': metrics['nSF6']['rmse'],
                'nF_improvement': (0.0112 - metrics['nF']['rmse']) / 0.0112,
                'nSF6_improvement': (0.0081 - metrics['nSF6']['rmse']) / 0.0081,
            },
        },
    }

    with open(os.path.join(RESULTS, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Config file
    config = {
        'winner_experiment': winner_name,
        'architecture': arch_label,
        'enhanced_features': enhanced,
        'optimizer': 'Adam',
        'lr': 1e-3,
        'scheduler': 'CosineAnnealing',
        'eta_min': 1e-6,
        'epochs': 2000,
        'batch_size': 4096,
        'loss_weights': {'nF': 1.0, 'nSF6': 1.5},
        'physics_reg': {
            'smoothness': 5e-4,
            'bounded_density': 1e-3,
            'wafer_smoothness': 2e-4,
        },
    }
    with open(os.path.join(RESULTS, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: surrogate_lxcat_v4_arch")
    print(f"  Winner: {winner_name} ({arch_label})")
    print(f"  nF RMSE:  {metrics['nF']['rmse']:.5f} (earlier baseline 0.0112, full recipe 0.0029)")
    print(f"  nSF6 RMSE: {metrics['nSF6']['rmse']:.5f} (earlier baseline 0.0081, full recipe 0.0027)")
    print(f"  nF improvement vs v3: {summary['comparison']['vs_surrogate_lxcat_v3']['nF_improvement']:.1%}")
    print(f"  nF gap to v4: {summary['comparison']['vs_surrogate_v4']['nF_ratio']:.2f}x")
    print(f"{'=' * 60}", flush=True)


if __name__ == '__main__':
    main()
