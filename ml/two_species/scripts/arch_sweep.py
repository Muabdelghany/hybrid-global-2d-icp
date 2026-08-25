"""
LXCat Architecture Upgrade: Systematic experiments to close the RMSE gap.

Key diagnosis finding: The LXCat and legacy datasets have near-identical
statistical properties (range, std, gradients, correlations, regime structure).
The RMSE gap is likely from training recipe differences, not data difficulty:
  - surrogate_v4 used physics regularization (smoothness, bounded density, wafer)
  - surrogate_v4 used 2000 epochs (vs 1500)
  - surrogate_v4 used output bias initialization [19.8, 20.0]

Experiments:
  E0: Baseline reproduction (confirm surrogate_lxcat_v3 result)
  E1: full recipe (physics regularization + 2000 epochs + bias init)
  E2: Wider/deeper (6x256 + physics reg)
  E3: Separate heads (shared trunk + per-species heads + physics reg)
  E4: Residual MLP (residual blocks + physics reg)
  E5: Enhanced features (interaction terms + physics reg)
  E6: Huber loss + physics reg
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

# Uses the local ml_dataset_loader rather than an external dataset module
# layout described in the paper. The mode (legacy|lxcat) is selected via env var.
from ml_dataset_loader import load_dataset as _load_dataset_impl
_ML_MODE = os.environ.get('ML_DATASET_MODE', 'legacy')
OUT = os.path.join(REPO, 'results', f'ml_arch_sweep_{_ML_MODE}')


def select_device():
    if torch.cuda.is_available(): return torch.device('cuda'), "CUDA"
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return torch.device('mps'), "MPS"
    return torch.device('cpu'), "CPU"


# ===============================================================
# Dataset loading
# ===============================================================

def load_dataset(val_frac=0.15, enhanced_features=False):
    """Wraps ml_dataset_loader.load_dataset with the configured mode."""
    return _load_dataset_impl(mode=_ML_MODE, val_frac=val_frac,
                            enhanced_features=enhanced_features)


def to_tensors(data, device, enhanced=False):
    cols = [data['r'] / R_PROC, data['z'] / Z_TOP,
            data['P'] / 1200, data['p'] / 20, data['Ar']]
    if enhanced:
        cols += [
            data['Pp'] / (1200 * 20),  # P*p normalized
            data['PAr'] / 1200,         # P*Ar normalized
            data['pAr'] / 20,           # p*Ar normalized
            data['logp'] / np.log(20),  # log(p) normalized
            data['inv_p'] * 3,          # 1/p scaled (max ~0.33 at p=3)
        ]
    X = np.column_stack(cols)
    Y = np.column_stack([data['lnF'], data['lnSF6']])
    return (torch.tensor(X, dtype=torch.float32, device=device),
            torch.tensor(Y, dtype=torch.float32, device=device))


# ===============================================================
# Model architectures
# ===============================================================

class BaselineMLP(nn.Module):
    """Same as surrogate_lxcat_v3 (Fourier + 4x128 GELU)."""
    def __init__(self, n_in=5, n_out=2, nh=128, nl=4, nf=64, fs=3.0, drop=0.05):
        super().__init__()
        B = torch.randn(n_in, nf) * fs
        self.register_buffer('B', B)
        ni = 2 * nf
        layers = [nn.Linear(ni, nh), nn.GELU(), nn.Dropout(drop)]
        for _ in range(nl - 1):
            layers += [nn.Linear(nh, nh), nn.GELU(), nn.Dropout(drop)]
        self.bb = nn.Sequential(*layers)
        self.proj = nn.Linear(ni, nh)
        self.head = nn.Linear(nh, n_out)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        p = x @ self.B
        e = torch.cat([torch.sin(p), torch.cos(p)], -1)
        return self.head(self.bb(e) + self.proj(e))


class BaselineWithBias(BaselineMLP):
    """Same as baseline but with output bias initialization (as in the full recipe)."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([19.8, 20.0]))


class WiderDeeperMLP(nn.Module):
    """6x256 with Fourier features and bias init."""
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
    """Shared trunk + separate heads for nF and nSF6."""
    def __init__(self, n_in=5, nf=64, fs=3.0, drop=0.05):
        super().__init__()
        B = torch.randn(n_in, nf) * fs
        self.register_buffer('B', B)
        ni = 2 * nf
        nh = 128
        # Shared trunk: 3 layers
        trunk = [nn.Linear(ni, nh), nn.GELU(), nn.Dropout(drop)]
        for _ in range(2):
            trunk += [nn.Linear(nh, nh), nn.GELU(), nn.Dropout(drop)]
        self.trunk = nn.Sequential(*trunk)
        self.proj = nn.Linear(ni, nh)
        # Separate heads: 2 layers each
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
    """Fourier + residual blocks for stable deep training."""
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


# ===============================================================
# Physics regularization (from train_v4.py)
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


def reg_wafer_smoothness(model, data, device):
    r = torch.linspace(0.001, 0.95, 30, device=device)
    z = torch.full((30,), 0.01, device=device)
    idx = np.random.randint(0, len(data['P']), 1)[0]
    P = torch.full((30,), data['P'][idx] / 1200, device=device)
    p = torch.full((30,), data['p'][idx] / 20, device=device)
    Ar = torch.full((30,), data['Ar'][idx], device=device)
    cols = [r, z, P, p, Ar]
    if 'Pp' in data:
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
# Training loop
# ===============================================================

def train_model(model, Xt, Yt, Xv, Yv, dev, train_data,
                n_ep=2000, lr=1e-3, bs=4096,
                use_physics_reg=False, use_huber=False,
                ws=5e-4, wb=1e-3, ww=2e-4,
                wF=1.0, wS=1.5, label=""):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep, eta_min=1e-6)
    sw = torch.tensor([wF, wS], device=dev)
    ys = Yt.std(0, keepdim=True).clamp(min=1e-3)
    n = Xt.shape[0]
    best_v, best_s = float('inf'), None
    huber = nn.HuberLoss(reduction='none', delta=0.1) if use_huber else None

    for ep in range(n_ep):
        model.train()
        pm = torch.randperm(n, device=dev)
        el, nb = 0, 0
        for s in range(0, n, bs):
            idx = pm[s:s + bs]
            xb, yb = Xt[idx], Yt[idx]
            pred = model(xb)

            if use_huber:
                Ld = (sw * huber((pred - yb) / ys, torch.zeros_like(pred))).mean()
            else:
                Ld = (sw * ((pred - yb) / ys).pow(2)).mean()

            loss = Ld
            if use_physics_reg:
                if nb % 4 == 0:
                    loss = loss + ws * reg_smoothness(model, xb, dev)
                    loss = loss + wb * reg_bounded_density(pred, ys)
                if nb % 8 == 0:
                    loss = loss + ww * reg_wafer_smoothness(model, train_data, dev)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            el += Ld.item()
            nb += 1
        sch.step()

        model.eval()
        with torch.no_grad():
            vp = model(Xv)
            vl = (sw * ((vp - Yv) / ys).pow(2)).mean().item()
        if vl < best_v:
            best_v = vl
            best_s = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % max(n_ep // 5, 1) == 0:
            print(f"    {label} [{ep:5d}] t={el / nb:.5f} v={vl:.5f}", flush=True)

    if best_s:
        model.load_state_dict(best_s)
    return best_v


def evaluate(model, Xv, Yv, dev):
    model.eval()
    with torch.no_grad():
        pred = model(Xv).cpu().numpy()
    Yv_np = Yv.cpu().numpy()
    metrics = {}
    for c, nm in enumerate(['nF', 'nSF6']):
        err = pred[:, c] - Yv_np[:, c]
        metrics[nm] = {
            'rmse': float(np.sqrt((err ** 2).mean())),
            'mae': float(np.abs(err).mean()),
            'max_err': float(np.abs(err).max()),
        }
    return metrics


# ===============================================================
# Experiments
# ===============================================================

def run_experiment(name, model_fn, dev, train_data, val_data,
                   n_ep=2000, use_physics_reg=False, use_huber=False,
                   enhanced=False, n_runs=3):
    """Run an experiment with multiple seeds for stability."""
    print(f"\n  === EXPERIMENT: {name} ===", flush=True)

    Xt, Yt = to_tensors(train_data, dev, enhanced=enhanced)
    Xv, Yv = to_tensors(val_data, dev, enhanced=enhanced)
    print(f"    Train: {Xt.shape}, Val: {Xv.shape}", flush=True)

    results = []
    t0 = time.time()

    for i in range(n_runs):
        torch.manual_seed(42 + i * 137)
        np.random.seed(42 + i * 137)
        model = model_fn().to(dev)
        n_params = sum(p.numel() for p in model.parameters())

        best_v = train_model(
            model, Xt, Yt, Xv, Yv, dev, train_data,
            n_ep=n_ep, use_physics_reg=use_physics_reg,
            use_huber=use_huber, label=f"{name}[{i}]")

        metrics = evaluate(model, Xv, Yv, dev)
        results.append({
            'seed': 42 + i * 137,
            'best_val': best_v,
            'metrics': metrics,
            'n_params': n_params,
        })
        print(f"    Run {i}: nF={metrics['nF']['rmse']:.5f} nSF6={metrics['nSF6']['rmse']:.5f}", flush=True)

    tt = time.time() - t0

    # Aggregate
    nF_rmses = [r['metrics']['nF']['rmse'] for r in results]
    nSF6_rmses = [r['metrics']['nSF6']['rmse'] for r in results]

    summary = {
        'name': name,
        'n_runs': n_runs,
        'n_epochs': n_ep,
        'n_params': results[0]['n_params'],
        'use_physics_reg': use_physics_reg,
        'use_huber': use_huber,
        'enhanced_features': enhanced,
        'train_time_s': tt,
        'nF_rmse_mean': float(np.mean(nF_rmses)),
        'nF_rmse_std': float(np.std(nF_rmses)),
        'nF_rmse_best': float(np.min(nF_rmses)),
        'nSF6_rmse_mean': float(np.mean(nSF6_rmses)),
        'nSF6_rmse_std': float(np.std(nSF6_rmses)),
        'nSF6_rmse_best': float(np.min(nSF6_rmses)),
        'nF_mae_mean': float(np.mean([r['metrics']['nF']['mae'] for r in results])),
        'nSF6_mae_mean': float(np.mean([r['metrics']['nSF6']['mae'] for r in results])),
        'nF_maxerr_mean': float(np.mean([r['metrics']['nF']['max_err'] for r in results])),
        'nSF6_maxerr_mean': float(np.mean([r['metrics']['nSF6']['max_err'] for r in results])),
        'runs': results,
    }

    print(f"  === {name}: nF={summary['nF_rmse_mean']:.5f}+/-{summary['nF_rmse_std']:.5f}  "
          f"nSF6={summary['nSF6_rmse_mean']:.5f}+/-{summary['nSF6_rmse_std']:.5f}  "
          f"({tt:.0f}s) ===", flush=True)

    return summary


def main():
    dev, dev_name = select_device()
    os.makedirs(OUT, exist_ok=True)

    print(f"{'=' * 70}", flush=True)
    print(f"  LXCat Architecture Upgrade — Device: {dev} ({dev_name})", flush=True)
    print(f"{'=' * 70}", flush=True)

    # Load data
    print("\nLoading standard features...", flush=True)
    train_std, val_std, meta, vi = load_dataset(enhanced_features=False)
    print("\nLoading enhanced features...", flush=True)
    train_enh, val_enh, _, _ = load_dataset(enhanced_features=True)

    experiments = []

    # E0: Baseline (reproduce surrogate_lxcat_v3)
    experiments.append(run_experiment(
        'E0_baseline',
        lambda: BaselineMLP(n_in=5),
        dev, train_std, val_std,
        n_ep=1500, use_physics_reg=False, n_runs=3))

    # E1: full recipe (physics regularization + 2000 epochs + bias init)
    experiments.append(run_experiment(
        'E1_v4_recipe',
        lambda: BaselineWithBias(n_in=5),
        dev, train_std, val_std,
        n_ep=2000, use_physics_reg=True, n_runs=3))

    # E2: Wider/deeper (6x256 + physics reg)
    experiments.append(run_experiment(
        'E2_wider_deeper',
        lambda: WiderDeeperMLP(n_in=5),
        dev, train_std, val_std,
        n_ep=2000, use_physics_reg=True, n_runs=3))

    # E3: Separate heads (shared trunk + per-species heads)
    experiments.append(run_experiment(
        'E3_separate_heads',
        lambda: SeparateHeadsMLP(n_in=5),
        dev, train_std, val_std,
        n_ep=2000, use_physics_reg=True, n_runs=3))

    # E4: Residual MLP
    experiments.append(run_experiment(
        'E4_residual',
        lambda: ResidualMLP(n_in=5),
        dev, train_std, val_std,
        n_ep=2000, use_physics_reg=True, n_runs=3))

    # E5: Enhanced features + physics reg
    experiments.append(run_experiment(
        'E5_enhanced_features',
        lambda: BaselineWithBias(n_in=10),
        dev, train_enh, val_enh,
        n_ep=2000, use_physics_reg=True, enhanced=True, n_runs=3))

    # E6: Huber loss + physics reg
    experiments.append(run_experiment(
        'E6_huber_loss',
        lambda: BaselineWithBias(n_in=5),
        dev, train_std, val_std,
        n_ep=2000, use_physics_reg=True, use_huber=True, n_runs=3))

    # -- Save experiment table --
    table = {
        'experiments': experiments,
        'reference': {
            'surrogate_lxcat_v3': {'nF_rmse': 0.0112, 'nSF6_rmse': 0.0081},
            'surrogate_v4': {'nF_rmse': 0.0029, 'nSF6_rmse': 0.0027},
        },
    }
    with open(os.path.join(OUT, 'experiment_table.json'), 'w') as f:
        json.dump(table, f, indent=2)

    # -- Generate markdown table --
    md_rows = []
    for exp in experiments:
        gap_nF = exp['nF_rmse_mean'] / 0.0029
        gap_nSF6 = exp['nSF6_rmse_mean'] / 0.0027
        improv_nF = (0.0112 - exp['nF_rmse_mean']) / 0.0112 * 100
        md_rows.append(
            f"| {exp['name']} | {exp['n_params']:,} | {exp['n_epochs']} | "
            f"{'Y' if exp['use_physics_reg'] else 'N'} | "
            f"{exp['nF_rmse_mean']:.5f}+/-{exp['nF_rmse_std']:.5f} | "
            f"{exp['nSF6_rmse_mean']:.5f}+/-{exp['nSF6_rmse_std']:.5f} | "
            f"{gap_nF:.1f}x | {improv_nF:+.1f}% |"
        )

    md = f"""# LXCat Architecture Upgrade — Experiment Results

## Reference baselines
- **surrogate_v4** (legacy): nF RMSE = 0.0029, nSF6 RMSE = 0.0027
- **surrogate_lxcat_v3** (LXCat baseline): nF RMSE = 0.0112, nSF6 RMSE = 0.0081

## Experiment table

| Experiment | Params | Epochs | Phys Reg | nF RMSE (mean+/-std) | nSF6 RMSE (mean+/-std) | nF gap to v4 | nF improv vs v3 |
|---|---|---|---|---|---|---|---|
""" + "\n".join(md_rows)

    # Find best
    best_idx = int(np.argmin([e['nF_rmse_mean'] for e in experiments]))
    best = experiments[best_idx]

    md += f"""

## Best experiment: {best['name']}
- nF RMSE: {best['nF_rmse_mean']:.5f} (vs the earlier baseline 0.0112)
- nSF6 RMSE: {best['nSF6_rmse_mean']:.5f} (vs the earlier baseline 0.0081)
- Improvement over the earlier baseline: {(0.0112 - best['nF_rmse_mean'])/0.0112*100:.1f}% nF, {(0.0081 - best['nSF6_rmse_mean'])/0.0081*100:.1f}% nSF6
- Remaining gap to the full recipe: {best['nF_rmse_mean']/0.0029:.1f}x nF, {best['nSF6_rmse_mean']/0.0027:.1f}x nSF6
"""

    with open(os.path.join(OUT, 'experiment_table.md'), 'w') as f:
        f.write(md)

    print(f"\n{'=' * 70}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"  Best: {best['name']}")
    print(f"  nF RMSE: {best['nF_rmse_mean']:.5f} (earlier baseline 0.0112, full recipe 0.0029)")
    print(f"  nSF6 RMSE: {best['nSF6_rmse_mean']:.5f} (earlier baseline 0.0081, full recipe 0.0027)")
    print(f"{'=' * 70}", flush=True)

    return table


if __name__ == '__main__':
    main()
