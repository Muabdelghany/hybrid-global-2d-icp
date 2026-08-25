"""
Ablation study: isolate the effect of each training recipe component.

Components from the v4 recipe:
  A) Output bias initialization [19.8, 20.0]
  B) Physics regularization (smoothness + bounded + wafer)
  C) 2000 epochs (vs 1500)

Tests:
  1. None (E0 baseline reproduction)     — no bias, no reg, 1500ep
  2. Bias only                            — bias, no reg, 1500ep
  3. Physics reg only                     — no bias, reg, 1500ep
  4. 2000 epochs only                     — no bias, no reg, 2000ep
  5. All three (E1 reproduction)          — bias, reg, 2000ep

Each test: 3 seeds for stability.
Uses CPU to avoid competing with the GPU architecture sweep.
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

# Uses the local ml_dataset_loader
from ml_dataset_loader import load_dataset as _load_dataset_impl
_ML_MODE = os.environ.get('ML_DATASET_MODE', 'legacy')
OUT = os.path.join(REPO, 'results', f'ml_ablation_{_ML_MODE}')
os.makedirs(OUT, exist_ok=True)


class SurrogateNet(nn.Module):
    def __init__(self, use_bias_init=False, nf=64, nh=128, nl=4, fs=3.0, drop=0.05):
        super().__init__()
        B = torch.randn(5, nf) * fs
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
        if use_bias_init:
            with torch.no_grad():
                self.head.bias.copy_(torch.tensor([19.8, 20.0]))

    def forward(self, x):
        p = x @ self.B
        e = torch.cat([torch.sin(p), torch.cos(p)], -1)
        return self.head(self.bb(e) + self.proj(e))


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
    x = torch.stack([r, z, P, p, Ar], dim=-1)
    pred = model(x)
    d2 = pred[2:] - 2 * pred[1:-1] + pred[:-2]
    return d2.pow(2).mean()


def load_dataset(val_frac=0.15):
    train, val, _meta, _val_idx = _load_dataset_impl(
        mode=_ML_MODE, val_frac=val_frac, enhanced_features=False)
    return train, val


def to_tensors(data, device):
    X = np.column_stack([data['r'] / R_PROC, data['z'] / Z_TOP,
                         data['P'] / 1200, data['p'] / 20, data['Ar']])
    Y = np.column_stack([data['lnF'], data['lnSF6']])
    return (torch.tensor(X, dtype=torch.float32, device=device),
            torch.tensor(Y, dtype=torch.float32, device=device))


def train_and_eval(use_bias, use_reg, n_ep, seed, Xt, Yt, Xv, Yv, train_data, dev):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SurrogateNet(use_bias_init=use_bias).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep, eta_min=1e-6)
    sw = torch.tensor([1.0, 1.5], device=dev)
    ys = Yt.std(0, keepdim=True).clamp(min=1e-3)
    n = Xt.shape[0]
    best_v, best_s = float('inf'), None

    for ep in range(n_ep):
        model.train()
        pm = torch.randperm(n, device=dev)
        el, nb = 0, 0
        for s in range(0, n, 4096):
            idx = pm[s:s + 4096]
            xb, yb = Xt[idx], Yt[idx]
            pred = model(xb)
            Ld = (sw * ((pred - yb) / ys).pow(2)).mean()
            loss = Ld
            if use_reg:
                if nb % 4 == 0:
                    loss = loss + 5e-4 * reg_smoothness(model, xb, dev)
                    loss = loss + 1e-3 * reg_bounded_density(pred, ys)
                if nb % 8 == 0:
                    loss = loss + 2e-4 * reg_wafer_smoothness(model, train_data, dev)
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

    if best_s:
        model.load_state_dict(best_s)
    model.eval()
    with torch.no_grad():
        pred = model(Xv).cpu().numpy()
    Yv_np = Yv.cpu().numpy()
    metrics = {}
    for c, nm in enumerate(['nF', 'nSF6']):
        err = pred[:, c] - Yv_np[:, c]
        metrics[nm] = {'rmse': float(np.sqrt((err ** 2).mean())),
                       'mae': float(np.abs(err).mean())}
    return metrics


def main():
    # Force CPU to avoid competing with GPU architecture sweep
    dev = torch.device('cpu')
    print(f"Ablation study — Device: CPU (to avoid MPS contention)", flush=True)

    print("Loading dataset...", flush=True)
    train_d, val_d = load_dataset()
    Xt, Yt = to_tensors(train_d, dev)
    Xv, Yv = to_tensors(val_d, dev)
    print(f"  Train: {Xt.shape[0]}, Val: {Xv.shape[0]}", flush=True)

    experiments = [
        {'name': 'none',       'bias': False, 'reg': False, 'epochs': 1500},
        {'name': 'bias_only',  'bias': True,  'reg': False, 'epochs': 1500},
        {'name': 'reg_only',   'bias': False, 'reg': True,  'epochs': 1500},
        {'name': 'epochs_only','bias': False, 'reg': False, 'epochs': 2000},
        {'name': 'all_three',  'bias': True,  'reg': True,  'epochs': 2000},
    ]

    seeds = [42, 179, 316]
    results = []

    for exp in experiments:
        print(f"\n  === {exp['name']} (bias={exp['bias']}, reg={exp['reg']}, ep={exp['epochs']}) ===", flush=True)
        runs = []
        t0 = time.time()
        for i, seed in enumerate(seeds):
            m = train_and_eval(exp['bias'], exp['reg'], exp['epochs'], seed,
                               Xt, Yt, Xv, Yv, train_d, dev)
            runs.append(m)
            print(f"    seed {seed}: nF={m['nF']['rmse']:.5f} nSF6={m['nSF6']['rmse']:.5f}", flush=True)
        dt = time.time() - t0

        nF_vals = [r['nF']['rmse'] for r in runs]
        nSF6_vals = [r['nSF6']['rmse'] for r in runs]
        summary = {
            'name': exp['name'],
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
        results.append(summary)
        print(f"  => nF={summary['nF_rmse_mean']:.5f}+/-{summary['nF_rmse_std']:.5f}  "
              f"nSF6={summary['nSF6_rmse_mean']:.5f}+/-{summary['nSF6_rmse_std']:.5f}  ({dt:.0f}s)", flush=True)

    with open(os.path.join(OUT, 'ablation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Markdown
    md = "# Ablation Study: Training Recipe Components\n\n"
    md += "Dataset: pinn_dataset_lxcat_v3 (221 LXCat cases)\n"
    md += "Architecture: Fourier + 4x128 GELU MLP (same as v4)\n"
    md += "3 seeds per experiment\n\n"
    md += "| Experiment | Bias Init | Phys Reg | Epochs | nF RMSE | nSF6 RMSE | Time |\n"
    md += "|---|---|---|---|---|---|---|\n"
    for r in results:
        md += f"| {r['name']} | {'Y' if r['bias_init'] else 'N'} | "
        md += f"{'Y' if r['physics_reg'] else 'N'} | {r['epochs']} | "
        md += f"{r['nF_rmse_mean']:.5f}+/-{r['nF_rmse_std']:.5f} | "
        md += f"{r['nSF6_rmse_mean']:.5f}+/-{r['nSF6_rmse_std']:.5f} | "
        md += f"{r['time_s']:.0f}s |\n"

    # Compute relative improvements
    baseline_nF = results[0]['nF_rmse_mean']
    md += "\n## Relative improvement vs baseline (no recipe)\n\n"
    for r in results[1:]:
        improv = (baseline_nF - r['nF_rmse_mean']) / baseline_nF * 100
        md += f"- **{r['name']}**: {improv:+.1f}% nF improvement\n"

    with open(os.path.join(OUT, 'ablation_results.md'), 'w') as f:
        f.write(md)

    print(f"\nResults written to {OUT}/", flush=True)


if __name__ == '__main__':
    main()
