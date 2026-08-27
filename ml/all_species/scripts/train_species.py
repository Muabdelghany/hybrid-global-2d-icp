#!/usr/bin/env python3
"""Per-species 5-model ensemble trainer for 6c.

Mirrors 6b's train_ensemble.py training loop (Adam, cosine-anneal,
2000 epochs, smoothness + bounded-density + wafer-smoothness regularizers,
weighted-σ-normalised MSE) but with a SingleHeadMLP and a one-column
target. Trains 5 seeds, saves model_{0..4}.pt + summary.json + config.json
under <out>/<species>/.

Resume-aware: if all 5 model_*.pt + summary.json already exist, exits.

Usage:
    python train_species.py --species nF --out results/ml_production_ensemble_all_species
"""
from __future__ import annotations

import argparse
import json
import os
_ML_MODE = os.environ.get("ML_DATASET_MODE", "legacy")  # dataset mode: was hardcoded lxcat (caught 2026-07-20)
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from species_loader import load_species_dataset, to_tensors_single  # noqa: E402
from single_head_arch import SingleHeadMLP  # noqa: E402


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "CUDA"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "MPS"
    return torch.device("cpu"), "CPU"


def reg_smoothness(model, xb, device):
    xg = xb.detach().clone().requires_grad_(True)
    pred = model(xg)
    L = torch.tensor(0.0, device=device)
    g = torch.autograd.grad(pred.sum(), xg, create_graph=True)[0]
    L = (g[:, 0] ** 2 + g[:, 1] ** 2).mean()
    return L


def reg_bounded_density(pred, y_std):
    # Penalise predictions outside roughly [-30, 50] in log-density.
    return (torch.relu(pred - 30) ** 2 + torch.relu(-pred - 30) ** 2).mean()


def reg_wafer_smoothness(model, train_data, device):
    r_arr = np.array(train_data["r"]); z_arr = np.array(train_data["z"])
    P_arr = np.array(train_data["P"]); p_arr = np.array(train_data["p"])
    Ar_arr = np.array(train_data["Ar"])
    z_min = float(z_arr.min())
    near = z_arr < (z_min + 0.005)
    if near.sum() < 32:
        return torch.tensor(0.0, device=device)
    idx = np.random.choice(np.where(near)[0],
                           size=min(512, int(near.sum())), replace=False)
    R_PROC = 0.105
    Z_TOP = 0.234
    X = np.column_stack([
        r_arr[idx] / R_PROC, z_arr[idx] / Z_TOP,
        P_arr[idx] / 1200, p_arr[idx] / 20, Ar_arr[idx],
    ]).astype(np.float32)
    xg = torch.tensor(X, device=device, requires_grad=True)
    pred = model(xg)
    g = torch.autograd.grad(pred.sum(), xg, create_graph=True)[0]
    return (g[:, 0] ** 2).mean()


def train_one_seed(seed, species, output_bias, Xt, Yt, Xv, Yv, ys,
                   train_data, dev, n_ep=2000, batch=4096):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = SingleHeadMLP(n_in=5, output_bias=output_bias).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_ep, eta_min=1e-6)
    n = Xt.shape[0]
    best_v, best_s = float("inf"), None

    for ep in range(n_ep):
        m.train()
        pm = torch.randperm(n, device=dev)
        el, nb = 0.0, 0
        for s in range(0, n, batch):
            idx = pm[s:s + batch]
            xb, yb = Xt[idx], Yt[idx]
            pred = m(xb)
            Ld = (((pred - yb) / ys).pow(2)).mean()
            loss = Ld
            if nb % 4 == 0:
                loss = loss + 5e-4 * reg_smoothness(m, xb, dev)
                loss = loss + 1e-3 * reg_bounded_density(pred, ys)
            if nb % 8 == 0:
                loss = loss + 2e-4 * reg_wafer_smoothness(m, train_data, dev)
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
            vl = (((vp - Yv) / ys).pow(2)).mean().item()
        if vl < best_v:
            best_v = vl
            best_s = {k: v.cpu().clone()
                      for k, v in m.state_dict().items()}
        if ep % 400 == 0:
            print(f"    [{species} seed={seed}] ep={ep:5d} "
                  f"t={el / max(nb,1):.5f} v={vl:.5f}", flush=True)

    if best_s:
        m.load_state_dict(best_s)
    return m, best_v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True,
                    help="e.g. nF, nSF6, ion_ne, ion_n+, Te")
    ap.add_argument("--out", required=True,
                    help="parent dir; per-species subdir is created")
    ap.add_argument("--n-ensemble", type=int, default=5)
    ap.add_argument("--n-epochs", type=int, default=2000)
    args = ap.parse_args()

    species = args.species
    out_dir = os.path.join(args.out, species)
    os.makedirs(out_dir, exist_ok=True)

    summary_path = os.path.join(out_dir, "summary.json")
    if os.path.exists(summary_path):
        print(f"==> {summary_path} exists; nothing to do.", flush=True)
        return

    print(f"=" * 60, flush=True)
    print(f"  Per-species ensemble training", flush=True)
    print(f"  Species:    {species}", flush=True)
    print(f"  Output dir: {out_dir}", flush=True)
    print(f"  Ensemble:   {args.n_ensemble} seeds × {args.n_epochs} epochs",
          flush=True)

    print("==> Loading dataset…", flush=True)
    train_d, val_d, meta, vi = load_species_dataset(
        species, mode=_ML_MODE, val_frac=0.15)
    print(f"  train cells: {len(train_d['r'])}; "
          f"val cells: {len(val_d['r'])}; cases: {len(meta)}", flush=True)

    output_bias = float(np.mean(train_d["ln" + species]))
    print(f"  bias-init (log-mean): {output_bias:.4f}", flush=True)

    dev, dev_name = select_device()
    print(f"  device: {dev_name}", flush=True)

    Xt, Yt = to_tensors_single(train_d, species, dev)
    Xv, Yv = to_tensors_single(val_d, species, dev)
    ys = Yt.std(0, keepdim=True).clamp(min=1e-3)

    seeds = [42, 179, 316, 453, 590][:args.n_ensemble]
    models = []
    val_losses = []
    t0 = time.time()
    for i, seed in enumerate(seeds):
        ckpt = os.path.join(out_dir, f"model_{i}.pt")
        if os.path.exists(ckpt):
            print(f"  === MODEL {i}/{len(seeds)} EXISTS — loading ===",
                  flush=True)
            m = SingleHeadMLP(n_in=5, output_bias=output_bias).to(dev)
            m.load_state_dict(torch.load(ckpt, map_location=dev,
                                         weights_only=True))
            m.eval()
            with torch.no_grad():
                vl = (((m(Xv) - Yv) / ys).pow(2)).mean().item()
        else:
            print(f"  === STARTING MODEL {i}/{len(seeds)} (seed={seed}) ===",
                  flush=True)
            m, vl = train_one_seed(
                seed=seed, species=species, output_bias=output_bias,
                Xt=Xt, Yt=Yt, Xv=Xv, Yv=Yv, ys=ys,
                train_data=train_d, dev=dev, n_ep=args.n_epochs)
            torch.save({k: v.cpu() for k, v in m.state_dict().items()},
                       ckpt)
            print(f"  === FINISHED MODEL {i}/{len(seeds)} | "
                  f"best val = {vl:.6f} ===", flush=True)
        models.append(m)
        val_losses.append(vl)

    tt = time.time() - t0

    # Ensemble eval (move to CPU if accelerator-bound)
    if dev.type in ("mps", "cuda"):
        Xv_eval = Xv.cpu()
        Yv_np = Yv.cpu().numpy()
        models_eval = [mm.cpu() for mm in models]
    else:
        Xv_eval = Xv
        Yv_np = Yv.numpy()
        models_eval = models

    with torch.no_grad():
        preds = torch.stack([mm(Xv_eval) for mm in models_eval]).numpy()
    pm_mean = preds.mean(0)
    pm_std = preds.std(0)

    err = pm_mean[:, 0] - Yv_np[:, 0]
    metrics = {
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(np.abs(err).mean()),
        "max_err": float(np.abs(err).max()),
        "ens_spread": float(pm_std[:, 0].mean()),
    }
    summary = {
        "species": species,
        "label": f"surrogate_{_ML_MODE}_v4_arch_{species}",
        "winner_experiment": "E3_separate_heads (single-head)",
        "architecture": "SingleHeadMLP (3×128 trunk + skip + 2×64 head)",
        "training_recipe": "v4 per-species (physics reg + bias init + 2000 epochs)",
        "device": str(dev),
        "n_cases": len(meta),
        "n_ensemble": len(models),
        "n_epochs": args.n_epochs,
        "train_time_s": tt,
        "ensemble_vals": [float(v) for v in val_losses],
        "ens_mean": float(np.mean(val_losses)),
        "ens_std": float(np.std(val_losses)),
        "metrics": metrics,
        "rate_source": _ML_MODE,
        "output_bias_init": output_bias,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"==> Wrote {summary_path}: RMSE={metrics['rmse']:.5f}", flush=True)

    config = {
        "species": species,
        "n_in": 5, "nf": 64, "fs": 3.0, "drop": 0.05,
        "trunk_layers": 3, "trunk_width": 128,
        "head_layers": 2, "head_width": 64,
        "output_bias_init": output_bias,
        "seeds": seeds,
        "n_epochs": args.n_epochs,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
