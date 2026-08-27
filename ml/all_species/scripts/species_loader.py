"""Single-species dataset loader for 6c.

Reuses 6b's mesh + manifest plumbing, then stacks one log10(species) target
column (rather than 6b's 2-column lnF + lnSF6).  Returns the same 4-tuple
shape as 6b's loader so train_species.py can swap in for train_ensemble.py
with minimal changes.
"""
from __future__ import annotations

import os
import sys
from typing import Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# V7: use THIS project's loader (6d) so the dataset resolves to the V7
# legacy set (gamma*=0.155 + f_e closure), not 6b's old-physics tree.
SIXB_ML = os.path.abspath(os.path.join(HERE, "..", "ml"))
if SIXB_ML not in sys.path:
    sys.path.insert(0, SIXB_ML)
import ml_dataset_loader as mdl  # noqa: E402


def load_species_dataset(species: str, mode: str = None,
                         val_frac: float = 0.15
                         ) -> Tuple[dict, dict, list, set]:
    """Load a per-cell single-species dataset.

    Mirrors mdl.load_dataset() but stacks one target column
    'ln<species>' = log10(field, clipped to >= 1) instead of lnF + lnSF6.
    Same seed=42 case-level train/val split as 6b.
    """
    if mode is None:
        mode = os.environ.get("ML_DATASET_MODE", "legacy")
    manifest = mdl._manifest_for_mode(mode)
    runs = manifest["runs"]
    for i, r in enumerate(runs):
        r["idx"] = i

    n = len(runs)
    np.random.seed(42)
    perm = np.random.permutation(n)
    n_val = max(int(n * val_frac), 10)
    val_idx = set(perm[:n_val].tolist())

    rc, zc, inside = mdl._load_mesh()
    Nr, Nz = len(rc), len(zc)

    cols = ["r", "z", "P", "p", "Ar", "ln" + species, "case"]
    data: dict = {k: [] for k in cols}
    meta: list = []

    for run in runs:
        case_id = run["case_id"]
        case_dir = os.path.join(mdl.DATASET_BASE, mode, case_id)
        f_path = os.path.join(case_dir, f"{species}.npy")
        if not os.path.exists(f_path):
            continue
        field = np.load(f_path)
        if field.shape != (Nr, Nz):
            raise ValueError(
                f"{case_id}: {species} shape {field.shape} != mesh "
                f"({Nr}, {Nz})"
            )

        P_val = float(run["P_rf_W"])
        p_val = float(run["p_mTorr"])
        Ar_val = float(run["x_Ar"])
        meta.append({
            "idx": run["idx"], "case_id": case_id,
            "P_rf": P_val, "p_mTorr": p_val, "frac_Ar": Ar_val,
            "file": case_id,
        })

        # Inside-mask AND positive (some species can be ~0 in places)
        mask = inside & (field > 0)
        ii, jj = np.where(mask)
        for i, j in zip(ii, jj):
            data["r"].append(rc[i])
            data["z"].append(zc[j])
            data["P"].append(P_val)
            data["p"].append(p_val)
            data["Ar"].append(Ar_val)
            data["ln" + species].append(np.log10(max(field[i, j], 1.0)))
            data["case"].append(run["idx"])

    arrays = {k: np.array(v, dtype=np.float32) for k, v in data.items()}
    arrays["case"] = arrays["case"].astype(np.int32)
    mask = np.array([c not in val_idx for c in arrays["case"]])
    split = lambda m: {k: v[m] for k, v in arrays.items()}
    return split(mask), split(~mask), meta, val_idx


def to_tensors_single(data: dict, species: str, device,
                      R_PROC: float = 0.105, Z_TOP: float = 0.234):
    """Pack a single-species dataset as (X, Y) tensors. Y has shape (N, 1)."""
    import torch
    X = np.column_stack([
        data["r"] / R_PROC,
        data["z"] / Z_TOP,
        data["P"] / 1200,
        data["p"] / 20,
        data["Ar"],
    ]).astype(np.float32)
    Y = data["ln" + species].astype(np.float32).reshape(-1, 1)
    return (torch.tensor(X, device=device),
            torch.tensor(Y, device=device))
