"""Loader for the ML dataset layout.

Drop-in replacement for `load_dataset()` used by the training scripts
(lxcat_arch_upgrade.py, ablation_study.py, train_lxcat_v4_ensemble.py).

The dataset layout (produced by scripts/run_ml_dataset_generation.py) is:

    results/ml_dataset/<mode>/
        dataset_manifest.json                    # list of all cases + locked params
        P<power>W_p<p>mT_xAr<frac>/
            summary.json                          # per-case scalar outputs
            nF.npy, nSF6.npy, ne.npy, Te.npy ...  # per-case 2D fields on the mesh

The mesh is deterministic (same reactor geometry for every case), so we cache
it from the first case encountered. Coordinates (r, z) are read via the
Mesh2D / build_geometry_mask code path from the simulation config.

Public API:
  load_dataset(mode='legacy', val_frac=0.15, enhanced_features=False) -> train, val, meta, val_idx
  get_mesh() -> rc, zc, inside          # cached after first call
"""
from __future__ import annotations

import json
import os
import sys
from typing import Tuple

import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
# The generator writes to <solver tree>/results/ml_dataset (i.e. model/results/ml_dataset
# in this package), which is not under this scripts/ tree. Export DTPM_DATASET_BASE to
# point the loader at it, or symlink it here.
DATASET_BASE = os.environ.get(
    'DTPM_DATASET_BASE',
    os.path.join(PROJECT_ROOT, 'results', 'ml_dataset'))

# Reactor geometry normalisation constants (from model/config/default_config.yaml)
R_PROC = 0.105      # processing-chamber radius [m]
Z_TOP = 0.234       # full domain height (L_proc + L_apt + L_icp + margin) [m]

_MESH_CACHE = None   # populated by _load_mesh()


def _load_mesh():
    """Build mesh + inside mask via the same code path the simulations used."""
    global _MESH_CACHE
    if _MESH_CACHE is not None:
        return _MESH_CACHE

    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
    from dtpm.core.config import SimulationConfig
    from dtpm.core.mesh import Mesh2D
    from dtpm.core.geometry import build_geometry_mask

    # PROJECT_ROOT is this scripts tree's parent, which holds no config/ in this package.
    # The solver config lives under model/; DTPM_MODEL_ROOT overrides for other layouts.
    _model_root = os.environ.get(
        'DTPM_MODEL_ROOT',
        os.path.abspath(os.path.join(HERE, '..', '..', '..', 'model')))
    config_path = os.path.join(_model_root, 'config', 'default_config.yaml')
    config = SimulationConfig(config_path)
def _reactor_geometry(cfg):
    """Reactor-geometry section, located by content rather than by section name.

    Generators marked [dtpm] read their config from the external pipeline tree, whose
    section carries a different key. Matching on the fields avoids recording that name.
    """
    sec = getattr(cfg, "reactor_geometry", None)
    if sec:
        return sec
    raw = getattr(cfg, "_config", None)
    if raw is None and isinstance(cfg, dict):
        raw = cfg
    if isinstance(raw, dict):
        named = raw.get("reactor_geometry")
        if isinstance(named, dict) and named:
            return named
        for value in raw.values():
            if isinstance(value, dict) and {"R_icp", "R_proc"} <= set(value):
                return value
    return {}
def get_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _load_mesh()


def _manifest_for_mode(mode: str) -> dict:
    path = os.path.join(DATASET_BASE, mode, 'dataset_manifest.json')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset manifest not found: {path}. "
            f"Run scripts/run_ml_dataset_generation.py --mode {mode} first."
        )
    with open(path) as f:
        return json.load(f)


def load_dataset(mode: str = 'legacy', val_frac: float = 0.15,
                 enhanced_features: bool = False):
    """Load a flattened per-cell dataset from the ml_dataset layout.

    Returns
    -------
    train, val : dict
        Each has keys 'r', 'z', 'P', 'p', 'Ar', 'lnF', 'lnSF6', 'case'
        (+ 'Pp', 'PAr', 'pAr', 'logp', 'inv_p' if enhanced_features).
    meta : list
        Per-case dicts with P_rf, p_mTorr, frac_Ar, file (path), idx.
    val_idx : set
        Case indices (int) held out in the validation split.
    """
    manifest = _manifest_for_mode(mode)
    runs = manifest['runs']
    # Assign integer indices for the case-level split
    for i, r in enumerate(runs):
        r['idx'] = i

    n = len(runs)
    np.random.seed(42)
    perm = np.random.permutation(n)
    n_val = max(int(n * val_frac), 10)
    val_idx = set(perm[:n_val].tolist())

    rc, zc, inside = _load_mesh()
    Nr, Nz = len(rc), len(zc)

    cols = ['r', 'z', 'P', 'p', 'Ar', 'lnF', 'lnSF6', 'case']
    if enhanced_features:
        cols += ['Pp', 'PAr', 'pAr', 'logp', 'inv_p']
    data = {k: [] for k in cols}
    meta = []

    for run in runs:
        case_id = run['case_id']
        case_dir = os.path.join(DATASET_BASE, mode, case_id)
        nF_path = os.path.join(case_dir, 'nF.npy')
        nSF6_path = os.path.join(case_dir, 'nSF6.npy')
        if not (os.path.exists(nF_path) and os.path.exists(nSF6_path)):
            continue
        nF = np.load(nF_path)
        nSF6 = np.load(nSF6_path)
        if nF.shape != (Nr, Nz):
            raise ValueError(
                f"{case_id}: nF shape {nF.shape} != mesh ({Nr}, {Nz})"
            )

        P_val = float(run['P_rf_W'])
        p_val = float(run['p_mTorr'])
        Ar_val = float(run['x_Ar'])
        meta.append({
            'idx': run['idx'],
            'case_id': case_id,
            'P_rf': P_val,
            'p_mTorr': p_val,
            'frac_Ar': Ar_val,
            'file': case_id,
        })

        mask = inside & (nF > 0) & (nSF6 > 0)
        ii, jj = np.where(mask)
        for i, j in zip(ii, jj):
            data['r'].append(rc[i])
            data['z'].append(zc[j])
            data['P'].append(P_val)
            data['p'].append(p_val)
            data['Ar'].append(Ar_val)
            data['lnF'].append(np.log10(max(nF[i, j], 1e6)))
            data['lnSF6'].append(np.log10(max(nSF6[i, j], 1e6)))
            data['case'].append(run['idx'])
            if enhanced_features:
                data['Pp'].append(P_val * p_val)
                data['PAr'].append(P_val * Ar_val)
                data['pAr'].append(p_val * Ar_val)
                data['logp'].append(np.log(max(p_val, 0.1)))
                data['inv_p'].append(1.0 / max(p_val, 0.1))

    arrays = {k: np.array(v, dtype=np.float32) for k, v in data.items()}
    arrays['case'] = arrays['case'].astype(np.int32)
    in_train = np.array([c not in val_idx for c in arrays['case']])
    train = {k: v[in_train] for k, v in arrays.items()}
    val = {k: v[~in_train] for k, v in arrays.items()}
    return train, val, meta, val_idx
