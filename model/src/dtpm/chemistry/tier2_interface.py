"""Tier-2 PINN Boltzmann-rate interface for the phase-1 Picard loop.

Provides a thin wrapper around ``tier2_pinn.get_rates_pinn.get_rates_pinn``
so that the m11 Picard solver can switch between Maxwellian Arrhenius rates
(tier-1, the existing ``sf6_rates.rates()`` baseline) and Boltzmann rates
derived from the BOLSIG+ / PINN surrogate.

The PINN is evaluated once per outer Picard iteration at a scalar-effective
E/N representative of the ICP volume. The returned rate coefficients overwrite
the aggregate electron-impact channels (``iz_SF6_total``, ``att_SF6_total``,
and the dissociation channels ``d1``–``d5`` proportionally scaled to match
the PINN total). All three-body (Troe), ion-ion, neutral-neutral, wall, and
Ar metastable channels remain on the Arrhenius path because the PINN does
not cover them.

Usage (from m11):

    from dtpm.chemistry import tier2_interface as tier2

    # Picard init:
    if config.chemistry.get('use_boltzmann_rates', False):
        tier2.install_pinn()

    # Inside Picard loop (after EM solve):
    if tier2.is_active():
        E_over_N = tier2.compute_eff_E_over_N(E_theta_rms, n_total_cm3, inside, mesh)
        tier2.refresh(E_over_N, x_Ar=frac_Ar, pressure_mTorr=p_mTorr)

The Arrhenius rates() function in sf6_rates reads the tier-2 cache
transparently when active, so no call-site changes are needed downstream.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np


# -------------------------------------------------------------------------
# Locate and load the Tier-2 PINN. The PINN lives in the Phase-2 project
# sibling folder or in the DTPM repo under
#   Plasma Chemistry Module/SF6_surrogate_and_LXCat/phase2_electron_kinetics/
# We search a small list of candidate paths, pick the first that imports.
# -------------------------------------------------------------------------

_CANDIDATE_PATHS = [
    # Local Phase-1 workspace sibling
    Path(__file__).resolve().parents[4] / "phase2_electron_kinetics",
    # DTPM repo layout
    Path(__file__).resolve().parents[6]
    / "Plasma Chemistry Module"
    / "SF6_surrogate_and_LXCat"
    / "phase2_electron_kinetics",
    # Fallback: respect an environment override
    Path(os.environ.get("PHASE2_ROOT", "")) if os.environ.get("PHASE2_ROOT") else None,
]
_CANDIDATE_PATHS = [p for p in _CANDIDATE_PATHS if p is not None]


def _import_pinn():
    """Try each candidate path until we find a working tier2_pinn package."""
    last_err: Optional[Exception] = None
    for root in _CANDIDATE_PATHS:
        if not root.exists():
            continue
        # Inserting parent so `tier2_pinn` is a valid package name
        sys.path.insert(0, str(root))
        try:
            from tier2_pinn.get_rates_pinn import get_rates_pinn  # type: ignore
            return get_rates_pinn, root
        except Exception as e:  # pragma: no cover - import-time debug path
            last_err = e
            sys.path.remove(str(root))
            continue
    raise ImportError(
        f"Could not locate tier2_pinn package. Tried: "
        f"{[str(p) for p in _CANDIDATE_PATHS]}. Last error: {last_err}"
    )


# Module-state: populated by install_pinn()
_INSTALLED = False
_GET_RATES = None
_PINN_ROOT: Optional[Path] = None

# Cache of the most-recent PINN rates (refreshed per Picard iter). The
# downstream rates() call reads these overrides if set.
_PINN_CACHE: Optional[dict] = None


def install_pinn() -> Path:
    """Load the PINN once at the start of the Picard loop."""
    global _INSTALLED, _GET_RATES, _PINN_ROOT
    if not _INSTALLED:
        _GET_RATES, _PINN_ROOT = _import_pinn()
        _INSTALLED = True
    return _PINN_ROOT


def is_active() -> bool:
    """True when install_pinn() has been called and a rate cache is set."""
    return _INSTALLED and _PINN_CACHE is not None


def current_cache() -> Optional[dict]:
    """Return the most-recent PINN rates (k_iz, k_att, k_diss, k_exc, Te_eff)."""
    return _PINN_CACHE


def refresh(E_over_N_Td: float, x_Ar: float, pressure_mTorr: float = 10.0) -> dict:
    """Evaluate the PINN at a scalar-effective E/N, store in cache.

    Parameters
    ----------
    E_over_N_Td : float
        Effective reduced field in Townsend (1 Td = 1e-21 V m^2).
    x_Ar : float
        Argon mole fraction in [0, 0.9].
    pressure_mTorr : float
        Total pressure in mTorr (currently unused by the PINN but kept for
        future pressure-aware surrogates).

    Returns
    -------
    dict
        PINN rates scaled for this operating point. Keys: Te_eff (eV),
        k_iz, k_att, k_diss, k_exc (all in m^3/s).
    """
    global _PINN_CACHE
    if not _INSTALLED:
        install_pinn()
    # Clamp x_Ar to PINN's valid training range [0, 0.9]
    x_Ar_clamped = float(np.clip(x_Ar, 0.0, 0.9))
    # Clamp E/N to avoid the PINN's invalid-input guard (>0)
    EN = float(max(E_over_N_Td, 1e-3))
    rates = _GET_RATES(np.array([EN]), x_Ar=x_Ar_clamped, pressure_mTorr=pressure_mTorr)
    cache = {
        "E_over_N_Td": EN,
        "x_Ar": x_Ar_clamped,
        "Te_eff": float(rates["Te_eff"][0]),
        "k_iz": float(rates["k_iz"][0]),
        "k_att": float(rates["k_att"][0]),
        "k_diss": float(rates["k_diss"][0]),
        "k_exc": float(rates["k_exc"][0]),
    }
    _PINN_CACHE = cache
    return cache


def clear() -> None:
    """Reset the cache (Tier-1 Arrhenius rates resume on next rates() call)."""
    global _PINN_CACHE
    _PINN_CACHE = None


# -------------------------------------------------------------------------
# Apply cached PINN rates over the Arrhenius rate dict.
# -------------------------------------------------------------------------
# Strategy: preserve the d1-d5 branching ratios from Arrhenius (which
# carries Lallement's published product-channel breakdown), but rescale
# so their sum matches the PINN k_diss aggregate. Overwrite the aggregates
# iz_SF6_total and att_SF6_total directly with PINN values.

_DISS_KEYS = ("d1", "d2", "d3", "d4", "d5")


def apply_overrides(k: dict) -> None:
    """Mutate an Arrhenius rate dict in-place with PINN-cached values.

    No-op when the cache is empty. Called from sf6_rates.rates() when the
    tier-2 switch is active.
    """
    if _PINN_CACHE is None:
        return

    # Aggregate overrides
    k["iz_SF6_total"] = _PINN_CACHE["k_iz"]
    k["att_SF6_total"] = _PINN_CACHE["k_att"]

    # Rescale the five dissociation channels to match PINN k_diss total
    d_sum_arrh = sum(k.get(name, 0.0) for name in _DISS_KEYS)
    if d_sum_arrh > 0 and _PINN_CACHE["k_diss"] > 0:
        scale = _PINN_CACHE["k_diss"] / d_sum_arrh
        for name in _DISS_KEYS:
            if name in k:
                k[name] = k[name] * scale

    # Expose cache metadata as keys for downstream inspection
    k["_tier2_Te_eff"] = _PINN_CACHE["Te_eff"]
    k["_tier2_E_over_N_Td"] = _PINN_CACHE["E_over_N_Td"]


# -------------------------------------------------------------------------
# Helper to compute a scalar-effective E/N from the 2D field + neutral density
# -------------------------------------------------------------------------

def compute_eff_E_over_N(
    E_theta_rms: np.ndarray,
    n_total_m3: float,
    inside: np.ndarray,
    mesh,
) -> float:
    """Volume-averaged effective E/N (in Townsend) over the active plasma.

    E_theta_rms is the per-cell RMS azimuthal E-field from FDTD (V/m).
    n_total_m3 is the gas-phase total neutral density (m^-3).
    """
    if n_total_m3 <= 0:
        return 0.0
    # Volume-averaged |E_rms| in the active plasma region
    mask = inside & (E_theta_rms > 0)
    if not np.any(mask):
        return 0.0
    E_mean = float(np.sum(E_theta_rms[mask] * mesh.vol[mask]) /
                   max(np.sum(mesh.vol[mask]), 1e-30))
    # E/N in V m^2, convert to Townsend (1 Td = 1e-21 V m^2)
    E_over_N = E_mean / n_total_m3
    return E_over_N / 1e-21
