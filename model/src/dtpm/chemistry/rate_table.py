"""Vectorised rate-coefficient access: tabulate once, interpolate per cell.

Why this exists
---------------
`sf6_rates.rates(Te)` is scalar-valued: it takes a float and returns a dict of
floats.  Two solvers need rate coefficients evaluated at the LOCAL Te of every
mesh cell:

  * `te_energy_transport` -- the volumetric electron energy sink, which must be
    the complete `energy_loss_density` expression, not a subset.
  * `multispecies_transport` -- which currently collapses Te(r,z) to one scalar
    (`Te_avg`) and calls `compute_rates` ONCE for the whole domain, so the nine
    neutral species (including the benchmarked atomic-F field) see no spatial
    variation in their rate coefficients at all.

Calling a scalar `rates()` per cell per inner step per Picard iteration is
~1780 x 60 x 20 = 2.1e6 invocations per solve, which is prohibitive.

Tabulating every coefficient on a fixed Te grid and interpolating reproduces
`rates()` to a verified accuracy (see `max_interp_error`, and
tests/test_rate_table.py) at a small fraction of the cost, and returns
dict-of-ARRAYS so downstream expressions vectorise with no change of form.

Accuracy note
-------------
Most channels are Arrhenius-like, k = A exp(-E/Te).  Two choices make the
interpolation accurate rather than merely fine-grained:

  1. Interpolate log(k), not k.  k spans many decades; log(k) does not.
  2. Interpolate against x = 1/Te, not Te.  For a pure Arrhenius channel
     log k = log A - E*(1/Te) is EXACTLY LINEAR in x, so linear interpolation is
     exact up to round-off.  Against Te it is strongly curved: at Te = 0.37 eV
     with E = 12 eV, d(log k)/dTe ~ 88 per eV, which a uniform-in-Te grid cannot
     track -- an earlier uniform-in-Te version of this table reached a 3801%
     worst-case error at Te = 0.28 eV, and still 1.3% with 2000 points.

Channels with a Te^n prefactor (e.g. vib_F2 ~ Te^1.72 exp(-1.55/Te)) acquire a
mild log-log curvature in x, and Ar_el is a clipped polynomial; both are handled
by the same grid and are covered by the measured error below.  Non-positive or
constant channels are carried through linearly.

The achieved accuracy is MEASURED, not assumed -- call `max_interp_error()`, or
run tests/test_rate_table.py, which pins it against the exact scalar rates().
"""

import numpy as np

from .sf6_rates import rates

_TINY = 1e-300


class RateTable:
    """Tabulated, vectorised stand-in for `sf6_rates.rates`.

    Parameters
    ----------
    Te_min, Te_max : float
        Table range [eV].  Values outside are clipped, matching the clipping the
        scalar callers already apply.
    n : int
        Number of tabulation points.

    Usage
    -----
    >>> tab = RateTable()
    >>> k = tab(Te_field)        # Te_field: ndarray -> dict of ndarrays
    >>> k['iz18'].shape == Te_field.shape
    """

    def __init__(self, fn=None, Te_min=0.2, Te_max=30.0, n=2000):
        # `fn` lets this serve BOTH rate modules, which are genuinely different:
        #   * sf6_rates.rates(Te)               -- used by the Te/ne solvers, and
        #     IMPURE (it applies tier-2 PINN overrides from mutable module state),
        #     so a table built from it must be rebuilt after every tier2.refresh().
        #   * sf6_chemistry.compute_rates(Te, ng_cm3, frac_Ar) -- used by the
        #     9-species neutral chemistry, and pure (no tier-2 hook), so its table
        #     is safe to build once per solve. Pass a closure binding ng_cm3/frac_Ar.
        self._fn = fn if fn is not None else rates
        self.Te_min = float(Te_min)
        self.Te_max = float(Te_max)

        # HYBRID grid.  A grid uniform in x = 1/Te is where Arrhenius log-rates
        # are linear, and it concentrates points at low Te where rates vary
        # fastest -- but it starves the high-Te end (at Te ~ 29 eV a uniform-1/Te
        # grid spans ~2 eV per interval, and the worst-case error migrates there).
        # A grid uniform in Te has the opposite failure.  Taking the union of
        # both gives adequate resolution across the whole range for the cost of
        # one extra sort.
        x_arr = np.linspace(1.0 / self.Te_max, 1.0 / self.Te_min, int(n))
        T_from_x = 1.0 / x_arr
        T_lin = np.linspace(self.Te_min, self.Te_max, int(n))
        T_nodes = np.unique(np.concatenate([T_from_x, T_lin]))   # ascending in T

        # np.interp REQUIRES an ascending sample axis.  We interpolate against
        # x = 1/Te, and x descends as Te ascends -- so the node arrays are stored
        # in DESCENDING-Te order, which is ascending in x.  Getting this backwards
        # makes np.interp return silently wrong values rather than raising, so the
        # ordering is asserted below and pinned by tests/test_rate_table.py.
        self.T = T_nodes[::-1]          # descending Te
        self.x = 1.0 / self.T           # ascending x  <- the interpolation axis
        assert np.all(np.diff(self.x) > 0), "interpolation axis must ascend"

        samples = [self._fn(float(t)) for t in self.T]
        proto = samples[0]
        self.keys = [key for key in proto
                     if isinstance(proto[key], (int, float, np.floating))]

        self._log_keys = []
        self._lin_keys = []
        self._tab = {}
        for key in self.keys:
            col = np.array([float(s[key]) for s in samples], dtype=np.float64)
            if np.all(col > 0.0):
                self._tab[key] = np.log(col)
                self._log_keys.append(key)
            else:
                self._tab[key] = col
                self._lin_keys.append(key)

    def __call__(self, Te):
        """Interpolate every coefficient at Te (scalar or ndarray)."""
        Tc = np.clip(np.asarray(Te, dtype=np.float64), self.Te_min, self.Te_max)
        xq = 1.0 / Tc
        out = {}
        for key in self._log_keys:
            out[key] = np.exp(np.interp(xq, self.x, self._tab[key]))
        for key in self._lin_keys:
            out[key] = np.interp(xq, self.x, self._tab[key])
        return out

    def max_interp_error(self, n_probe=997, seed=0):
        """Worst relative error against the exact scalar `rates()`.

        Probes at points deliberately OFF the tabulation nodes (that is where
        interpolation error is largest); returns (max_rel_err, worst_key, worst_Te).
        """
        rng = np.random.default_rng(seed)
        probes = rng.uniform(self.Te_min, self.Te_max, n_probe)
        worst, wkey, wT = 0.0, None, None
        for t in probes:
            exact = self._fn(float(t))
            approx = self(float(t))
            for key in self.keys:
                a, b = float(exact[key]), float(approx[key])
                denom = max(abs(a), _TINY)
                if abs(a) < 1e-30:      # both effectively zero
                    continue
                rel = abs(b - a) / denom
                if rel > worst:
                    worst, wkey, wT = rel, key, float(t)
        return worst, wkey, wT


class CellRates:
    """Per-cell view of tabulated coefficients, with the scalar k['key'] interface.

    The 9-species inner loop is pure Python over cells and reads coefficients as
    `k['d1']`, one domain-wide scalar. Making the rates per-cell without rewriting
    every source/loss expression means giving it an object that answers the SAME
    `k['key']` call with the value at the current cell:

        k = CellRates(table(Te_flat))
        for idx in range(N):
            k._i = idx
            ...  # every existing k['...'] expression is untouched

    Keeping those expressions byte-identical is deliberate: they encode the
    reaction network, and re-typing ~40 of them to add an index is exactly the
    kind of edit that introduces a silent transcription error.

    Values are stored as Python lists rather than ndarrays because scalar indexing
    of an ndarray yields np.float64, which is markedly slower than a float across
    the ~1e7 scalar operations of the inner loop.
    """

    __slots__ = ('_k', '_i', '_n')

    def __init__(self, kf):
        self._k = {}
        self._n = None
        for key, val in kf.items():
            if hasattr(val, 'tolist') and getattr(val, 'ndim', 0) > 0:
                lst = val.tolist()
                self._k[key] = lst
                if self._n is None:
                    self._n = len(lst)
            else:
                self._k[key] = float(val)   # Te-independent (e.g. Troe nr* rates)
        self._i = 0

    def __getitem__(self, key):
        v = self._k[key]
        return v[self._i] if type(v) is list else v

    def __contains__(self, key):
        return key in self._k

    def keys(self):
        return self._k.keys()

    def __len__(self):
        return self._n if self._n is not None else 0
