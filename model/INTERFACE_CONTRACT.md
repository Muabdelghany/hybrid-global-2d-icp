# The 0D <-> 2D interface — what actually crosses it

Part of the data repository for *A hybrid global–2D model for SF₆/Ar inductively coupled plasma etchers with neural-network acceleration*.

Derived by tracing data flow through this certified tree rather than from the call
signatures, which are easy to misread: the arguments that look like 2D feedback are in
fact the 0D warm-starting itself.

Method: enumerate every read of `result_0D` anywhere in `src/`, then classify each by
whether the 2D later overwrites it (a seed) or keeps consuming it (a coefficient);
then enumerate every argument of the `solve_0D` call inside the outer loop.

## 0D -> 2D: exactly four quantities

| quantity | read at | role in the 2D | refreshed each outer pass? | 2D overwrites it? |
|---|---|---|---|---|
| `Te` | `m11:253` | initial field, `Te = where(inside, result_0D['Te'], 0)` | no | **yes** — the Te equation solves it |
| `ne` | `m11:251` | initial field via `prescribe_bessel_cosine` | no | **yes** — the ambipolar PDE solves it |
| `alpha` | `m11:218`, **`m11:368`** | coefficient in `D_a` and the Bohm speed | **yes** | no — no 2D alpha equation in production |
| `nArm` | `m11:282`, **`m11:369`** | source/loss coefficient, three terms | **yes** | no — no 2D Ar* equation |

So two are genuine warm-start seeds that wash out, and **two are persistent
coefficients** refreshed at every outer pass. `nArm` is a plain `float`, i.e. imported
as a single spatially uniform value.

`nArm` is consumed in `sf6_rates.py` by:

- `electron_source` (`:191` stepwise ionization `Ar_iz_m * nArm`, `:192` Penning),
- `fluorine_source` (`:204-206` Penning + quenching on SF6 and F2),
- `energy_loss_density` (defined at `:211`; the 12 + 4.95 eV stepwise channel is the
  `Ar_iz_m * nArm` term at `:241`).

Measured share at the reference point, obtained by evaluating each source with and
without `nArm`. Run at the real config geometry (`R_icp` 0.038 m, `L_icp` 0.150 m from
`config/default_config.yaml`); it returns alpha = 0.0141 against the paper's 0.0136, so
it is representative. `nArm` = 6.25e16 m^-3, 0.73% of n_e:

| term | Ar* share |
|---|---|
| electron source | **10.9%** |
| electron energy loss | 3.7% |
| fluorine source | 0.02% |

## 2D -> 0D: exactly one quantity

`m11:347-364` calls `solve_0D` with:

- **`eta`** (`:353`) — the only quantity carrying 2D information back.
- Operating point (`P_rf`, `p_mTorr`, `frac_Ar`, `Q_sccm`, `Tgas`), geometry
  (`R_icp`, `L_icp`), config (`ne_cap`, tolerances) — inputs, not interface traffic.
- `init_Te`, `init_ne`, `init_alpha`, `init_ns` (`:355-356`) — these come from
  `result_0D`, i.e. the 0D's **own** previous state. That is 0D->0D warm starting, not
  an interface crossing. Mistaking these for feedback is the easiest error to make here.

## Why Ar* is the only species density imported

It is not a preference among candidates; it is the unique residual.

- `sf6_chemistry.py:25` — `SPECIES = ['SF6','SF5','SF4','SF3','SF2','SF','S','F','F2']`.
  The 2D transports those nine, so importing 0D values for them would be redundant and
  would inject the 0D's answer into the 2D, which is precisely what the model exists to
  avoid.
- Ar ground state is algebraic from the operating point:
  `te_energy_transport.py:320`, `nAr = oper['frac_Ar'] * ng`. No import needed.
- The 2D neutral seed comes from the feed, not the 0D: `m11:258`,
  `nSF6_feed = ng * (1 - frac_Ar)`.

Ar* is the only species that appears in the 2D source terms while having **neither** a
2D transport equation **nor** an operating-point expression: its density is set by a
local balance (electron excitation against wall loss and heavy-particle quenching), so
it cannot be recovered from p, T_g and x_Ar. It therefore has to come from the 0D.

## Is importing a uniform scalar the right treatment? Measured answer

Two numbers settle it.

**1. Ar\* is local, so no transport equation is needed.** Its destruction frequency at
the reference point is 3.65e5 s^-1 (electron collisions 3.61e5, heavy-particle quenching
4.2e3), a 2.7 us lifetime. With a hard-sphere diffusion coefficient of 1.55 m^2/s the
diffusion length is

    sqrt(D / nu) = 2.1 mm      against a ~200 mm chamber, so lambda/L ~ 0.01.

Ar* therefore lives where it is made. Giving it a transport PDE would buy almost nothing.

**2. But the scalar is evaluated at the wrong temperature.** Because electron collisions
supply 98%+ of the destruction, n_e cancels out of the local balance and

    nArm / nAr0  ~  k_Ar_exc(Te) / (k_Ar_iz_m(Te) + k_Ar_q(Te))

is a function of **Te alone**. That ratio is steep:

| Te (eV) | nArm/nAr0 | relative |
|---|---|---|
| 2.42 — the 0D value, and what is imported | 6.77e-4 | x1.00 |
| 3.30 — 2D source region | 1.51e-3 | x2.23 |
| 3.54 — archived 2D <Te> | 1.75e-3 | x2.58 |
| 4.00 — 2D source region | 2.19e-3 | x3.23 |

The 0D solves at <Te> = 2.42 eV while the 2D runs at 3.3-4.0 eV through the ICP source,
so the imported Ar* is **low by a factor of about 2.2-3.2 exactly where the source terms
are largest**.

**Conclusion.** The weakness is not the missing transport, it is that a Te-sensitive
local equilibrium is imported at the 0D's volume-averaged Te instead of being evaluated
at the local Te(r,z) the 2D already has. The principled refinement is therefore cheap:
replace the imported scalar with the same algebraic expression evaluated per cell. No
new PDE, no new species, no new transport. Until that is done, the uniform scalar is a
disclosed approximation affecting a channel that carries about 11% of the electron
source and 3.7% of the electron energy loss.

## The source comments

The two comment blocks in `m11_plasma_chemistry.py` that describe the interface, at the
`nArm` import and at the outer-loop header, state what this document states: `nArm` is a
refreshed coefficient rather than a warm start, it feeds three source terms, and four
quantities cross 0D to 2D while one crosses back. Earlier revisions of those comments
described `nSF6` as part of the interface, which it is not, and omitted `nArm` from the
outer-loop summary. Trace the data flow if a comment and the code ever appear to disagree.
