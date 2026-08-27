# A self-consistent hybrid global–2D model for SF₆/Ar inductively coupled plasma etchers with neural-network surrogate acceleration

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22116275.svg)](https://doi.org/10.5281/zenodo.22116275)

Data and model code accompanying:

> **A self-consistent hybrid global–2D model for SF₆/Ar inductively coupled plasma etchers with
> neural-network surrogate acceleration**
> M. A. Abdelghany, Z. Ngan, D. Qerimi. *Journal of Physics D: Applied Physics.*

If you have read the paper and want to check a number, look at a curve more closely, or replot
something your own way, this repository has what you need. It holds the numbers behind every
figure, the solver that produced them, and the trained surrogate weights. The figures themselves
are in the paper, so they are not duplicated here.

---

## Where to start

**To check or replot a figure**, go to `data/plotted_values/`. Most figures have a directory
there holding the plotted values as plain CSV, one file per panel, with units in the column
headers. Nothing needs to be run; open them in whatever you use. Schematics have no data, and
the three figures that plot two-dimensional fields keep their `.npy` arrays instead, because
flattening a field map to CSV would lose the mesh. `FIGURE_DATA.md` maps every figure number to
whichever applies.

**To look at the underlying physics output**, the directories under `data/` hold the solver
runs themselves: field arrays as `.npy`, per-run scalars as `summary.json`. These are the raw
results the figures were drawn from, not summaries of them.

**To reproduce a result yourself**, `model/` contains the solver. One converged operating point
takes a few minutes on a single core. `model/VERIFICATION_CASE.json` pins one such point exactly,
so you can confirm a local build reproduces our numbers before trusting it on anything else.

This repository deliberately holds data rather than plots. The scripts that drew the published
figures are not included, and neither are the figure files, which you already have in the paper.
What is here is what they were drawn from, so you can check it or redraw it however you prefer.

---

## What is in it

```
data/
  plotted_values/   one directory per figure: the plotted numbers as CSV
  reference_case/   the published reference solve: 1000 W, 10 mTorr, 30% Ar, bias on
                      25 field arrays on the (r,z) mesh, mesh.npz with the coordinates,
                      summary.json, and a README explaining how to plot them
  composition_scan/ the four-arm SF6 composition study (90% and 30%, bias on and off)
  parameter_sweeps/ power and pressure sweeps
  mesh_convergence/ the four-grid resolution study
  identifiability_grid/  the 60-solve (γ_Al, f_e,bias) grid
  calibration_grid/ the γ_Al sweep behind the calibration
  power_balance_study/   the constrained power-balance walkdown
  ambipolar_comparison/  the electropositive / electronegative pair
  coil_resistance_sweep/ the coil-resistance study
  picard_convergence/    iteration history of the reference solve
  sensitivity/      inputs to the sensitivity analysis
  benchmark_0d_model/     this model's 0D benchmark solve
  benchmark_0d_reference/ the published study it is compared against, digitized
  measurements/     digitized experimental traces used for validation
  boltzmann/        two-term Boltzmann swarm table and a sample EEDF
  surrogate/        surrogate metrics, timings, ablations, the train/validation split
  *_prior_closure/  the same runs under the earlier closure (see below)

model/            the solver that produced all of the above
ml/               surrogate training code and all 115 trained checkpoints
cluster/          the batch scripts the studies were submitted with
```

---

## Which figure draws on what

`FIGURE_DATA.md` is the index. In summary:

| figure | what you get |
|---|---|
| 1, 4, 5, 19 | schematics and flowcharts; nothing behind them to publish |
| 2, 3 | the 0D benchmark: our solve in `data/benchmark_0d_model/`, the published comparison in `data/benchmark_0d_reference/`, both flat CSV |
| 6, 9, 10 | two-dimensional field maps; the plotted quantity *is* the `.npy` array in `data/reference_case/`, with `mesh.npz` giving the coordinates |
| 7, 8, 11–18, 20–27, S1 | CSV of the plotted curves in `data/plotted_values/figureNN/` |

Where a figure plots a field map rather than a curve, exporting it to CSV would lose the mesh, so
the `.npy` array is the digest. The mesh is described in `model/config/default_config.yaml`.

---

## Two closures, and why some directories are doubled

The paper reports results under one closure: a steady electron-energy transport equation, an
outer Picard loop on the absorbed power fraction, per-cell rate coefficients, and a wall
recombination coefficient γ_Al = 0.155. Directories named `*_prior_closure` hold the same runs
made under the earlier treatment, a local electron-temperature balance with γ_Al = 0.18. They
are included because two comparisons in the paper are between the two closures. No figure in
the paper is drawn from them.

Unless a filename says `prior_closure`, the data is from the production closure.

---

## Running the solver

To confirm a local build reproduces our numbers, solve the verification point. It is 1000 W,
10 mTorr, **10% Ar**, bias on, under the production closure. Write the output outside the
checkout so the repository stays byte-clean:

```bash
cd model
export PYTHONPATH=src:scripts
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/run_one_verification_case.py \
  --power 1000 --pressure 10 --x-ar 0.10 --bias --p-bias-w 200 \
  --gamma-al 0.155 --f-e-bias 1.0 --bias-closure f_e_bias \
  --te-closure energy_transport --per-cell-rates --depleted-att --ec-wide \
  --ne-cap 1e20 --reconciled-rates --outer-loop --mode legacy \
  --out /tmp/dtpm_check
```

A converged solve takes three to four minutes on one core and writes `summary.json` plus 25 field
arrays. Compare against `model/VERIFICATION_CASE.json`, which holds the expected scalars to full
precision and a SHA-256 digest of each array. With BLAS pinned to one thread, as above, the
digests should match exactly. To check:

```python
import json, hashlib, os
ref = json.load(open("model/VERIFICATION_CASE.json"))
bad = [f for f, h in ref["field_arrays_sha256"].items()
       if hashlib.sha256(open(os.path.join("/tmp/dtpm_check", f), "rb").read()).hexdigest() != h]
print("mismatched arrays:", bad or "none")
```

**This point is not the published reference dataset.** The runs behind figures 6, 9, 10, 11 and
12 are at **30% Ar**, and sit in `data/reference_case/`. Change `--x-ar` to `0.30` to solve that
condition instead; its own `summary.json` records the values to expect.

The solver needs only NumPy, SciPy and PyYAML.

### One thing that will catch you out

The closure above is selected entirely by those command-line flags, and the defaults are set so
that running **without** them reproduces the earlier baseline model rather than the published
one. That was a deliberate choice, because it makes the two closures directly comparable, but it
means a run with the flags omitted does not fail. It completes and returns different physics.

Two consequences. Reading a default value in the source does not tell you what the paper's runs
did. And every run writes a `fix_provenance` block into its `summary.json` recording the flags it
actually used, which is the authority for any given result. The flag set the paper used is in
`model/production_flags.json`; sweep drivers read it from the `DTPM_FIX_FLAGS` environment
variable rather than from the command line.

`model/README.md` covers this in more detail, and `model/INTERFACE_CONTRACT.md` documents what
actually passes between the 0D and 2D parts of the model, which is easy to misread from the call
signatures alone.

---

## The surrogate

`ml/` holds the training code and all 115 trained checkpoints: five seeds for each of the
two-species models, and five seeds for each of 21 single-species channels. Inference metrics can
be recomputed from the weights directly; anything that needs the validation targets also needs
the training dataset, which is not shipped and is regenerated as described below. `data/surrogate/split_manifest.json`
records the exact case-level train and validation split.

The 220-case solver dataset the networks were trained on is not included, being 346 MB of derived
output. `ml/run_ml_dataset_generation.py` regenerates it:

```bash
export DTPM_MODEL_ROOT=$PWD/model
export DTPM_FIX_FLAGS=$PWD/model/production_flags.json
python3 ml/run_ml_dataset_generation.py --mode legacy --workers 14
```

**One note for anyone reusing the architecture.** The deployed checkpoints use a random Fourier
input encoding. `data/surrogate/fourier_ablation.json` shows raw inputs are better on this target
on every measure: root-mean-square error, mean absolute error, maximum error, parameter count and
training time. The encoding is retained here only because it is what the deployed weights were
trained with, and every performance number in the paper was measured on those weights. Prefer the
raw-input configuration for new work. Section 5.2 of the paper says the same.

---

## Verifying you have an intact copy

```bash
python3 verify_package.py            # structural checks over the whole tree
python3 make_tree_manifest.py --check # SHA-256 of every file
cd model && md5sum -c MD5SUMS_certified.txt   # 43/43 on the solver source and config
```

`MD5SUMS_certified.txt` covers the 43 files under `model/config` and `model/src`. It differs from
the checksums recorded when the studies ran, in `data/*/code_md5.txt`, by comments and identifier
names only; nothing that executes differs. The check that establishes this for you is the
reference case above, which reproduces bit for bit.

---

## Reactor geometry

The geometry in `model/config/*.yaml` describes an industrial ICP etcher of the class the
manuscript models, under the neutral reference name **IPI ICP**. The dimensions are those the
manuscript itself reports.

---

## Licence and third-party material

Code is released under the MIT licence (`LICENSE`). The data and the trained weights are
released under CC BY 4.0 (`LICENSE-DATA`). Attribution for material that originates elsewhere, including the
digitized measurement traces and the cross-section source behind the Boltzmann calculations, is
in `THIRD_PARTY_DATA.md`. If you use any of it, cite the original source rather than this
repository.

## Citing

Cite the paper; `CITATION.cff` carries the metadata. To refer to this repository specifically,
cite the archived deposit: **10.5281/zenodo.22116275**. That is the concept DOI, and it always
resolves to the latest version, which is what the paper cites. Each release also has its own
version DOI if you need to pin an exact snapshot.
