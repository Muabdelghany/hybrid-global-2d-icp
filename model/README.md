# The solver

Part of the data repository for *A hybrid global–2D model for SF₆/Ar inductively coupled plasma etchers with neural-network acceleration*.

`src/dtpm/` is the production source for the hybrid global–2D model, together with the
configuration it ran under and two driver scripts. This is the code that produced the results in
the paper.

```
src/dtpm/       core/       grid, geometry, configuration
                modules/    m01–m11, the pipeline stages
                chemistry/  the global model, rate tables, wall chemistry
                solvers/    ambipolar diffusion, species and electron-energy transport
                utils/
config/         default_config.yaml, test_config.yaml
scripts/        run_one_verification_case.py, run_parameter_sweeps.py
production_flags.json   the production closure as a flag set
VERIFICATION_CASE.json    expected output of the reference solve
MD5SUMS_certified.txt  checksums over config/ and src/, 43 files
INTERFACE_CONTRACT.md  what crosses the 0D–2D interface
```

---

## Running a single operating point

```bash
export PYTHONPATH=src:scripts
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/run_one_verification_case.py \
    --power 1000 --pressure 10 --x-ar 0.10 --bias --p-bias-w 200 \
    --gamma-al 0.155 --f-e-bias 1.0 --bias-closure f_e_bias \
    --te-closure energy_transport --per-cell-rates --depleted-att --ec-wide \
    --ne-cap 1e20 --reconciled-rates --outer-loop --mode legacy \
    --out results/reference
```

A converged solve takes roughly three to four minutes on one core and writes `summary.json`
plus 25 field arrays. Compare the result against `VERIFICATION_CASE.json`, which carries the expected scalars to
full precision and a SHA-256 digest for each field array. Field digests match bit-for-bit when
BLAS is held to a single thread, as above.

---

## The production closure

The flag set the paper's runs used, held in `production_flags.json` and exported shell-wide by the batch
scripts:

```json
{ "coupling.te_closure": "energy_transport",  "coupling.outer_loop": true,
  "chemistry.per_cell_rates": true,           "chemistry.depleted_attachment": true,
  "chemistry.ec_clip_wide": true,             "chemistry.ne_cap_0D": 1e20,
  "chemistry.reconciled_rates": true,         "wall_chemistry.gamma_Al": 0.155,
  "bias.closure": "f_e_bias",                 "bias.f_e_bias": 1.0 }
```

**Read the flag branch, not the default.** The defaults are set so that running with the flags
off reproduces the earlier baseline model exactly, which makes the two closures directly
comparable but means a default value in the source does not describe what the paper's runs did.
Two places where that distinction changes the answer:

- `m11_plasma_chemistry.py`, `_ec_clip` at line 157. The expression is
  `(10.0, 2000.0) if ec_clip_wide else (80.0, 400.0)`. Production runs set `ec_clip_wide`, so the
  live 2D window is [10, 2000] eV, matching the 0D. The default branch suggests an [80, 400]
  window that the production configuration never uses.
- `solve_Te_local_power_balance` in the same module is not reached under the production closure,
  because `te_closure = energy_transport` selects `solvers/te_energy_transport.py` instead.

Every run writes a `fix_provenance` block into its `summary.json` recording the flags it actually
used. That block is the authority for any given result.

---

## Verifying the source

```bash
md5sum -c MD5SUMS_certified.txt
```

All 43 files should verify. `MD5SUMS_certified.txt` is the single manifest for this tree; a clean
43/43 means the solver you have is bit-for-bit the one this package ships.

The archived run records under `../data/*/code_md5.txt` carry the checksum of the source
as it executed on the cluster, and are deliberately left at that value: they record what ran.
That copy and this one differ in comments, docstrings and identifier names, and in nothing that
executes. Two independent checks establish it.

*Structural.* Every file parses to the same abstract syntax tree as the copy it was derived from
once the renamed identifiers are mapped across, so the differences are confined to comments,
docstrings and string constants.

*Numerical.* The reference case was solved before and after each editing pass, with BLAS held to
one thread. All 25 field arrays are bit-identical and all 74 `summary.json` keys agree, including
η = 0.9444527265526386 and a center-to-wall [F] drop of 74.00886012296833%.

Of the two, the numerical check is the one that carries the weight. A syntax-tree comparison
passes through a renaming that silently rebinds a name already in scope, and such a change can
move a result while leaving the tree identical. Only a bit-level comparison of the solved fields
rules it out.

Both checks were run against the copy this package was built from, which is not shipped here, so
they record how the source was prepared rather than something a reader can rerun end to end. The
half a reader can rerun is the numerical one: solve the reference case from `src/` and compare
against `VERIFICATION_CASE.json`. Agreement there is what ties this source to the published numbers.

---

## Configuration compatibility

The geometry section is read through an accessor that resolves it by content rather than by key
name, so a configuration written against either the current or an earlier section name loads
unchanged. Generators that take their configuration from an external pipeline tree go through the
same helper.

`config/*.yaml` describes an industrial ICP etcher of the class the manuscript models, under the
neutral reference name **IPI ICP**. The dimensions are those the manuscript itself reports.

---

## The 0D–2D interface

`INTERFACE_CONTRACT.md` sets out which quantities actually cross the interface in each direction,
derived by tracing the data flow through the source. It distinguishes warm-start seeds, which the
2D later overwrites, from persistent coefficients, which the 2D keeps consuming and the outer
loop refreshes. Read it before drawing conclusions about the coupling from the call signatures
alone, because the two are easy to confuse.
