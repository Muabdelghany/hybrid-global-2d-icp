# Cluster jobs

Part of the data repository for *A self-consistent hybrid global–2D model for SF₆/Ar inductively coupled plasma etchers with neural-network surrogate acceleration*.

These are the batch scripts as submitted rather than tidied rewrites. Every solver and
machine-learning result in the paper came from one of them, so the resource requests, the flag
strings and the job structure are the real ones. The jobs ran on a Slurm cluster. Account and
partition names ship as placeholders; set them to your own site's before submitting.

## Running a job

Each script takes its working tree from a variable rather than carrying a site path, and fails
immediately if the variable is unset. Submit from the package root with the overrides exported:

```bash
cd <package>                              # so the cluster/... paths resolve
export DTPM_JOB_ROOT=$PWD/model           # solver jobs: the tree holding src/ and scripts/
export DTPM_ML_SCRIPTS=$PWD/ml/two_species/scripts
export DTPM_FIX_FLAGS=$PWD/model/production_flags.json
sbatch cluster/solver/joint_grid.sbatch
```

Slurm resolves the `#SBATCH --output` paths against the submission directory and opens them
before the script body runs, so `cluster/logs/`, `cluster/ml/logs/` and `cluster/solver/logs/`
are pre-created here. The `mkdir -p` inside some jobs runs too late to help and is kept only
because it is what the original job did.

Three jobs need more than a root override and say so in their own headers.
`cluster/solver/mesh_convergence.sbatch` ran against a separate solver checkout rather than the
production tree; `cluster/ml/00_dataset_gen.sbatch` calls a script that lives in the package's `ml/`
folder rather than under the solver tree; and `cluster/ml/train_one_species.sbatch` is the 21-channel
trainer, so its `DTPM_ML_SCRIPTS` must point at `ml/all_species/scripts` rather than the
two-species directory used by the other training jobs:

```bash
export DTPM_ML_SCRIPTS=$PWD/ml/all_species/scripts
sbatch --export=ALL,SPECIES=nF cluster/ml/train_one_species.sbatch
```

The machine-learning jobs also need `PHASE2_ROOT`, pointing at the electron-kinetics tree that
supplies the LXCat rate interface. Every one of these variables is required rather than
defaulted, so a job submitted without them stops immediately with a message naming the variable
instead of running against the wrong tree.

## The two environments

| jobs | environment | pinned |
|---|---|---|
| `ml/*.sbatch` | conda env `dtpm-lxcat` | yes, `environment.yml` |
| `solver/*.sbatch` | site PyTorch conda module, module `base` env | no |

The solver environment was not captured. Those jobs ran against whatever the site module provided
at the time. The solver is pure NumPy and SciPy and we have seen no sensitivity to versions, but
the specification does not exist and we say so rather than imply a pin. `environment.yml` covers
the machine-learning side only.

## What each job produces

| script | produces | consumed by |
|---|---|---|
| `cluster/solver/run_pb_gated.sbatch` | gated power-balance walkdown | supplement sustainment test; `data/power_balance_study/pb_gated/` |
| `cluster/solver/joint_grid.sbatch` | 60-solve (γ_Al, f_e,bias) grid | figure S1; `data/identifiability_grid/` |
| `cluster/solver/mesh_convergence.sbatch` | four-grid mesh study | figure 18; `data/mesh_convergence/` |
| `cluster/ml/00_dataset_gen.sbatch` | the 220-case training dataset, 346 MB, not shipped | everything below |
| `cluster/ml/01_arch_sweep.sbatch` | architecture sweep | figure 20 |
| `cluster/ml/02_ablation*.sbatch` | recipe ablation, four stages | figure 20 and the ablation table |
| `cluster/ml/03_ensemble_production.sbatch` | the published ensemble and ablation, `legacy` dataset | figures 20, 21, 24 |
| `cluster/ml/03_ensemble.sbatch` | the earlier `lxcat`-variant ensemble | not reported in the paper |
| `cluster/ml/03_fourier_ablation.sbatch` | Fourier-encoding ablation | section 5 |
| `cluster/ml/train_one_species.sbatch` | one of the 21 single-head channels | figure 22 |

The dataset that `cluster/ml/00_dataset_gen.sbatch` writes is excluded from this package for size.
Regenerate it before running any training job; the command is in the top-level `README.md`.
