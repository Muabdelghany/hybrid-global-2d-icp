#!/usr/bin/env bash
#SBATCH --job-name=dtpm-lxcat-ablate-all3
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=YOUR_CPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=results/cluster_logs/02d_ablation_all_three-%j.log
#SBATCH --error=results/cluster_logs/02d_ablation_all_three-%j.log
#
# Single-config ablation for `all_three` with multi-threaded BLAS.
# Same pattern as 02c but for the heavier (2000 epoch) config.

set -euo pipefail

mkdir -p results/cluster_logs

echo "==> Job $SLURM_JOB_ID on $(hostname) at $(date -Iseconds)"

if [[ ! -f results/.cluster_step0.done ]]; then
  echo "FAIL: results/.cluster_step0.done missing." >&2
  exit 2
fi

# submission path of the original run; override to relocate

export PHASE2_ROOT="${PHASE2_ROOT:?export PHASE2_ROOT to the electron-kinetics tree; see cluster/README.md}"
module load miniforge3-python
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dtpm-lxcat
set -u

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
# Dataset mode. This job as submitted used lxcat, and that is kept as the default so
# the archived job reproduces itself. The PUBLISHED two-species ensemble and ablation
# used the LEGACY (the production closure) dataset instead -- see 03_ensemble_production.sbatch and
# data/surrogate/split_manifest.json. Export ML_DATASET_MODE to switch.
export ML_DATASET_MODE="${ML_DATASET_MODE:-lxcat}"
OUT_DIR="results/ml_ablation_lxcat_split"
mkdir -p "$OUT_DIR"
RESULT_FILE="$OUT_DIR/result_all_three.json"

if [[ -f "$RESULT_FILE" ]]; then
  echo "==> $RESULT_FILE already exists; skipping (resume)."
  exit 0
fi

echo "==> Launching all_three training (3 seeds, OMP=$OMP_NUM_THREADS)"
srun python3 -c "
import sys, os, json, time
sys.path.insert(0, '${DTPM_ML_SCRIPTS}')
import ablation_study as abl
import run_ablation_parallel as rap

target = next(e for e in rap.EXPERIMENTS if e['name'] == 'all_three')
print(f'Running config: {target}', flush=True)
t0 = time.time()
summary = rap._worker(target)
wall = time.time() - t0
out = '$RESULT_FILE'
with open(out, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'  [OK] all_three: nF={summary[\"nF_rmse_mean\"]:.5f}+/-{summary[\"nF_rmse_std\"]:.5f}  '
      f'nSF6={summary[\"nSF6_rmse_mean\"]:.5f}  ({wall:.0f}s)', flush=True)
print(f'  Wrote: {out}', flush=True)
"

if [[ ! -f "$RESULT_FILE" ]]; then
  echo "FAIL: $RESULT_FILE not produced." >&2
  exit 3
fi

echo "==> all_three complete at $(date -Iseconds)"
