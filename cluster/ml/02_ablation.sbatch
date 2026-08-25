#!/usr/bin/env bash
#SBATCH --job-name=dtpm-lxcat-ablate
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=YOUR_CPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=07:00:00
#SBATCH --output=results/cluster_logs/02_ablation-%j.log
#SBATCH --error=results/cluster_logs/02_ablation-%j.log
#
# Step 2: 5 configs × 3 seeds = 15 runs via ProcessPoolExecutor (workers=5).
# CPU partition, runs in parallel with Step 1 on the GPU partition.
# Doc: SUPERVISOR_ML_HANDOFF.md "Step 2 — LXCat ablation study".
# Output: results/ml_ablation_lxcat/ablation_results.{json,md}
# Sentinel: results/.cluster_step2.done

set -euo pipefail

mkdir -p results/cluster_logs

echo "==> Job $SLURM_JOB_ID on $(hostname) at $(date -Iseconds)"

# Guard: Step 0 must have completed (NOT Step 1 — ablation is Step 0-dependent only)
if [[ ! -f results/.cluster_step0.done ]]; then
  echo "FAIL: results/.cluster_step0.done missing — Step 0 didn't complete cleanly." >&2
  exit 2
fi

# submission path of the original run; override to relocate

export PHASE2_ROOT="${PHASE2_ROOT:?export PHASE2_ROOT to the electron-kinetics tree; see cluster/README.md}"
if [[ -z "${PHASE2_ROOT:?export PHASE2_ROOT to the electron-kinetics tree; see cluster/README.md}" ]]; then
  echo "FAIL: PHASE2_ROOT not set." >&2
  exit 2
fi

module load miniforge3-python
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dtpm-lxcat
set -u

# Critical: doc gotcha #2. Without thread caps, 5 workers × multi-threaded
# BLAS = load average 225 and ~5x slowdown.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Dataset mode. This job as submitted used lxcat, and that is kept as the default so
# the archived job reproduces itself. The PUBLISHED two-species ensemble and ablation
# used the LEGACY (the production closure) dataset instead -- see 03_ensemble_production.sbatch and
# data/surrogate/split_manifest.json. Export ML_DATASET_MODE to switch.
export ML_DATASET_MODE="${ML_DATASET_MODE:-lxcat}"
echo "==> Launching ablation (workers=5, ML_DATASET_MODE=$ML_DATASET_MODE)"
# Script location. Upstream these lived in ${DTPM_ML_SCRIPTS}/ inside PHASE2_ROOT; in this
# package they are ml/two_species/scripts/ and ml/all_species/scripts/. Export
# DTPM_ML_SCRIPTS to point at whichever tree you are running from.
export DTPM_ML_SCRIPTS="${DTPM_ML_SCRIPTS:-${DTPM_ML_SCRIPTS}}"

srun python3 ${DTPM_ML_SCRIPTS}/run_ablation_parallel.py --workers 5

if [[ ! -f results/ml_ablation_lxcat/ablation_results.json ]]; then
  echo "FAIL: ablation_results.json not produced." >&2
  exit 3
fi

touch results/.cluster_step2.done
echo "==> Step 2 complete at $(date -Iseconds)"
