#!/usr/bin/env bash
#SBATCH --job-name=dtpm-lxcat-arch
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=YOUR_GPU_PARTITION
# Fallback if the GPU queue is congested (>2 h pending): swap to a smaller GPU partition
##SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=results/cluster_logs/01_arch_sweep-%j.log
#SBATCH --error=results/cluster_logs/01_arch_sweep-%j.log
#
# Step 1: 7 archs × 3 seeds = 21 runs. arch_sweep.py is serial, runs all
# experiments on one device. A100 ~4-8 h vs MPS ~32 h. 12 h cap = 50% margin.
# Doc: SUPERVISOR_ML_HANDOFF.md "Step 1 — LXCat architecture sweep".
# Output: results/ml_arch_sweep_lxcat/{experiment_table.json, .md, <exp>/run_*.json}
# Sentinel: results/.cluster_step1.done

set -euo pipefail

mkdir -p results/cluster_logs

echo "==> Job $SLURM_JOB_ID on $(hostname) at $(date -Iseconds)"

# Guard: Step 0 must have completed
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

# Dataset mode. This job as submitted used lxcat, and that is kept as the default so
# the archived job reproduces itself. The PUBLISHED two-species ensemble and ablation
# used the LEGACY (the production closure) dataset instead -- see 03_ensemble_production.sbatch and
# data/surrogate/split_manifest.json. Export ML_DATASET_MODE to switch.
export ML_DATASET_MODE="${ML_DATASET_MODE:-lxcat}"
echo "==> CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-<unset>}"
python -c "import torch; print(f'  cuda available: {torch.cuda.is_available()}')"

# Defensive: if CUDA falls through to CPU, this run will take ~7 days.
# Background-watch the log for the device line and scancel if not CUDA.
LOG=results/cluster_logs/01_arch_sweep-${SLURM_JOB_ID}.log
(
  sleep 90
  if ! grep -qE 'Device:\s*cuda|cuda available: True' "$LOG" 2>/dev/null; then
    echo "FAIL: arch_sweep did not pick CUDA after 90 s — cancelling to avoid 7-day CPU run." >> "$LOG"
    scancel "$SLURM_JOB_ID"
  fi
) &

echo "==> Launching arch sweep (ML_DATASET_MODE=$ML_DATASET_MODE)"
# Script location. Upstream these lived in ${DTPM_ML_SCRIPTS}/ inside PHASE2_ROOT; in this
# package they are ml/two_species/scripts/ and ml/all_species/scripts/. Export
# DTPM_ML_SCRIPTS to point at whichever tree you are running from.
export DTPM_ML_SCRIPTS="${DTPM_ML_SCRIPTS:-${DTPM_ML_SCRIPTS}}"

srun python3 ${DTPM_ML_SCRIPTS}/arch_sweep.py

# Verify the table landed
TABLE=results/ml_arch_sweep_lxcat/experiment_table.json
if [[ ! -f "$TABLE" ]]; then
  echo "FAIL: $TABLE not produced." >&2
  exit 3
fi
echo "==> experiment_table.json size: $(wc -c < "$TABLE") bytes"

touch results/.cluster_step1.done
echo "==> Step 1 complete at $(date -Iseconds)"
