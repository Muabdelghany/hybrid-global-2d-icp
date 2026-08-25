#!/usr/bin/env bash
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/cluster_logs/train_species-%j.log
#SBATCH --error=results/cluster_logs/train_species-%j.log
#
# Per-species ensemble trainer.  Submitter must export SPECIES=<name>
# (e.g. SPECIES=nF, ion_ne, Te) via `sbatch --export=SPECIES=...`.

set -euo pipefail

mkdir -p results/cluster_logs

if [[ -z "${SPECIES:-}" ]]; then
  echo "FAIL: SPECIES env var not set." >&2
  exit 2
fi

echo "==> Job $SLURM_JOB_ID on $(hostname) at $(date -Iseconds)"
echo "==> SPECIES=$SPECIES"

if [[ ! -e results/.cluster_step0.done ]]; then
  echo "FAIL: results/.cluster_step0.done missing — dataset not built." >&2
  exit 2
fi

# submission path of the original run; override to relocate

export PHASE2_ROOT="${PHASE2_ROOT:?export PHASE2_ROOT to the electron-kinetics tree; see cluster/README.md}"
module load miniforge3-python
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dtpm-lxcat
set -u

# Dataset mode. The 21-channel trainer hard-codes lxcat at ml/all_species/scripts/
# train_species.py:155, so this job is lxcat by construction, not by convention.
# Exporting ML_DATASET_MODE does NOT change what train_species.py loads.
export ML_DATASET_MODE="${ML_DATASET_MODE:-lxcat}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "==> Launching trainer for ${SPECIES}"
# Script location. Upstream these lived in ${DTPM_ML_SCRIPTS}/ inside PHASE2_ROOT; in this
# package they are ml/two_species/scripts/ and ml/all_species/scripts/. Export
# DTPM_ML_SCRIPTS to point at whichever tree you are running from.
export DTPM_ML_SCRIPTS="${DTPM_ML_SCRIPTS:?export DTPM_ML_SCRIPTS to ml/all_species/scripts for this job}"

srun python3 ${DTPM_ML_SCRIPTS}/train_species.py \
  --species "${SPECIES}" \
  --out results/ml_production_ensemble_all_species

OUT_SUMMARY="results/ml_production_ensemble_all_species/${SPECIES}/summary.json"
if [[ ! -f "$OUT_SUMMARY" ]]; then
  echo "FAIL: $OUT_SUMMARY not produced." >&2
  exit 3
fi

echo "==> Training for ${SPECIES} complete at $(date -Iseconds)"
