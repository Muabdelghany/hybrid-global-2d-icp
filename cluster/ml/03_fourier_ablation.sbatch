#!/usr/bin/env bash
#SBATCH --job-name=dtpm-fourier-ablation
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=YOUR_CPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=results/cluster_logs/03_fourier_ablation-%j.log
#SBATCH --error=results/cluster_logs/03_fourier_ablation-%j.log
#
# R2.2(a) Fourier-vs-raw ablation on the E3 (separate-heads) winning architecture.
# Output: results/ml_ablation_fourier/fourier_ablation.json
set -euo pipefail
mkdir -p results/cluster_logs results/ml_ablation_fourier
echo "==> Job $SLURM_JOB_ID on $(hostname) at $(date -Iseconds)"

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
# Script location. Upstream these lived in ${DTPM_ML_SCRIPTS}/ inside PHASE2_ROOT; in this
# package they are ml/two_species/scripts/ and ml/all_species/scripts/. Export
# DTPM_ML_SCRIPTS to point at whichever tree you are running from.
export DTPM_ML_SCRIPTS="${DTPM_ML_SCRIPTS:-${DTPM_ML_SCRIPTS}}"

srun python3 ${DTPM_ML_SCRIPTS}/fourier_ablation.py

OUT=results/ml_ablation_fourier/fourier_ablation.json
[[ -f "$OUT" ]] || { echo "FAIL: $OUT not produced." >&2; exit 3; }
echo "==> Fourier ablation complete at $(date -Iseconds)"
