#!/bin/bash
#SBATCH --job-name=ml_dataset_gen
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=YOUR_CPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128g
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
# ML training dataset: 220 cases, LEGACY mode (the validated physics — Level-2 showed
# the lxcat mode does not self-sustain), full production flag set incl. calibration.
set -eo pipefail
# submission path of the original run; override with DTPM_JOB_ROOT to relocate
# (in this package the equivalent tree is <package>/model)
cd "${DTPM_JOB_ROOT:?export DTPM_JOB_ROOT to the solver tree (the directory holding src/ and scripts/); see cluster/README.md}"
module load pytorch-conda/2.8 2>&1 || true
[ -n "${CONDA_PREFIX:-}" ] && source "$CONDA_PREFIX/etc/profile.d/conda.sh" && conda activate base || true
export PYTHONPATH="$PWD/src:$PWD/scripts:${PYTHONPATH:-}"
export ML_DATASET_MODE=legacy OMP_NUM_THREADS=2
# The production closure. In this package the file is model/production_flags.json (one level up
# from scripts/); upstream it sat in scripts/. Either path works via this variable.
export DTPM_FIX_FLAGS="${DTPM_FIX_FLAGS:-$PWD/production_flags.json}"
# Default resolves from <package>/model (DTPM_JOB_ROOT=$PWD/model); upstream it was
# scripts/run_ml_dataset_generation.py inside PHASE2_ROOT.
python "${DTPM_DATASET_SCRIPT:-../ml/run_ml_dataset_generation.py}" \
    --mode legacy --workers 14 2>&1 | tail -12
echo '=== manifest ==='
python3 -c "
import json
m=json.load(open('results/ml_dataset/legacy/dataset_manifest.json'))
print({k:m[k] for k in list(m)[:6]})
" 2>/dev/null || ls results/ml_dataset/legacy | head -5

# Sentinel. Six downstream jobs (01_arch_sweep, 02*, train_one_species) refuse to start
# without it. The archived job wrote it; this one is the ad-hoc production job, which
# did not, so it is written here after the case count is confirmed.
COUNT=$(find results/ml_dataset/legacy -name summary.json 2>/dev/null | wc -l | tr -d " ")
if [ "$COUNT" = "220" ]; then
  touch results/.cluster_step0.done
  echo "==> sentinel written: results/.cluster_step0.done ($COUNT/220 cases)"
else
  echo "FAIL: expected 220 cases, got $COUNT -- sentinel NOT written" >&2
  exit 3
fi
