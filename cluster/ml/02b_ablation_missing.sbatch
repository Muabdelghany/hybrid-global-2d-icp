#!/usr/bin/env bash
#SBATCH --job-name=dtpm-lxcat-ablate-missing
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=YOUR_CPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/cluster_logs/02b_ablation_missing-%j.log
#SBATCH --error=results/cluster_logs/02b_ablation_missing-%j.log
#
# Recovery: J2 (17891232) timed out after completing none/bias_only/epochs_only.
# This sbatch runs ONLY the two missing configs (reg_only, all_three) by
# monkey-patching EXPERIMENTS in run_ablation_parallel before main() reads it.
# Output goes to results/ml_ablation_lxcat_missing/ to avoid clobbering anything.

set -euo pipefail

mkdir -p results/cluster_logs

echo "==> Job $SLURM_JOB_ID on $(hostname) at $(date -Iseconds)"

if [[ ! -f results/.cluster_step0.done ]]; then
  echo "FAIL: results/.cluster_step0.done missing — Step 0 didn't complete." >&2
  exit 2
fi

# submission path of the original run; override to relocate

export PHASE2_ROOT="${PHASE2_ROOT:?export PHASE2_ROOT to the electron-kinetics tree; see cluster/README.md}"
module load miniforge3-python
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dtpm-lxcat
set -u

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Dataset mode. This job as submitted used lxcat, and that is kept as the default so
# the archived job reproduces itself. The PUBLISHED two-species ensemble and ablation
# used the LEGACY (the production closure) dataset instead -- see 03_ensemble_production.sbatch and
# data/surrogate/split_manifest.json. Export ML_DATASET_MODE to switch.
export ML_DATASET_MODE="${ML_DATASET_MODE:-lxcat}"
echo "==> Launching ablation (configs=reg_only,all_three, workers=2)"
srun python3 -c "
import sys, os
sys.path.insert(0, '${DTPM_ML_SCRIPTS}')
import ablation_study as abl
abl.OUT = os.path.join(abl.REPO, 'results', 'ml_ablation_lxcat_missing')
os.makedirs(abl.OUT, exist_ok=True)
import run_ablation_parallel as rap
rap.EXPERIMENTS = [e for e in rap.EXPERIMENTS if e['name'] in ('reg_only', 'all_three')]
sys.argv = ['rap', '--workers', '2']
rap.main()
"

OUT_JSON=results/ml_ablation_lxcat_missing/ablation_results.json
if [[ ! -f "$OUT_JSON" ]]; then
  echo "FAIL: $OUT_JSON not produced." >&2
  exit 3
fi

echo "==> Missing-configs ablation complete at $(date -Iseconds)"
