# AUTHORITATIVE production job: trained the 5-member two-species ensemble AND the
# recipe ablation on the legacy-mode dataset. This is the pair the manuscript reports.
# The numbered 01/02/03 jobs in this folder are the earlier lxcat-variant study.
#
# CAVEAT, kept as submitted: the two python calls end in `|| echo FAILED-...`, which
# masks a trainer failure, and the job still prints DONE-ens. Check for FAILED-ens /
# FAILED-abl in the log, and for the ensemble directory, before trusting a run.
#!/bin/bash
#SBATCH --job-name=ens_a40
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=20:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
set -eo pipefail
# submission path of the original run; override with DTPM_JOB_ROOT to relocate
cd "${DTPM_JOB_ROOT:?export DTPM_JOB_ROOT to the solver tree (the directory holding src/ and scripts/); see cluster/README.md}"
module load pytorch-conda/2.8 2>&1 || true
[ -n "${CONDA_PREFIX:-}" ] && source "$CONDA_PREFIX/etc/profile.d/conda.sh" && conda activate base || true
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/scripts/ml:${PYTHONPATH:-}"
# legacy = the published dataset (split_manifest.json)
export ML_DATASET_MODE="${ML_DATASET_MODE:-legacy}"
export DTPM_ML_SCRIPTS="${DTPM_ML_SCRIPTS:-../ml/two_species/scripts}"   # packaged: ml/two_species/scripts
[ -x ./preflight.sh ] && ./preflight.sh || true
mkdir -p results/ml_production_ensemble
echo '#### E3 ensemble (freshly generated dataset) ####'
python "${DTPM_ML_SCRIPTS}"/train_ensemble.py 2>&1 | tail -6 || echo FAILED-ens
echo '#### ablation ####'
python "${DTPM_ML_SCRIPTS}"/ablation_study.py 2>&1 | tail -6 || echo FAILED-abl
echo DONE-ens
