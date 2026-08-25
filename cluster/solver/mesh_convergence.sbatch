#!/bin/bash
#SBATCH --job-name=dtpm_mesh
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=YOUR_CPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16g
#SBATCH --time=03:00:00
#SBATCH --output=logs/mesh_%j.out
#SBATCH --error=logs/mesh_%j.err
#
# Mesh-convergence study for manuscript Sec. 4.8 (700 W / 10 mTorr / pure SF6).
# Runs all four grids 25x40 / 50x80 / 100x160 / 200x320 and reports BOTH the
# wafer-window (7.6 cm) and chamber-wall drop.
#
# IMPORTANT -- this study was deliberately run on a separate solver checkout, not the
# production tree, because the original finer-grid production run was never stored. The
# result file records that itself ("a separate solver checkout; reproduces Sec. 4.8
# unambiguously"). Consequently this job does NOT export DTPM_FIX_FLAGS, and it
# does NOT reproduce against <package>/model: pointing DTPM_JOB_ROOT at the production
# tree without the flags gives ~61.2/60.1/60.6/60.9%, and with them a third answer.
# The shipped figure-18 values 60.16/58.98/59.32/59.69 are from that checkout.

set -euo pipefail
module purge
module load cray-python/3.11.7
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# submission path of the original run; override with DTPM_JOB_ROOT to relocate

cd "${DTPM_JOB_ROOT:?export DTPM_JOB_ROOT to the solver tree (the directory holding src/ and scripts/); see cluster/README.md}"
export PYTHONPATH=src

echo "host=$(hostname)  start=$(date)"
# The driver that sweeps the grids is a figure-side script and is not part of this
# repository, which ships the study's results rather than the plotting chain. Point
# DTPM_MESH_SCRIPT at your own driver, or read the results in data/mesh_convergence/.
python3 "${DTPM_MESH_SCRIPT:?set DTPM_MESH_SCRIPT to a driver that sweeps the grids; this repository ships the results of this study in data/mesh_convergence/ rather than the driver}" \
        --grids 25x40 50x80 100x160 200x320
echo "=== RESULT TABLE ==="
cat the mesh-convergence results directory/mesh_convergence_v2.md
echo "done=$(date)"
