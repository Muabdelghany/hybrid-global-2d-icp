#!/bin/bash
#SBATCH --job-name=pbgated
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=YOUR_CPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=03:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
# Gated power-balance run: the outer loop certifies
# on the emergent absorbed fraction eta_eff=P_abs/P_rf (fed to solve_0D AND
# gated), requiring inner ne-convergence + a saved-field energy balance.
#  leg_gated  = Maxwellian POSITIVE CONTROL (must converge)
#  lx_*_gated = Boltzmann honest verdict at 10 and 3 mTorr
set -eo pipefail
# submission path of the original run; override with DTPM_JOB_ROOT to relocate
cd "${DTPM_JOB_ROOT:?export DTPM_JOB_ROOT to the solver tree (the directory holding src/ and scripts/); see cluster/README.md}"
module load pytorch-conda/2.8 2>&1 || true
[ -n "${CONDA_PREFIX:-}" ] && source "$CONDA_PREFIX/etc/profile.d/conda.sh" && conda activate base || true
export PYTHONPATH="$PWD/src:$PWD/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2

B=results/pb_gated
mkdir -p "$B"
# provenance: code id + requested config, immutable
md5sum src/dtpm/modules/m11_plasma_chemistry.py > "$B/code_md5.txt"
FLAGS_JSON="{\"coupling.max_picard_iter\": 120, \"coupling.power_balance_relax\": 0.5, \"coupling.max_outer\": 8, \"coupling.pb_eta_gated\": true, \"coupling.pb_energy_tol\": 0.05}"
echo "$FLAGS_JSON" > "$B/study_flags.json"
export DTPM_FIX_FLAGS="$PWD/$B/study_flags.json"

FIX="--te-closure energy_transport --per-cell-rates --depleted-att --ec-wide --ne-cap 1e20 --reconciled-rates --outer-loop --gamma-al 0.155 --bias-closure f_e_bias --f-e-bias 1.0 --power-balance-eta"

run () {
  local tag="$1" mode="$2" p="$3"
  ( export ML_DATASET_MODE="$mode" DTPM_FIX_FLAGS="$PWD/$B/study_flags.json"
    python scripts/run_one_verification_case.py --mode "$mode" $FIX \
      --power 1000 --pressure "$p" --x-ar 0.30 --no-bias \
      --out "$B/$tag" > "$B/$tag.log" 2>&1 \
      && echo "OK $tag" || echo "FAIL $tag" )
}
export -f run; export B FIX

printf "%s\n" \
  "run leg_gated    legacy 10" \
  "run lx_ref_gated lxcat  10" \
  "run lx_lowp_gated lxcat  3" \
  | xargs -P 3 -I{} bash -c "{}"

echo "=== SUMMARY ==="
python3 - <<PYEOF
import json, os
B="results/pb_gated"
for tag in ("leg_gated","lx_ref_gated","lx_lowp_gated"):
    p=os.path.join(B,tag,"summary.json")
    if not os.path.exists(p):
        print(tag,"MISSING"); continue
    j=json.load(open(p))
    print("%-14s conv=%s inner_conv=%s eta_eff=%.4e Psrc=%.2f Ploss=%.2f Te=%.2f" % (
      tag, j.get("outer_converged"), j.get("inner_converged_final"),
      (j.get("eta_eff_final") or 0), (j.get("te_P_source_W") or 0),
      (j.get("te_P_volloss_W") or 0)+(j.get("te_P_wall_W") or 0), j.get("Te_avg") or 0))
PYEOF
echo "=== PBGATED DONE ==="
