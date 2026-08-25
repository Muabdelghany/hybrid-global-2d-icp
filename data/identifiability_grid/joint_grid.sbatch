#!/bin/bash
#SBATCH --job-name=joint_grid
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --partition=YOUR_CPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48g
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#
# Joint (gamma_Al, f_e_bias) grid over the four measured branches, for R1.2.
#
# Every point is run fresh under one flag string and one code fingerprint rather
# than splicing archived rays from different sweeps. Uniform provenance across the
# grid is worth the extra ten minutes, since a single point carrying a different
# flag set or a different code revision is not detectable in the assembled grid.
#
# f_e_bias is inert with the bias off, so the bias-off branches take the gamma
# axis only. 5x5x2 (bias-on) + 5x2 (bias-off) = 60 solves.
set -eo pipefail
cd "${DTPM_JOB_ROOT:?export DTPM_JOB_ROOT to the solver tree (the directory holding src/ and scripts/); see cluster/README.md}"
module load pytorch-conda/2.8 2>&1 || true
[ -n "${CONDA_PREFIX:-}" ] && source "$CONDA_PREFIX/etc/profile.d/conda.sh" && conda activate base || true
export PYTHONPATH="$PWD/src:$PWD/scripts:${PYTHONPATH:-}"
export ML_DATASET_MODE=legacy OMP_NUM_THREADS=2

B=results/joint_grid
mkdir -p "$B"

# code fingerprint, recorded before anything runs
md5sum src/dtpm/modules/m11_plasma_chemistry.py | tee "$B/code_md5.txt"
cp scripts/production_flags.json "$B/production_flags_used.json"

# the production flag string, minus the two parameters this grid sweeps
BASE="--te-closure energy_transport --per-cell-rates --depleted-att --ec-wide --ne-cap 1e20 --reconciled-rates --outer-loop --mode legacy --bias-closure f_e_bias"

one () {
  local g="$1" f="$2" b="$3" x="$4" o="$5"
  if [ "$b" = on ]; then BW="--bias --p-bias-w 200"; else BW="--no-bias"; fi
  python scripts/run_one_verification_case.py \
      --power 1000 --pressure 10 --x-ar "$x" $BW \
      --gamma-al "$g" --f-e-bias "$f" $BASE \
      --out "$B/$o" > "$B/$o.log" 2>&1 \
    && echo "OK   $o" || echo "FAIL $o"
}
export -f one; export B BASE

GAMMAS="0.10 0.14 0.155 0.18 0.24"
FES="0.20 0.50 0.70 0.85 1.00"

JOBS="$B/jobs.txt"; : > "$JOBS"
for g in $GAMMAS; do
  for f in $FES; do
    echo "one $g $f on  0.10 g${g}_f${f}_on_90"  >> "$JOBS"   # 90% SF6, bias on
    echo "one $g $f on  0.70 g${g}_f${f}_on_30"  >> "$JOBS"   # 30% SF6, bias on
  done
  # bias off: f_e_bias is inert, so the f axis collapses. Stamped f=1.00.
  echo "one $g 1.00 off 0.10 g${g}_off_90" >> "$JOBS"
  echo "one $g 1.00 off 0.70 g${g}_off_30" >> "$JOBS"
done
echo "grid points: $(wc -l < "$JOBS")"

xargs -P 16 -I{} bash -c '{}' < "$JOBS"

# ---- reference assertion: nothing in this grid is trusted until the published
#      point reproduces, to the same tolerance the other solver jobs use.
python3 - <<'PYEOF'
import json, os, sys
B = "results/joint_grid"
REF = {  # the published production point, 90% SF6, gamma=0.155, f_e=1.0
    "g0.155_f1.00_on_90":  (3.03688660961e14, 74.0088601252),
    "g0.155_off_90":       (2.18293438795e14, 74.479824071206),
}
bad = 0
for name, (nf_ref, dr_ref) in REF.items():
    p = os.path.join(B, name, "summary.json")
    if not os.path.exists(p):
        print("MISSING %s" % name); bad += 1; continue
    d = json.load(open(p))
    nf, dr = d["nF_centre_wafer_cm3"], d["F_drop_pct"]
    dn, dd = abs(nf/nf_ref - 1), abs(dr - dr_ref)
    ok = dn < 1e-6 and dd < 1e-6
    bad += (not ok)
    print("%-22s nF=%.6e (rel %.2e)  drop=%.6f (abs %.2e)  %s"
          % (name, nf, dn, dr, dd, "OK" if ok else "*** MISMATCH"))
n_ok = sum(1 for d in os.listdir(B)
           if os.path.isdir(os.path.join(B, d))
           and os.path.exists(os.path.join(B, d, "summary.json")))
n_nf = sum(1 for d in os.listdir(B)
           if os.path.isdir(os.path.join(B, d))
           and os.path.exists(os.path.join(B, d, "nF.npy")))
print("converged points: %d   with nF.npy: %d" % (n_ok, n_nf))
print("REFERENCE CHECK: %s" % ("PASS" if bad == 0 else "FAIL"))
sys.exit(1 if bad else 0)
PYEOF
