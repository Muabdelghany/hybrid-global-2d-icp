#!/usr/bin/env python3
"""Run one verification case with explicit per-field overrides.

Used to produce the bias-off counterpart of an existing bias-on dataset
case for the eta-tautology verification.  Writes the full per-case
artefact set (summary.json + all 2D .npy fields) into a chosen output
directory, exactly matching the layout under results/ml_dataset/lxcat/.

Example:
    python scripts/run_one_verification_case.py \
      --power 700 --pressure 10 --x-ar 0.3 \
      --no-bias \
      --out results/verification/P0700W_p10mT_xAr030_NObias
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "default_config.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--power", type=int, required=True,
                    help="Source RF power [W]")
    ap.add_argument("--pressure", type=float, required=True,
                    help="Gas pressure [mTorr]")
    ap.add_argument("--x-ar", type=float, required=True,
                    help="Ar fraction [0..1]")
    ap.add_argument("--bias", dest="bias", action="store_true",
                    help="Enable bias (default)")
    ap.add_argument("--no-bias", dest="bias", action="store_false",
                    help="Disable bias")
    ap.set_defaults(bias=True)
    ap.add_argument("--p-bias-w", type=int, default=200,
                    help="Bias power if enabled [W]")
    ap.add_argument("--lambda-exp", type=float, default=3.20)
    ap.add_argument("--r-coil", type=float, default=0.8)
    ap.add_argument("--gamma-al", type=float, default=0.18)
    ap.add_argument("--mode", choices=["lxcat", "legacy"], default="lxcat")
    ap.add_argument("--per-cell-rates", dest="per_cell_rates",
                    action="store_true", default=False,
                    help="Evaluate the 9-species chemistry rate coefficients "
                         "at the local Te of each cell instead of one scalar <Te>")
    ap.add_argument("--bias-closure", choices=["lambda_exp", "f_e_bias"],
                    default="lambda_exp")
    ap.add_argument("--f-e-bias", type=float, default=0.5,
                    help="Electron-heating fraction for the f_e_bias closure")
    ap.add_argument("--power-balance-eta", action="store_true",
                    help="Level-2: constrain absorbed power to dissipatable (emergent eta)")
    ap.add_argument("--bias-in-te", action="store_true",
                    help="Include bias power in the electron energy equation")
    ap.add_argument("--outer-loop", action="store_true",
                    help="Enable the emergent-eta outer 0D<->2D loop")
    ap.add_argument("--depleted-att", action="store_true",
                    help="C6: local depleted nSF6 in the attachment loss")
    ap.add_argument("--ec-wide", action="store_true",
                    help="C2: widen the 2D Ec clip to the 0D window (10,2000)")
    ap.add_argument("--ne-cap", type=float, default=1e19,
                    help="C1: 0D ne ceiling [m^-3] (default 1e19 = legacy)")
    ap.add_argument("--reconciled-rates", action="store_true",
                    help="D8: sf6_chemistry uses the canonical sf6_rates iz27/28/29")
    ap.add_argument("--te-closure",
                    choices=["local_balance", "energy_transport"],
                    default="local_balance",
                    help="Te closure: legacy per-cell algebraic balance, "
                         "or the conduction+convection energy PDE")
    ap.add_argument("--out", required=True,
                    help="Output dir (will be created)")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    from run_parameter_sweeps import run_simulation, save_sweep_point

    overrides = {
        "circuit.source_power": args.power,
        "circuit.R_coil": args.r_coil,
        "operating.pressure_mTorr": args.pressure,
        "operating.frac_Ar": args.x_ar,
        "bias.enabled": args.bias,
        "bias.P_bias_W": args.p_bias_w if args.bias else 0,
        "bias.lambda_exp": args.lambda_exp,
        "wall_chemistry.gamma_Al": args.gamma_al,
        "chemistry.use_boltzmann_rates": (args.mode == "lxcat"),
        "coupling.te_closure": args.te_closure,
        "coupling.outer_loop": bool(args.outer_loop),
        "coupling.bias_in_te": bool(getattr(args,'bias_in_te',False)),
        "coupling.power_balance_eta": bool(getattr(args, 'power_balance_eta', False)),
        "bias.closure": args.bias_closure,
        "bias.f_e_bias": float(args.f_e_bias),
        "chemistry.depleted_attachment": bool(args.depleted_att),
        "chemistry.ec_clip_wide": bool(args.ec_wide),
        "chemistry.ne_cap_0D": float(args.ne_cap),
        "chemistry.reconciled_rates": bool(args.reconciled_rates),
        "chemistry.per_cell_rates": args.per_cell_rates,
    }
    tag = (f"P{args.power:04d}W_p{int(args.pressure):02d}mT_"
           f"xAr{int(round(args.x_ar*100)):03d}"
           f"_{'BIAS' if args.bias else 'NObias'}")
    s_te_closure = args.te_closure

    print(f"==> Running verification case {tag}")
    print(f"    overrides: {overrides}", flush=True)
    print(f"    out_dir: {out_dir}", flush=True)

    t0 = time.time()
    state, mesh, inside, rgeom, config, r0D = run_simulation(overrides,
                                                           CONFIG_PATH)
    elapsed = time.time() - t0
    print(f"    converged in {elapsed:.1f}s", flush=True)

    s = save_sweep_point(state, mesh, inside, r0D, config, out_dir, tag)
    s["case_id"] = tag
    s["te_closure"] = s_te_closure
    s["power_balance_eta_on"] = bool(state.get("power_balance_eta_on", False))
    s["eta_emergent_power_balance"] = state.get("eta_emergent_power_balance")
    s["sink_frac_final"] = state.get("sink_frac_final")
    s["outer_loop_on"] = bool(state.get("outer_loop_on", False))
    s["outer_iters"] = int(state.get("outer_iters", 0) or 0)
    s["outer_converged"] = bool(state.get("outer_converged", True))
    s["outer_history"] = state.get("outer_history", [])
    s["convergence_history"] = state.get("convergence_history", [])
    s["per_cell_rates"] = bool(args.per_cell_rates)
    _ti = state.get("te_info")
    if _ti:
        s["te_sink_frac"] = _ti.get("sink_frac")
        s["te_volavg"] = _ti.get("Te_volavg")
        s["te_frac_at_clamp"] = _ti.get("frac_at_clamp")
        s["te_converged"] = _ti.get("converged")
        s["te_newton_iters"] = _ti.get("newton_iters")
        s["te_P_source_W"] = _ti.get("P_source_W")
        s["te_P_volloss_W"] = _ti.get("P_volloss_W")
        s["te_P_wall_W"] = _ti.get("P_wall_W")
    s["mode"] = args.mode
    s["P_rf_W"] = int(args.power)
    s["p_mTorr"] = float(args.pressure)
    s["x_Ar"] = float(args.x_ar)
    s["bias_enabled"] = bool(args.bias)
    s["P_bias_W"] = int(args.p_bias_w if args.bias else 0)
    s["gamma_Al"] = float(args.gamma_al)
    s["lambda_exp"] = float(args.lambda_exp)
    s["R_coil"] = float(args.r_coil)
    s["nF_centre_wafer_cm3"] = float(state["nF"][0, 0]) * 1e-6
    s["eta_computed"] = float(state.get("eta_computed", 0.0))
    s["I_peak_final"] = float(state.get("I_peak_final", 0.0))
    s["R_plasma_final"] = float(state.get("R_plasma_final", 0.0))
    s["V_peak_final"] = float(state.get("V_peak_final", 0.0))
    s["P_abs_final"] = float(state.get(
        "P_abs_final", state.get("P_abs", 0.0)))
    s["converged"] = True
    s["elapsed_sec"] = elapsed

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(s, f, indent=2)
    print(f"==> Wrote {summary_path}")
    print(f"    eta={s['eta_computed']:.4f}, "
          f"P_abs={s['P_abs_final']:.1f}W, "
          f"I_peak={s['I_peak_final']:.3f}A, "
          f"R_plasma={s['R_plasma_final']:.3f}Ohm", flush=True)


if __name__ == "__main__":
    main()
