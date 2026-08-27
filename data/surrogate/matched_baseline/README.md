# Matched solver-baseline records (speedup denominator)

Part of the data repository for *A self-consistent hybrid global–2D model for SF₆/Ar inductively coupled plasma etchers with neural-network surrogate acceleration*.

The manuscript speedups (Section 5.3, Table 5, abstract) are `solver_time / inference_time`,
per platform. This directory holds the **solver-time** denominator; the inference times are in
`../inference_timing*.json`.

## Workload

- **Solver**: standalone `ChemSolver` (chemistry + species transport), the per-operating-point
  cost the surrogate replaces. Run on the 50x80 production grid, **1669 active cells**, fixed
  80 Picard iterations, under-relaxation 0.12, `rate_mode=legacy`, dedicated cores.
- **Surrogate inference**: a **1780-cell** wafer-relevant batch (`../inference_timing*.json`).

The two workloads share the 50x80 grid but use **different masks** (1669 vs 1780 cells); they
are grid-matched, not identical-workload. The manuscript states both counts.

## Files

| File | Role |
|---|---|
| `workstation_M1.json` | The reported workstation baseline: 7.41 s +/- 0.39 (Apple M1 CPU, dedicated). Gives 872x (2-ch) and 55x (21-ch). |
| `cluster_cpu.json` | The reported cluster baseline: 12.33 s +/- 0.13 (node cluster-cpu-node, dedicated, tight std). Gives 1750x (2-ch) and 91x (21-ch). |
| `matched_baseline_percondition_workstation_M1.json` | Instrumented rerun: 20 per-condition times + convergence certificate. Mean 7.35 s reproduces the headline within run-to-run noise (<1%). |
| `matched_baseline_percondition_cluster_node_b.json` | Instrumented rerun on the cluster node cluster-cpu-node: 20 per-condition times + convergence certificate. See node-variation note below. |

## Convergence certificate

The benchmark records, for three spanning conditions (low/mid/high power), the wafer F
drop and the relative `nF` field change across `n_iter in {40, 80, 120}`. The largest relative
`nF` field change from iter 80 to 120 is **1.314e-3** (about 0.13%, below 1.4e-3) across the
three spanning conditions (identical on both platforms), so the fixed 80-iteration budget is
nF-stable at these conditions. This is an nF-stability check
at three conditions, not a universal convergence proof for all channels. Convergence is a
property of the iteration, not the hardware, so this certificate is platform-independent; only
the timing aggregate is per-platform.

## Node-to-node variation on the cluster

The reported cluster baseline (`cluster_cpu.json`, node cluster-cpu-node) is 12.33 s with a tight 0.13 s std.
An independent instrumented rerun landed on a different, apparently busier node (cluster-cpu-node) and
gave 13.93 s with a looser 0.64 s std -- about 13% higher. This is node-to-node/contention
spread on the shared `cpu` partition. The manuscript retains the lower, tighter cluster-cpu-node figure,
which is both the cleaner measurement and the more conservative speedup (a larger solver
time would only raise the reported 1750x). The convergence certificate is identical on cluster-cpu-node,
confirming it is platform-independent.

