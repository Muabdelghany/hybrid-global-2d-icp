# Ablation recompute (not reported in the paper)

Part of the data repository for *A hybrid global–2D model for SF₆/Ar inductively coupled plasma
etchers with neural-network acceleration*.

`ml_ablation_legacy_ablation_results.json` is an additional run of the training-recipe ablation
on the legacy corpus. It is **not** the ablation the paper reports.

The paper's table 4 and figure 20 come from the two-platform ablation in the parent directory,
`ablation_results_gpu.json` and `ablation_results_workstation.json`, which were run under a
common truncated schedule with the same dataset, split and seeds so the two platforms are
comparable. This recompute was made separately and does not share that schedule, so its absolute
errors differ. It is kept because the ordering of the five configurations is the quantity the
paper draws a conclusion from, and this run is a third independent check of that ordering.

Read the parent directory's files for the reported numbers.
