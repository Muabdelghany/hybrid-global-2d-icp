# Plotted values, one directory per figure

Each directory is named for its manuscript figure and holds the numbers that figure draws, as
plain CSV. Nothing here needs to be run. The first column is the abscissa; every other column is
one plotted series, named for its legend entry, with units in the header. Where a figure has
several panels there is one file per panel.

These are digests of the plotted curves. The solver output they were taken from sits in the
sibling directories under `data/`; `FIGURE_DATA.md` at the repository root maps each figure to
both.

## Figures without a CSV here, and where their data is instead

**Field maps: figures 6, 9 and 10.** These plot two-dimensional fields on the (r,z) mesh.
Flattening them to CSV would lose the mesh, so the `.npy` arrays in `data/reference_case/` are
the digest. The mesh is defined in `model/config/default_config.yaml`.

**Already flat: figures 2 and 3.** Both halves of the 0D benchmark are CSV already, our solve
in `data/benchmark_0d_model/` and the published comparison in `data/benchmark_0d_reference/`, so
they are not duplicated here.

**Schematics: figures 1, 4, 5 and 19.** Reactor geometry, the coupling diagram, the iteration
flowchart and the network architecture carry no data.
