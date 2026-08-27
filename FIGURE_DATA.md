# Figure data index

Part of the data repository for *A hybrid global–2D model for SF₆/Ar inductively coupled plasma etchers with neural-network acceleration*.

One row per figure in the manuscript, giving the numbers behind it and where the underlying
solver output sits. The figures themselves are in the paper; this repository holds what they
were drawn from.

| figure | subject | plotted values | underlying data |
|---|---|---|---|
| 1 | Reactor geometry | — | schematic, no data |
| 2 | 0D benchmark: n_e and T_e | data/benchmark_0d_model/ (this model) + data/benchmark_0d_reference/ (published comparison) | both already flat CSV |
| 3 | 0D benchmark: species densities | data/benchmark_0d_model/ (this model) + data/benchmark_0d_reference/ (published comparison) | both already flat CSV |
| 4 | Coupling schematic | — | drawn inline in the manuscript |
| 5 | Picard iteration flowchart | — | schematic, no data |
| 6 | Azimuthal E-field and power deposition | data/reference_case/ (E_theta_rms.npy, P_rz.npy) | field maps |
| 7 | Picard convergence history | data/plotted_values/figure07/ | raw: the `convergence_history` block of data/reference_case/summary.json (26 iterations) |
| 8 | Coil-resistance sweep | data/plotted_values/figure08/ | raw: data/coil_resistance_sweep/ |
| 9 | Plasma state, six panels | data/reference_case/ | field maps: ion_*.npy, Te.npy, nF.npy |
| 10 | Neutral density maps | data/reference_case/ | field maps: n*.npy, ne.npy |
| 11 | Multispecies radial profiles | data/plotted_values/figure11/ | raw: data/reference_case/ |
| 12 | Charged-species radial profiles | data/plotted_values/figure12/ | raw: data/reference_case/ |
| 13 | Wafer-plane [F] against measurement | data/plotted_values/figure13/ | raw: data/composition_scan/ + data/measurements/ |
| 14 | Ambipolar closure comparison | data/plotted_values/figure14/ | raw: data/ambipolar_comparison/ |
| 15 | Power scalings | data/plotted_values/figure15/ | raw: data/parameter_sweeps/power_1000W_biased/ |
| 16 | Power and pressure sweeps | data/plotted_values/figure16/ | raw: data/parameter_sweeps/ |
| 17 | Wafer-plane [F]: 2D against 0D and measurement | data/plotted_values/figure17/ | raw: data/reference_case/ + data/measurements/ |
| 18 | Mesh convergence | data/plotted_values/figure18/ | raw: data/mesh_convergence/ |
| 19 | Network architecture | — | schematic, no data |
| 20 | Surrogate recipe ablation | data/plotted_values/figure20/ | raw: data/surrogate/ablation_results_*.json |
| 21 | Surrogate predicted against true | data/plotted_values/figure21/ | raw: data/surrogate/predicted_vs_true.npz |
| 22 | Per-species surrogate error | data/plotted_values/figure22/ | raw: data/surrogate/ml21_channel_metrics.json |
| 23 | Boltzmann against Arrhenius rate coefficients | data/plotted_values/figure23/ | our computed rates; see THIRD_PARTY_DATA.md |
| 24 | Electron energy distribution | data/plotted_values/figure24/ | raw: data/boltzmann/ |
| 25 | Rate-coefficient composition | data/plotted_values/figure25/ | raw: data/boltzmann/bolsig_data.h5 |
| 26 | Sensitivity tornado | data/plotted_values/figure26/ | raw: data/sensitivity/ |
| 27 | Magnitude and drop trade-off | data/plotted_values/figure27/ | raw: data/calibration_grid/ |
| S1 | Joint identifiability grid | data/plotted_values/figureS1/ | raw: data/identifiability_grid/ |

## Notes

**Field maps.** Figures 6, 9 and 10 plot two-dimensional fields on the (r,z) mesh. Flattening
them to CSV would lose the mesh, so the `.npy` arrays in `data/reference_case/` are the digest.
The mesh is defined in `model/config/default_config.yaml`.

**CSV format.** One file per figure panel. The first column is the abscissa; each remaining
column is one plotted series, named for its legend entry, with units in the header.

**Prior closure.** Directories ending `_prior_closure` hold runs made under the earlier
local-balance electron-temperature closure with γ_Al = 0.18. No figure in the paper is drawn
from them; they are kept so the closure comparisons in the text can be repeated. The README
explains the two closures.
