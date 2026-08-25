# Figure data index

Each row gives the figure as published, the plotted values, and where the underlying solver
output sits. Figure files are in `figures/`, named by their manuscript number.

| figure | file | subject | plotted values | underlying data |
|---|---|---|---|---|
| 1 | figure01.pdf | Reactor geometry | — | schematic, no data |
| 2 | figure02.pdf | 0D benchmark: n_e and T_e | data/benchmark_0d_model/ (this model) + data/benchmark_0d_reference/ (published comparison) | both already flat CSV |
| 3 | figure03.pdf | 0D benchmark: species densities | data/benchmark_0d_model/ (this model) + data/benchmark_0d_reference/ (published comparison) | both already flat CSV |
| 4 | — | Coupling schematic | — | drawn inline in the manuscript |
| 5 | figure05.pdf | Picard iteration flowchart | — | schematic, no data |
| 6 | figure06a.pdf, figure06b.pdf | Azimuthal E-field and power deposition | data/reference_case/ (E_theta_rms.npy, P_rz.npy) | field maps |
| 7 | figure07.pdf | Picard convergence history | data/plotted_values/figure07/ | raw: data/picard_convergence/ |
| 8 | figure08.pdf | Coil-resistance sweep | data/plotted_values/figure08/ | raw: data/coil_resistance_sweep/ |
| 9 | figure09.pdf | Plasma state, six panels | data/reference_case/ | field maps: ion_*.npy, Te.npy, nF.npy |
| 10 | figure10.pdf | Neutral density maps | data/reference_case/ | field maps: n*.npy, ne.npy |
| 11 | figure11.pdf | Multispecies radial profiles | data/plotted_values/figure11/ | raw: data/reference_case/ |
| 12 | figure12.pdf | Charged-species radial profiles | data/plotted_values/figure12/ | raw: data/reference_case/ |
| 13 | figure13.pdf | Wafer-plane [F] against measurement | data/plotted_values/figure13/ | raw: data/composition_scan/ + data/measurements/ |
| 14 | figure14.pdf | Ambipolar closure comparison | data/plotted_values/figure14/ | raw: data/ambipolar_comparison/ |
| 15 | figure15.pdf | Power scalings | data/plotted_values/figure15/ | raw: data/parameter_sweeps/power_1000W_biased/ |
| 16 | figure16.pdf | Pressure sweep | data/plotted_values/figure16/ | raw: data/parameter_sweeps/ |
| 17 | figure17.pdf | Wafer-plane [F]: 2D against 0D and measurement | data/plotted_values/figure17/ | raw: data/reference_case_prior_closure/ + data/measurements/ |
| 18 | figure18.pdf | Mesh convergence | data/plotted_values/figure18/ | raw: data/mesh_convergence/ |
| 19 | figure19.pdf | Network architecture | — | schematic, no data |
| 20 | figure20.pdf | Surrogate recipe ablation | data/plotted_values/figure20/ | raw: data/surrogate/ablation_results_*.json |
| 21 | figure21.pdf | Surrogate predicted against true | data/plotted_values/figure21/ | raw: data/surrogate/predicted_vs_true.npz |
| 22 | figure22.pdf | Per-species surrogate error | data/plotted_values/figure22/ | raw: data/surrogate/ml21_channel_metrics.json |
| 23 | figure23.pdf | Boltzmann against Arrhenius rate coefficients | data/plotted_values/figure23/ | our computed rates; see THIRD_PARTY_DATA.md |
| 24 | figure24.pdf | Electron energy distribution | data/plotted_values/figure24/ | raw: data/boltzmann/ |
| 25 | figure25.pdf | Rate-coefficient composition | data/plotted_values/figure25/ | raw: data/boltzmann/bolsig_data.h5 |
| 26 | figure26.pdf | Sensitivity tornado | data/plotted_values/figure26/ | raw: data/sensitivity/ |
| 27 | figure27.pdf | Magnitude and drop trade-off | data/plotted_values/figure27/ | raw: data/calibration_grid/ |
| S1 | figureS1.pdf | Joint identifiability grid | data/plotted_values/figureS1/ | raw: data/identifiability_grid/ |

## Notes

**Field maps.** Figures 6, 9 and 10 plot two-dimensional fields on the (r,z) mesh. Flattening
them to CSV would lose the mesh, so the `.npy` arrays in `data/reference_case/` are the digest.
The mesh is defined in `model/config/default_config.yaml`.

**CSV format.** One file per figure panel. The first column is the abscissa; each remaining
column is one plotted series, named for its legend entry, with units in the header.

**Prior closure.** Figure 17's published curve comes from the earlier-closure reference case,
which is why its underlying data is `data/reference_case_prior_closure/`. The README explains
the two closures.
