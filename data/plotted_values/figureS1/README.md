# Figure S1: joint identifiability of gamma_Al and f_e,bias

Part of the data repository for *A self-consistent hybrid global–2D model for SF₆/Ar inductively coupled plasma etchers with neural-network surrogate acceleration*.

Three panels, one CSV each.

**panel_a** is the best-fit ridge of the joint calibration residual: for each value of f_e,bias,
the gamma_Al that minimises the residual. Across the whole f_e,bias range the ridge moves only
between about 0.1439 and 0.1473, which is the point of the panel: the two parameters are not
strongly correlated, so gamma_Al is determined almost independently of f_e,bias.

**panel_b** is the observable that fixes f_e,bias: the ratio of wafer-centre [F] with the bias on
to the same quantity with it off, tabulated against f_e,bias for five values of gamma_Al. The
measured enhancement the curves are compared against is **1.6217**, drawn in the figure as a
horizontal reference line rather than as data.

**panel_c** is the observable that constrains gamma_Al: the error in bias-off wafer-centre [F]
against the measurement, in per cent, as a function of gamma_Al. It runs from +8.63% at
gamma_Al = 0.10 to -17.96% at 0.24, passing through zero between 0.10 and 0.14. At the adopted
gamma_Al = 0.155 the residual is -6.19%. The figure marks zero and gamma_Al = 0.155 with guide
lines; those are not data and are not tabulated here.

The adopted operating point is (gamma_Al, f_e,bias) = (0.155, 1.0). Section 6 of the paper sets
out how the two observables are weighed against each other to arrive at it; the panels here are
the evidence, not the argument.

The 60 solves these panels summarise are in `data/identifiability_grid/`.
