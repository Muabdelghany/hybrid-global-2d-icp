# Mesh-convergence study

Condition: 700 W / 10 mTorr / 100% SF6 (frac_Ar=0.00), beta_r=1.2.

| grid | cells | [F]_centre (m^-3) | drop@7.6cm (%) | drop@wall (%) | eta | time (s) |
|---|---|---|---|---|---|---|
| 25x40 | 1000 | 2.693e+20 | 60.2 | 74.4 | 0.899 | 55 |
| 50x80 | 4000 | 2.691e+20 | 59.0 | 74.1 | 0.950 | 161 |
| 100x160 | 16000 | 2.663e+20 | 59.3 | 74.2 | 0.895 | 697 |
| 200x320 | 64000 | 2.665e+20 | 59.7 | 74.6 | 0.889 | 3607 |

Coarsest grid 25x40 vs finest 200x320: drop@7.6cm 60.2% -> 59.7% (the cluster 0.5 pp); drop@wall 74.4% -> 74.6%.
