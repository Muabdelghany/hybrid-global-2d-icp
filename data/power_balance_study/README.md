# Constrained power balance: the sustainment test

Part of the data repository for *A self-consistent hybrid global–2D model for SF₆/Ar inductively coupled plasma etchers with neural-network surrogate acceleration*.

The study behind the sustainment result in section 5.5. It asks whether the discharge sustains
when the electron kinetics come from a two-term Boltzmann solution on the Biagi cross sections
rather than from a Maxwellian rate set, with the absorbed power fraction constrained rather than
prescribed (`--power-balance-eta`). The production flag set applies otherwise.

## The gated runs

`pb_gated/` holds the runs of record. A pass is accepted only when three conditions hold
together: the emergent absorbed fraction eta_eff = P_abs/P_rf and the electronegativity alpha are
both stationary, the inner electron-density loop met its tolerance, and the saved-field energy
budget closes to better than 5 per cent. The flags are in `pb_gated/study_flags.json` and the
solver checksum in `pb_gated/code_md5.txt`.

| case | eta_eff | power dissipated | energy residual | Te (eV) | ne (m^-3) | converged |
|---|---|---|---|---|---|---|
| Maxwellian, 10 mTorr (control) | 0.94 | 98% | 2.1% | 2.65 | 5.28e17 | **yes** |
| Boltzmann, 10 mTorr | 0.20 | 24% | 76% | 25.0 (ceiling) | 1.52e15 | no |
| Boltzmann, 3 mTorr | 0.44 | 10% | 90% | 25.0 (ceiling) | 4.60e15 | no |

The Maxwellian control converges with the budget closed, which shows the acceptance conditions
are not simply too strict to satisfy. Neither Boltzmann case reaches a physical solution: the
electron temperature pins at the 25 eV ceiling, the density falls about two orders of magnitude
below the sustained value, and 80 to 90 per cent of the coupled power cannot be dissipated.

At 3 mTorr the (eta_eff, alpha) pair does reach a stationary point, but with the energy budget 90
per cent open. A stationary interface is therefore not sufficient on its own, which is why the
energy condition is part of the test.

## The ungated rounds

`r3/`, `r4/` and `r5/` are earlier rounds of the same study in which acceptance rested on the
alpha residual alone. They are kept because they show how the outcome depends on the acceptance
criterion, and they differ from each other only in `coupling.max_outer`.

| round | max_outer | condition | outcome under the alpha-only criterion |
|---|---|---|---|
| r3 | 3  | 10 and 3 mTorr | not converged |
| r4 | 6  | 10 mTorr | not converged, eta 9.8e-2 |
| r4 | 6  | 3 mTorr  | converged, residual 1.6e-3, emergent eta 3.03e-3 |
| r5 | 12 | 10 mTorr | not converged; eta oscillates 2.3e-4 to 2.6e-1, residual never below 0.38 |

The r4 3 mTorr row is the stationary-but-unphysical point discussed above. Read these rounds
alongside the gated table rather than on their own.

`legacy_control/` is the sustained Maxwellian-rate run at the same operating point, the density
baseline the collapse is measured against.

## Units

`ne.npy` is in m^-3. The summary key `ne_avg_icp` is in cm^-3, a factor of 1e6 apart. Comparisons
here use the arrays throughout: sustained mean 2.573e17 m^-3 against 2.007e14 m^-3 for the
Boltzmann 10 mTorr case, a factor of 1282, and 1.052e15 m^-3 for the 3 mTorr case, a factor of
245. On an array-to-array basis the collapse is about two orders of magnitude.
