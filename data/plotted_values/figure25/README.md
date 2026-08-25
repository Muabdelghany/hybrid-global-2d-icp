# Figure 25: rate coefficients against reduced field

Three channels, one file each: ionization, attachment and dissociation. Each file gives the
rate coefficient against reduced field E/N for three gas compositions, computed two ways: with
a Maxwellian electron energy distribution and with a two-term Boltzmann solution.

Cells reading `nan` are values the figure suppresses: anything below 1e-28 m^3 s^-1 is numerical floor
rather than physics, and is neither drawn nor tabulated. This is why the two-term Boltzmann
ionization column is largely blank. Below roughly 60 Td the Boltzmann distribution puts almost
no electrons above the SF6 ionization threshold, so the rate coefficient falls under the floor
while the Maxwellian estimate does not. That gap is the point the figure makes.

The shaded band in the figure marks the 30 to 70 Td reference window and is not data.

The swarm table these were computed from is `data/boltzmann/bolsig_data.h5`. The cross sections
behind that table are not redistributed here; see `THIRD_PARTY_DATA.md`.
