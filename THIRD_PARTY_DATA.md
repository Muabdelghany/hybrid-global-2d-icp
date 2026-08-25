# Material that originates elsewhere

Most of this repository is our own output and is covered by the licences in `LICENSE` and
`LICENSE-DATA`. The items below are **not**: they originate elsewhere and remain under the terms
of their own sources, and nothing in `LICENSE-DATA` relicenses them. Each entry names the source
and the attribution its providers ask for. If you use any of them, cite the original source
rather than this repository.

## Electron-impact cross sections

The Boltzmann calculations behind figures 23, 24 and 25 use SF₆ electron-impact cross sections
from the Biagi database, obtained through the LXCat project. Those cross sections remain the
property of their contributors and are **not redistributed here**. Download them yourself from
[www.lxcat.net](https://www.lxcat.net), selecting the Biagi database for SF₆.

The reference format the database asks for is:

> Biagi database, www.lxcat.net, retrieved on April 16, 2026.

The cross sections were transcribed by S. F. Biagi from MAGBOLTZ; the database header records the
version number for each species and asks that it be included in any reference.

What this repository contains instead is our own computed output: the rate coefficients derived
from those cross sections, in `data/plotted_values/figure23/` and `data/boltzmann/`. Those are
our results and are covered by `LICENSE-DATA`. Regenerating them from scratch needs the cross
sections above.

## Boltzmann solver

The swarm table in `data/boltzmann/bolsig_data.h5` holds our own two-term Boltzmann solutions
over an (E/N, x_Ar) grid. The solver used is BOLSIG+, by G. J. M. Hagelaar and L. C. Pitchford,
*Plasma Sources Science and Technology* **14** 722 (2005). Cite that paper for the solver. The
table itself is derived output produced by this project.

## Digitized 0D benchmark traces

`data/benchmark_0d_reference/` holds curves digitized from the published figures of the prior
0D study the model is benchmarked against in figures 2 and 3. The manuscript's reference list
carries the full citation. Only the curves needed for the comparison were digitized, at the
resolution the comparison needs, and they are not a substitute for the original work. Our own
solve of the same conditions is in `data/benchmark_0d_model/` and is covered by `LICENSE-DATA`.

## Experimental comparison data

`data/measurements/` holds traces digitized from the published figures of the cited dissertation,
for benchmark comparison only. The manuscript's reference list carries the full citation.
Digitized traces are reproduced at the resolution the comparison needs and are not a substitute
for the original measurements.

## Arrhenius rate set

The Arrhenius rate coefficients compared against the Boltzmann set in figure 23 are taken from
the reaction set cited in the manuscript. See the manuscript's reference list.

## Reactor geometry

The geometry in `model/config/*.yaml` describes an industrial ICP etcher of the class the
manuscript models, under the neutral reference name **IPI ICP**. The dimensions are those the
manuscript itself reports.
