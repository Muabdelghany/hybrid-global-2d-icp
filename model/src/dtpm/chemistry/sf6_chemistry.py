"""
SF6/Ar Chemistry Module for 2D Transport — Full Lallement Model
================================================================
Contains all rate coefficients, source/sink terms, diffusion coefficients,
and wall recombination probabilities from the validated 0D Lallement model.

Species solved (9 neutrals):
  SF6, SF5, SF4, SF3, SF2, SF, S, F, F2

Charged species computed from quasi-neutrality:
  ne, n+, n-, alpha, SF5+, SF3+, SF4+, F+, F-, SF6-, SF5-, SF4-

Ar* metastable in local steady-state.

Ported from Stage 10 simulation package sf6_chemistry.py.
"""

import numpy as np
from scipy.constants import k as kB, e as eC, pi, atomic_mass as AMU

# Molar masses (amu)
M = {'S': 32.06, 'F': 19.0, 'F2': 38.0, 'SF': 51.06, 'SF2': 70.06,
     'SF3': 89.06, 'SF4': 108.06, 'SF5': 127.06, 'SF6': 146.06, 'Ar': 39.948}

SPECIES = ['SF6', 'SF5', 'SF4', 'SF3', 'SF2', 'SF', 'S', 'F', 'F2']


def compute_diffusion_coefficients(Tgas, p_mTorr):
    """Compute diffusion coefficient D for each neutral species via Chapman-Enskog theory."""
    p_Pa = p_mTorr * 0.1333
    ng = p_Pa / (kB * Tgas)
    sigma = 3.0e-10  # m, Lennard-Jones collision diameter (F-SF6 combining rule)
    D = {}
    for sp in SPECIES:
        m_kg = M[sp] * AMU
        v_th = np.sqrt(8 * kB * Tgas / (pi * m_kg))
        mfp = 1.0 / (ng * pi * sigma**2)
        D[sp] = v_th * mfp / 3.0
    return D


def compute_thermal_speeds(Tgas):
    """Thermal speeds for Robin BC computation."""
    v_th = {}
    for sp in SPECIES:
        m_kg = M[sp] * AMU
        v_th[sp] = np.sqrt(8 * kB * Tgas / (pi * m_kg))
    return v_th


def troe_rate(k0, kinf, Fc, M_cm3):
    """Troe fall-off rate coefficient (Ryan & Plumb 1990)."""
    if M_cm3 <= 0 or k0 <= 0:
        return 0.0
    Pr = k0 * M_cm3 / kinf
    log_Pr = np.log10(max(Pr, 1e-30))
    F = Fc ** (1.0 / (1.0 + log_Pr**2))
    return k0 * M_cm3 / (1.0 + Pr) * F


def compute_rates(Te, ng_cm3, frac_Ar=0.0, reconciled=False):
    """Compute all 54+ rate coefficients at given Te (eV) and gas density.

    Parameters
    ----------
    Te : float
        Electron temperature [eV].
    ng_cm3 : float
        Total gas number density [cm^-3].
    frac_Ar : float
        Argon fraction in gas mixture.

    Returns
    -------
    dict : All rate coefficients in m^3/s (SI).
    """
    Te = max(Te, 0.3)
    cm3 = 1e-6  # cm^3 -> m^3

    k = {}

    # SF6 dissociation (G1-G5)
    k['d1']  = 1.5e-7  * np.exp(-8.1 / Te) * cm3
    k['d2']  = 9e-9    * np.exp(-13.4 / Te) * cm3
    k['d3']  = 2.5e-8  * np.exp(-33.5 / Te) * cm3
    k['d4']  = 2.3e-8  * np.exp(-23.9 / Te) * cm3
    k['d5']  = 1.5e-9  * np.exp(-26.0 / Te) * cm3

    # F2 dissociation and sequential SFx (G6-G11)
    k['d6']  = 1.2e-8  * np.exp(-5.8 / Te) * cm3
    k['d7']  = 1.5e-7  * np.exp(-9.0 / Te) * cm3
    k['d8']  = 6.2e-8  * np.exp(-9.0 / Te) * cm3
    k['d9']  = 8.6e-8  * np.exp(-9.0 / Te) * cm3
    k['d10'] = 4.5e-8  * np.exp(-9.0 / Te) * cm3
    k['d11'] = 6.2e-8  * np.exp(-9.0 / Te) * cm3

    # Dissociative ionization from SF6 (iz18-iz24)
    k['iz18'] = 1.2e-7  * np.exp(-18.1 / Te) * cm3
    k['iz19'] = 8.4e-9  * np.exp(-19.9 / Te) * cm3
    k['iz20'] = 3.2e-8  * np.exp(-20.7 / Te) * cm3
    k['iz21'] = 7.6e-9  * np.exp(-24.4 / Te) * cm3
    k['iz22'] = 1.2e-8  * np.exp(-26.0 / Te) * cm3
    k['iz23'] = 1.4e-8  * np.exp(-39.9 / Te) * cm3
    k['iz24'] = 1.2e-8  * np.exp(-31.7 / Te) * cm3

    # Fragment ionization (iz25-iz29)
    k['iz25'] = 1.0e-7  * np.exp(-17.8 / Te) * cm3
    k['iz26'] = 9.4e-8  * np.exp(-22.8 / Te) * cm3
    if reconciled:
        # D8 (chemistry.reconciled_rates): use the sf6_rates.py values — the two
        # modules are the same Lallement transcription (iz18-26 identical) and
        # diverge only here; sf6_rates is canonical (its 0D is the validated
        # model, and e.g. iz29 = S->S+ with IP 10.36 eV: activation 13.3 eV is
        # physical, the 20.0 eV below is ~2x the IP — a transcription error).
        k['iz27'] = 1.0e-7  * np.exp(-18.9 / Te) * cm3
        k['iz28'] = 1.3e-8  * np.exp(-16.5 / Te) * cm3
        k['iz29'] = 1.6e-7  * np.exp(-13.3 / Te) * cm3
    else:
        k['iz27'] = 1.0e-7  * np.exp(-17.0 / Te) * cm3
        k['iz28'] = 1.6e-7  * np.exp(-18.6 / Te) * cm3
        k['iz29'] = 1.0e-7  * np.exp(-20.0 / Te) * cm3

    # Dissociative attachment (at30-at31)
    k['at30'] = 2.4e-10 / Te**1.49 * cm3
    k['at31'] = 2.0e-11 / Te**1.46 * cm3
    k['at_total'] = k['at30'] + k['at31']

    # Neutral recombination (Troe fall-off)
    k['nr42'] = troe_rate(3.4e-23, 1.0e-11, 0.43, ng_cm3) * cm3
    k['nr41'] = troe_rate(3.7e-28, 5.0e-12, 0.46, ng_cm3) * cm3
    k['nr40'] = troe_rate(2.8e-26, 2.0e-11, 0.47, ng_cm3) * cm3
    k['nr39'] = troe_rate(1.7e-28, 2.0e-11, 0.56, ng_cm3) * cm3
    k['nr38'] = troe_rate(1.0e-30, 2.0e-11, 0.67, ng_cm3) * cm3
    k['nr37'] = troe_rate(7.5e-33, 2.0e-11, 0.73, ng_cm3) * cm3
    k['nr45'] = 2.5e-11 * cm3

    # Penning / Ar* chemistry
    k['Penn_SF6']  = 2.0e-10 * cm3
    k['qnch_SF6']  = 3.0e-10 * cm3
    k['qnch_SFx']  = 1.0e-10 * cm3
    k['qnch_F2']   = 5.0e-11 * cm3
    k['qnch_F']    = 5.0e-12 * cm3
    k['Ar_exc']    = 4.2e-9  * np.exp(-8.0 / Te) * cm3
    k['Ar_iz_m']   = 2.05e-7 * np.exp(-4.95 / Te) * cm3
    k['Ar_q']      = 2.0e-7  * cm3

    # Totals
    k['iz_SF6_total'] = sum(k[f'iz{i}'] for i in [18, 19, 20, 21, 22, 23, 24])
    k['k_e_SF6'] = (k['d1'] + k['d2'] + k['d3'] + k['d4'] + k['d5']
                    + k['iz_SF6_total'] + k['at_total'])

    return k


# Wall recombination probabilities (Kokkoris 2009) per species per material
WALL_GAMMA = {
    'quartz': {
        'SF6': 0, 'SF5': 0.0005, 'SF4': 0.0005, 'SF3': 0.0005,
        'SF2': 0.0005, 'SF': 0.0005, 'S': 0.001, 'F': 0.001, 'F2': 0.0001,
    },
    'aluminium': {
        'SF6': 0, 'SF5': 0.008, 'SF4': 0.008, 'SF3': 0.008,
        'SF2': 0.008, 'SF': 0.008, 'S': 0.01, 'F': 0.035, 'F2': 0.001,
    },
    'silicon': {
        'SF6': 0, 'SF5': 0.001, 'SF4': 0.001, 'SF3': 0.001,
        'SF2': 0.001, 'SF': 0.001, 'S': 0.001, 'F': 0.025, 'F2': 0.0001,
    },
    'window': {
        'SF6': 0, 'SF5': 0.0003, 'SF4': 0.0003, 'SF3': 0.0003,
        'SF2': 0.0003, 'SF': 0.0003, 'S': 0.001, 'F': 0.001, 'F2': 0.0001,
    },
}
