"""
SF6/Ar chemistry rate coefficients.

Contains the complete 54-reaction set from Lallement et al. (2009),
plus Penning ionization and Ar* quenching channels.

All rate coefficients returned in SI (m^3/s).
"""
import numpy as np

cm3 = 1e-6  # cm^3 to m^3 conversion
AMU = 1.66054e-27  # atomic mass unit in kg

M_SPECIES = {
    'S': 32.06, 'F': 19.0, 'F2': 38.0, 'SF': 51.06, 'SF2': 70.06,
    'SF3': 89.06, 'SF4': 108.06, 'SF5': 127.06, 'SF6': 146.06, 'Ar': 39.948,
}


def troe_rate(k0, kinf, Fc, M_cm3):
    """Troe fall-off rate (Ryan & Plumb 1990).

    Parameters
    ----------
    k0 : float
        Low-pressure termolecular rate constant [cm^6/s].
    kinf : float
        High-pressure bimolecular limit [cm^3/s].
    Fc : float
        Troe broadening factor.
    M_cm3 : float
        Total gas number density [cm^-3].

    Returns
    -------
    float
        Effective bimolecular rate coefficient [cm^3/s].
    """
    if M_cm3 <= 0 or k0 <= 0:
        return 0.0
    Pr = k0 * M_cm3 / kinf
    log_Pr = np.log10(max(Pr, 1e-30))
    F = Fc ** (1.0 / (1.0 + log_Pr**2))
    return k0 * M_cm3 / (1.0 + Pr) * F


def rates(Te):
    """Compute all 54+ rate coefficients given electron temperature.

    Parameters
    ----------
    Te : float
        Electron temperature [eV].

    Returns
    -------
    dict
        All rate coefficients in m^3/s (SI).
    """
    Te = max(Te, 0.3)
    k = {}

    # --- SF6 dissociation (reactions d1-d5) ---
    k['d1']  = 1.5e-7  * np.exp(-8.1 / Te)  * cm3
    k['d2']  = 9e-9    * np.exp(-13.4 / Te)  * cm3
    k['d3']  = 2.5e-8  * np.exp(-33.5 / Te)  * cm3
    k['d4']  = 2.3e-8  * np.exp(-23.9 / Te)  * cm3
    k['d5']  = 1.5e-9  * np.exp(-26.0 / Te)  * cm3

    # --- F2 dissociation and sequential SFx dissociation (d6-d11) ---
    k['d6']  = 1.2e-8  * np.exp(-5.8 / Te)   * cm3
    k['d7']  = 1.5e-7  * np.exp(-9.0 / Te)   * cm3
    k['d8']  = 6.2e-8  * np.exp(-9.0 / Te)   * cm3
    k['d9']  = 8.6e-8  * np.exp(-9.0 / Te)   * cm3
    k['d10'] = 4.5e-8  * np.exp(-9.0 / Te)   * cm3
    k['d11'] = 6.2e-8  * np.exp(-9.0 / Te)   * cm3

    # --- Excitation and elastic ---
    k['vib_SF6'] = 7.9e-8 * np.exp(-0.1 * Te + 0.002 * Te**2) * cm3
    k['el_SF6']  = 2.8e-7 * np.exp(-1.5 / Te) * cm3
    k['exc_F']   = 9.2e-9 * np.exp(-14.3 / Te) * cm3
    k['el_F']    = 1.1e-7 * np.exp(-1.93 / Te) * cm3
    k['vib_F2']  = 1.8e-10 * Te**1.72 * np.exp(-1.55 / Te) * cm3
    k['el_F2']   = 2.5e-7 * np.exp(-0.48 / Te) * cm3

    # --- Ionization from SF6 (reactions iz18-iz24) ---
    k['iz18'] = 1.2e-7  * np.exp(-18.1 / Te) * cm3
    k['iz19'] = 8.4e-9  * np.exp(-19.9 / Te) * cm3
    k['iz20'] = 3.2e-8  * np.exp(-20.7 / Te) * cm3
    k['iz21'] = 7.6e-9  * np.exp(-24.4 / Te) * cm3
    k['iz22'] = 1.2e-8  * np.exp(-26.0 / Te) * cm3
    k['iz23'] = 1.4e-8  * np.exp(-39.9 / Te) * cm3
    k['iz24'] = 1.2e-8  * np.exp(-31.7 / Te) * cm3

    # --- Ionization from fragments (iz25-iz29) ---
    k['iz25'] = 1.0e-7  * np.exp(-17.8 / Te) * cm3
    k['iz26'] = 9.4e-8  * np.exp(-22.8 / Te) * cm3
    k['iz27'] = 1.0e-7  * np.exp(-18.9 / Te) * cm3
    k['iz28'] = 1.3e-8  * np.exp(-16.5 / Te) * cm3
    k['iz29'] = 1.6e-7  * np.exp(-13.3 / Te) * cm3

    # --- Dissociative attachment (at30-at36) ---
    k['at30'] = 2.4e-10 / Te**1.49 * cm3
    k['at31'] = 2.0e-11 / Te**1.46 * cm3
    k['at32'] = 3.9e-12 * np.exp(0.45*Te - 0.04*Te**2) * cm3
    k['at33'] = 1.2e-13 * np.exp(0.70*Te - 0.05*Te**2) * cm3
    k['at34'] = 5.4e-15 * np.exp(0.77*Te - 0.05*Te**2) * cm3
    k['at35'] = 3.4e-11 * np.exp(0.46*Te - 0.04*Te**2) * cm3
    k['at36'] = 2.2e-13 * np.exp(0.71*Te - 0.05*Te**2) * cm3

    # --- Neutral recombination Troe parameters (Ryan & Plumb 1990) ---
    k['nr42_k0'] = 3.4e-23;  k['nr42_kinf'] = 1.0e-11; k['nr42_Fc'] = 0.43
    k['nr41_k0'] = 3.7e-28;  k['nr41_kinf'] = 5.0e-12; k['nr41_Fc'] = 0.46
    k['nr40_k0'] = 2.8e-26;  k['nr40_kinf'] = 2.0e-11; k['nr40_Fc'] = 0.47
    k['nr39_k0'] = 1.7e-28;  k['nr39_kinf'] = 2.0e-11; k['nr39_Fc'] = 0.56
    k['nr38_k0'] = 1.0e-30;  k['nr38_kinf'] = 2.0e-11; k['nr38_Fc'] = 0.67
    k['nr37_k0'] = 7.5e-33;  k['nr37_kinf'] = 2.0e-11; k['nr37_Fc'] = 0.73

    # --- Disproportionation ---
    k['nr45'] = 2.5e-11 * cm3
    k['nr44'] = 2.5e-11 * cm3
    k['nr43'] = 2.5e-11 * cm3

    # --- Ion-ion recombination ---
    k['rec'] = 1.5e-9 * cm3

    # --- Ar electron-impact reactions ---
    k['Ar_iz']   = 1.2e-10  * np.exp(-21.7 / Te)  * cm3
    k['Ar_exc']  = 4.2e-9   * np.exp(-8.0 / Te)   * cm3
    k['Ar_iz_m'] = 2.05e-7  * np.exp(-4.95 / Te)  * cm3
    k['Ar_q']    = 2.0e-7   * cm3
    k['Ar_el']   = max((-1.1e-8 + 3.9e-8*Te - 1.3e-8*Te**2
                        + 2e-9*Te**3 - 1.4e-10*Te**4 + 3.9e-12*Te**5) * cm3, 1e-20)

    # --- Penning ionization and Ar* quenching ---
    k['Penn_SF6']  = 2.0e-10 * cm3
    k['qnch_SF6']  = 3.0e-10 * cm3
    k['qnch_SFx']  = 1.0e-10 * cm3
    k['qnch_F2']   = 5.0e-11 * cm3
    k['qnch_F']    = 5.0e-12 * cm3

    # --- Totals ---
    k['iz_SF6_total']  = sum(k[f'iz{i}'] for i in [18, 19, 20, 21, 22, 23, 24])
    k['att_SF6_total'] = sum(k[f'at{i}'] for i in [30, 31, 32, 33, 34, 35, 36])

    # --- Tier-2 PINN overrides (no-op unless tier2_interface.refresh() was called) ---
    # When D4 (use_boltzmann_rates) is active, the m11 Picard loop has called
    # tier2_interface.refresh(E/N, x_Ar, p) after the EM solve; this updates
    # the aggregate channels iz_SF6_total, att_SF6_total and rescales d1-d5
    # to match the PINN's k_diss total while preserving Lallement branching.
    try:
        from . import tier2_interface as _tier2
        _tier2.apply_overrides(k)
    except Exception:
        # tier2_interface failing to import should never break tier-1 runs.
        pass

    return k


def compute_troe_rates(k, ng_m3):
    """Evaluate Troe fall-off neutral recombination rates at local gas density.

    Parameters
    ----------
    k : dict
        Rate coefficient dictionary from rates().
    ng_m3 : float
        Local total gas number density [m^-3].

    Returns
    -------
    dict
        Updated k with keys 'nr37' through 'nr42' added [m^3/s].
    """
    ng_cm3 = ng_m3 * 1e-6
    for idx in [42, 41, 40, 39, 38, 37]:
        key = f'nr{idx}'
        k[key] = troe_rate(k[f'{key}_k0'], k[f'{key}_kinf'], k[f'{key}_Fc'], ng_cm3) * cm3
    return k


def electron_source(k, ne, nSF6, nSF5, nSF3, nF, nS, nAr, nArm):
    """Net electron production rate [m^-3 s^-1]."""
    Riz = ne * (k['iz_SF6_total'] * nSF6
                + (k['iz25'] + k['iz26']) * nSF5
                + k['iz27'] * nSF3
                + k['iz28'] * nF
                + k['iz29'] * nS
                + k['Ar_iz'] * nAr
                + k['Ar_iz_m'] * nArm)
    R_Penn = k['Penn_SF6'] * nArm * nSF6
    Ratt = ne * k['att_SF6_total'] * nSF6
    return Riz + R_Penn - Ratt


def fluorine_source(k, ne, nSF6, nSF5, nSF4, nSF3, nSF2, nSF, nF2, nArm):
    """Net F atom volumetric production rate [m^-3 s^-1]."""
    RF = ne * nSF6 * (k['d1'] + 2*k['d2'] + 3*k['d3'] + 2*k['d4'] + 3*k['d5']
                       + k['iz18'] + 2*k['iz19'] + 3*k['iz20'] + 2*k['iz21']
                       + 3*k['iz22'] + 4*k['iz23'] + k['at31'])
    RF += ne * (nSF5 * (k['d7'] + k['iz26']) + nSF4 * k['d8']
                + nSF3 * k['d9'] + nSF2 * k['d10'] + nSF * k['d11'])
    RF += k['Penn_SF6'] * nArm * nSF6
    RF += k['qnch_SF6'] * nArm * nSF6
    RF += 2 * k['qnch_F2'] * nArm * nF2
    RF += 2 * k['d6'] * ne * nF2
    return RF


def energy_loss_density(k, ne, Te, nSF6, nSF5, nSF3, nF, nF2, nS, nAr, nArm):
    """Electron energy loss rate per unit volume [eV m^-3 s^-1].

    Multiply by e (1.6e-19 J/eV) for W/m^3.
    """
    from scipy.constants import m_e as me

    # np.maximum/np.minimum (not the builtins) so this works elementwise on
    # ndarray Te/ne/species fields as well as on scalars; for scalar input the
    # result is identical, so existing callers are unaffected.
    R_quench = ((k['Ar_iz_m'] + k['Ar_q']) * ne + 1e10)
    frac_stepwise = k['Ar_iz_m'] * ne / np.maximum(R_quench, 1e-30)
    frac_stepwise = np.minimum(frac_stepwise, 1.0)
    Ar_exc_loss = 12.0 * k['Ar_exc'] * nAr * (1.0 - frac_stepwise)

    Eloss = (
        (16*k['iz18'] + 20*k['iz19'] + 20.5*k['iz20'] + 28*k['iz21']
         + 37.5*k['iz22'] + 18*k['iz23'] + 29*k['iz24']) * nSF6
        + (9.6*k['d1'] + 12.1*k['d2'] + 16*k['d3'] + 18.6*k['d4'] + 22.7*k['d5']) * nSF6
        + 0.09 * k['vib_SF6'] * nSF6
        + k['el_SF6'] * nSF6 * 3 * me / (M_SPECIES['SF6'] * AMU) * Te
        + (11*k['iz25'] + 15*k['iz26']) * nSF5 + 5*k['d7'] * nSF5
        + 11*k['iz27'] * nSF3 + 5*k['d9'] * nSF3
        + 15*k['iz28'] * nF + 14.4*k['exc_F'] * nF
        + k['el_F'] * nF * 3 * me / (M_SPECIES['F'] * AMU) * Te
        + 3.2*k['d6'] * nF2 + 0.11*k['vib_F2'] * nF2
        + 10*k['iz29'] * nS
        + 16*k['Ar_iz'] * nAr
        + Ar_exc_loss
        + k['Ar_el'] * nAr * 3 * me / (M_SPECIES['Ar'] * AMU) * Te
        + (12 + 4.95) * k['Ar_iz_m'] * nArm
    )

    return ne * Eloss
