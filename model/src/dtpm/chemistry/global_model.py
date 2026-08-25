"""
SF6/Ar 0D Global Model with Penning Ionization.

Solves the 0D particle and power balance to obtain Te, ne, alpha,
and all 9 neutral species densities. Used as the initial guess for
the Picard-coupled 2D simulation, and as a standalone reference.

Adapted from Stage 10 simulation package global_model.py.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.constants import e as eC, k as kB, m_e, pi

from .sf6_rates import rates, troe_rate, compute_troe_rates, M_SPECIES, AMU, cm3

MTORR_TO_PA = 0.133322


class Reactor:
    """Simple cylindrical reactor geometry for the 0D model."""

    def __init__(self, R=0.180, L=0.175):
        self.R, self.L = R, L
        self.V = pi * R**2 * L
        self.A = 2 * pi * R**2 + 2 * pi * R * L
        self.Lambda = 1.0 / np.sqrt((pi / L)**2 + (2.405 / R)**2)


def kw_neutral(gamma, Ma, Tg, ng, reactor):
    """Neutral wall loss frequency [s^-1] (diffusion + surface reaction)."""
    if gamma <= 0:
        return 0.0
    Mk = Ma * AMU
    v = np.sqrt(8 * kB * Tg / (pi * Mk))
    lam = 1.0 / (ng * 4e-19) if ng > 0 else 0.1
    De = 1.0 / (1.0 / max(kB * Tg * lam / (Mk * v), 1e-30)
                + 1.0 / max(v * lam / 3, 1e-30))
    return 1.0 / (reactor.Lambda**2 / De + 2 * reactor.V * (2 - gamma) / (reactor.A * v * gamma))


def kw_ion(Te, Ma, alpha, Tn, reactor, ng):
    """Ion wall loss frequency [s^-1] with EN correction (Lee & Lieberman)."""
    Mk = Ma * AMU
    gam = Te / max(Tn, 0.01)
    uB = np.sqrt(eC * Te * (1 + alpha) / (Mk * (1 + gam * alpha)))
    sig = 5e-19
    lam = 1.0 / (ng * sig) if ng > 0 else 0.1
    Da = eC * Te / (Mk * np.sqrt(eC * Tn / Mk) / lam)
    EN = (1 + 3 * alpha / gam) / (1 + alpha)
    hL = EN * 0.86 / np.sqrt(3 + reactor.L / (2 * lam) + (0.86 * reactor.L * uB / (pi * gam * Da))**2)
    hR = EN * 0.8 / np.sqrt(4 + reactor.R / lam + (0.8 * reactor.R * uB / (2.405 * 0.5191 * 2.405 * gam * Da))**2)
    hL = np.clip(hL, 1e-5, 1)
    hR = np.clip(hR, 1e-5, 1)
    return 2 * uB * (hL / reactor.L + hR / reactor.R)


def solve_0D(P_rf=700, p_mTorr=10, frac_Ar=0.0, Q_sccm=100, Tgas=313,
             T_neg=0.3, gamma_F=0.01, beta_SFx=0.02, eta=0.43,
             R_icp=0.038, L_icp=0.1815,
             init_Te=None, init_ne=None, init_alpha=None, init_ns=None,
             ne_cap=1e19, tol_Te=5e-5, tol_ne=5e-4, tol_alpha=1e-3):
    """Solve the 0D global model for Te, ne, alpha, and species densities.

    Parameters
    ----------
    P_rf : float
        RF power [W].
    p_mTorr : float
        Pressure [mTorr].
    frac_Ar : float
        Argon fraction in gas mixture.
    Q_sccm : float
        Total gas flow [sccm].
    Tgas : float
        Gas temperature [K].
    eta : float
        Power coupling efficiency (initial guess; replaced by FDTD in Phase 1).
    R_icp, L_icp : float
        ICP source dimensions for the reactor geometry [m].

    Returns
    -------
    dict with Te, ne, alpha, species densities, convergence info.
    """
    reactor = Reactor(R=R_icp, L=L_icp)
    P_abs = P_rf * eta
    p_Pa = p_mTorr * MTORR_TO_PA
    ng0 = p_Pa / (kB * Tgas)
    nSF6_0 = ng0 * (1 - frac_Ar)
    nAr0 = ng0 * frac_Ar
    Q_tp = Q_sccm * 1e-6 / 60 * 1.01325e5 * (Tgas / 273.15)
    tau = p_Pa * reactor.V / Q_tp if Q_tp > 0 else 1e10

    Te = init_Te if init_Te else 3.0
    ne = init_ne if init_ne else 5e15
    alpha = init_alpha if init_alpha else 5.0

    if init_ns:
        nSF6 = init_ns.get('SF6', nSF6_0 * 0.4)
        nSF5 = init_ns.get('SF5', nSF6_0 * 0.02)
        nSF4 = init_ns.get('SF4', nSF6_0 * 0.01)
        nSF3 = init_ns.get('SF3', nSF6_0 * 0.005)
        nSF2 = init_ns.get('SF2', nSF6_0 * 0.001)
        nSF  = init_ns.get('SF',  nSF6_0 * 0.0005)
        nS   = init_ns.get('S',   nSF6_0 * 0.0001)
        nF   = init_ns.get('F',   nSF6_0 * 0.3)
        nF2  = init_ns.get('F2',  nSF6_0 * 0.01)
    else:
        nSF6 = nSF6_0 * 0.4
        nSF5 = nSF6_0 * 0.02
        nSF4 = nSF6_0 * 0.01
        nSF3 = nSF6_0 * 0.005
        nSF2 = nSF6_0 * 0.001
        nSF  = nSF6_0 * 0.0005
        nS   = nSF6_0 * 0.0001
        nF   = nSF6_0 * 0.3
        nF2  = nSF6_0 * 0.01

    M = M_SPECIES
    converged = False

    for outer in range(500):
        Te_old, ne_old, al_old = Te, ne, alpha
        k = rates(Te)
        ng = ng0

        # Troe recombination rates
        k = compute_troe_rates(k, ng)

        kwF = kw_neutral(gamma_F, M['F'], Tgas, ng, reactor)
        kwS = kw_neutral(beta_SFx, M['SF5'], Tgas, ng, reactor)
        k_e = (k['d1'] + k['d2'] + k['d3'] + k['d4'] + k['d5']
               + k['iz_SF6_total'] + k['att_SF6_total'])

        # Ar* metastable balance
        k_Arm_wall = kw_neutral(1.0, M['Ar'], Tgas, ng, reactor)
        R_quench_heavy = (k['Penn_SF6'] * nSF6 + k['qnch_SF6'] * nSF6
                         + k['qnch_SFx'] * (nSF5 + nSF4 + nSF3 + nSF2 + nSF)
                         + k['qnch_F2'] * nF2 + k['qnch_F'] * nF)
        nArm = (k['Ar_exc'] * ne * nAr0
                / ((k['Ar_iz_m'] + k['Ar_q']) * ne + k_Arm_wall + R_quench_heavy + 1e-30))
        R_Penning = k['Penn_SF6'] * nArm * nSF6

        # Neutral sub-iteration
        for sub in range(20):
            k_SF6_loss = k_e * ne + k['Penn_SF6'] * nArm + k['qnch_SF6'] * nArm + 1 / tau
            s6 = nSF6_0 / tau + k['nr42'] * nSF5 * nF + k['nr45'] * nSF5**2
            nSF6_n = s6 / (k_SF6_loss + 1e-30)
            nSF6_n = np.clip(nSF6_n, 1e10, nSF6_0)
            nSF6 = 0.3 * nSF6 + 0.7 * nSF6_n

            nSF5_prod = k['d1'] * ne * nSF6 + k['nr41'] * nSF4 * nF + k['qnch_SF6'] * nArm * nSF6
            nSF5 = nSF5_prod / ((k['d7'] + k['iz25'] + k['iz26']) * ne + k['nr42'] * nF
                                + 2 * k['nr45'] * nSF5 + kwS + k['qnch_SFx'] * nArm + 1 / tau + 1e-30)

            nSF4 = ((k['d2'] * ne * nSF6 + k['d7'] * ne * nSF5 + k['nr45'] * nSF5**2)
                    / (k['d8'] * ne + k['nr41'] * nF + kwS + k['qnch_SFx'] * nArm + 1 / tau + 1e-30))
            nSF3 = ((k['d3'] * ne * nSF6 + k['d8'] * ne * nSF4)
                    / ((k['d9'] + k['iz27']) * ne + k['nr40'] * nF + kwS + k['qnch_SFx'] * nArm + 1 / tau + 1e-30))
            nSF2 = ((k['d4'] * ne * nSF6 + k['d9'] * ne * nSF3)
                    / (k['d10'] * ne + k['nr39'] * nF + 1 / tau + 1e-30))
            nSF  = ((k['d5'] * ne * nSF6 + k['d10'] * ne * nSF2)
                    / (k['d11'] * ne + k['nr38'] * nF + 1 / tau + 1e-30))
            nS   = (k['d11'] * ne * nSF
                    / (k['iz29'] * ne + k['nr37'] * nF + 1 / tau + 1e-30))

            RF = (ne * nSF6 * (k['d1'] + 2*k['d2'] + 3*k['d3'] + 2*k['d4'] + 3*k['d5']
                               + k['iz18'] + 2*k['iz19'] + 3*k['iz20'] + 2*k['iz21']
                               + 3*k['iz22'] + 4*k['iz23'] + k['at31'])
                  + ne * (nSF5 * (k['d7'] + k['iz26']) + nSF4 * k['d8']
                          + nSF3 * k['d9'] + nSF2 * k['d10'] + nSF * k['d11'])
                  + k['Penn_SF6'] * nArm * nSF6 + k['qnch_SF6'] * nArm * nSF6
                  + 2 * k['qnch_F2'] * nArm * nF2)

            nF2 = ((ne * nSF6 * (k['d4'] + k['d5'] + k['iz21'] + k['iz22'] + k['iz23'])
                    + 0.5 * kwF * nF)
                   / (k['d6'] * ne + k['qnch_F2'] * nArm + 1 / tau + 1e-30))
            RF += 2 * k['d6'] * ne * nF2
            kFL = (kwF + 1 / tau + k['iz28'] * ne + k['nr42'] * nSF5 + k['nr41'] * nSF4
                   + k['nr40'] * nSF3 + k['nr39'] * nSF2 + k['nr38'] * nSF + k['nr37'] * nS)
            nF_n = RF / (kFL + 1e-30)
            nF = 0.3 * nF + 0.7 * min(nF_n, 6 * nSF6_0)

            # Update Ar* with new neutrals
            R_quench_heavy = (k['Penn_SF6'] * nSF6 + k['qnch_SF6'] * nSF6
                             + k['qnch_SFx'] * (nSF5 + nSF4 + nSF3 + nSF2 + nSF)
                             + k['qnch_F2'] * nF2 + k['qnch_F'] * nF)
            nArm = (k['Ar_exc'] * ne * nAr0
                    / ((k['Ar_iz_m'] + k['Ar_q']) * ne + k_Arm_wall + R_quench_heavy + 1e-30))
            R_Penning = k['Penn_SF6'] * nArm * nSF6

        ng = max(nSF6 + nSF5 + nSF4 + nSF3 + nSF2 + nSF + nS + nF + nF2 + nAr0, ng0 * 0.1)

        # Ionization / attachment
        Riz_electron = (k['iz_SF6_total'] * nSF6 + (k['iz25'] + k['iz26']) * nSF5
                       + k['iz27'] * nSF3 + k['iz28'] * nF + k['iz29'] * nS
                       + k['Ar_iz'] * nAr0 + k['Ar_iz_m'] * nArm)
        R_Penn_vol = k['Penn_SF6'] * nArm * nSF6
        Ratt = k['att_SF6_total'] * nSF6
        kwi = kw_ion(Te, M['SF5'], alpha, T_neg, reactor, ng)

        # Alpha
        rhs_q = Ratt / (k['rec'] * ne) if ne > 0 else 0
        alpha_new = (-1 + np.sqrt(1 + 4 * rhs_q)) / 2 if rhs_q > 0 else 0

        # Collisional energy loss
        R_quench_total = (k['Ar_iz_m'] + k['Ar_q']) * ne + k_Arm_wall + R_quench_heavy
        frac_stepwise = k['Ar_iz_m'] * ne / max(R_quench_total, 1e-30)
        frac_stepwise = min(frac_stepwise, 1.0)
        Ar_exc_loss = 12 * k['Ar_exc'] * nAr0 * (1.0 - frac_stepwise)

        Eloss = ((16*k['iz18'] + 20*k['iz19'] + 20.5*k['iz20'] + 28*k['iz21']
                  + 37.5*k['iz22'] + 18*k['iz23'] + 29*k['iz24']) * nSF6
                 + (9.6*k['d1'] + 12.1*k['d2'] + 16*k['d3'] + 18.6*k['d4'] + 22.7*k['d5']) * nSF6
                 + 0.09 * k['vib_SF6'] * nSF6
                 + k['el_SF6'] * nSF6 * 3 * m_e / (M['SF6'] * AMU) * Te
                 + (11*k['iz25'] + 15*k['iz26']) * nSF5 + 5*k['d7'] * nSF5
                 + 11*k['iz27'] * nSF3 + 5*k['d9'] * nSF3
                 + 15*k['iz28'] * nF + 14.4*k['exc_F'] * nF
                 + k['el_F'] * nF * 3 * m_e / (M['F'] * AMU) * Te
                 + 3.2*k['d6'] * nF2 + 0.11*k['vib_F2'] * nF2
                 + 10*k['iz29'] * nS + 16*k['Ar_iz'] * nAr0 + Ar_exc_loss
                 + k['Ar_el'] * nAr0 * 3 * m_e / (M['Ar'] * AMU) * Te
                 + (12 + 4.95) * k['Ar_iz_m'] * nArm)
        Ec = np.clip(Eloss / max(Riz_electron, 1e-30), 10, 2000)
        eiw = 0.5 * Te * np.log(max(M['SF5'] * AMU / (2 * pi * m_e), 1))
        eT = Ec + eiw + 2 * Te

        # Power balance (with Penning correction)
        eT_wall = eiw + 2 * Te
        ne_new_numer = P_abs - R_Penn_vol * eT_wall * eC * reactor.V
        ne_new_numer = max(ne_new_numer, P_abs * 0.1)
        ne_new = (ne_new_numer / (Riz_electron * eT * eC * reactor.V)
                  if Riz_electron > 0 and eT > 0 else 1e15)
        ne_new = np.clip(ne_new, 1e10, ne_cap)

        # Te from particle balance
        def Tf(T):
            kk = rates(T)
            Ri = (kk['iz_SF6_total'] * nSF6 + (kk['iz25'] + kk['iz26']) * nSF5
                  + kk['iz27'] * nSF3 + kk['iz28'] * nF + kk['iz29'] * nS
                  + kk['Ar_iz'] * nAr0)
            Ra = kk['att_SF6_total'] * nSF6
            kw = kw_ion(T, M['SF5'], alpha, T_neg, reactor, ng)
            R_qh = (kk['Penn_SF6'] * nSF6 + kk['qnch_SF6'] * nSF6
                    + kk['qnch_SFx'] * (nSF5 + nSF4 + nSF3 + nSF2 + nSF)
                    + kk['qnch_F2'] * nF2 + kk['qnch_F'] * nF)
            nArm_t = kk['Ar_exc'] * ne * nAr0 / ((kk['Ar_iz_m'] + kk['Ar_q']) * ne + k_Arm_wall + R_qh + 1e-30)
            Ri += kk['Ar_iz_m'] * nArm_t
            R_Penn_per_ne = kk['Penn_SF6'] * nArm_t * nSF6 / max(ne, 1e10)
            return Ri + R_Penn_per_ne - Ra - kw * (1 + alpha)

        try:
            Te_new = brentq(Tf, 0.5, 15, xtol=0.005)
        except (ValueError, RuntimeError):
            Te_new = Te

        # Adaptive relaxation
        w = 0.08
        Te = Te + w * (Te_new - Te)
        ne = ne * (ne_new / ne)**w
        alpha = alpha * (max(alpha_new, 0.001) / max(alpha, 0.001))**w
        Te = np.clip(Te, 0.5, 15)
        ne = np.clip(ne, 1e10, ne_cap)
        alpha = np.clip(alpha, 0, 5000)

        if outer > 50:
            dTe = abs(Te - Te_old) / (Te_old + 0.1)
            dne = abs(ne - ne_old) / (ne_old + 1e10)
            dal = abs(alpha - al_old) / (al_old + 0.1)
            if dTe < tol_Te and dne < tol_ne and dal < tol_alpha:
                converged = True
                break

    dissoc = 1 - nSF6 / max(nSF6_0, 1) if nSF6_0 > 0 else 1
    return {
        'Te': Te, 'ne': ne, 'alpha': alpha,
        'n_SF6': nSF6, 'n_SF5': nSF5, 'n_SF4': nSF4,
        'n_SF3': nSF3, 'n_SF2': nSF2, 'n_SF': nSF, 'n_S': nS,
        'n_F': nF, 'n_F2': nF2,
        'Ec': Ec, 'eps_T': eT, 'dissoc_frac': dissoc,
        'converged': converged, 'iter': outer,
        'nArm': nArm, 'nAr0': nAr0, 'R_Penning': R_Penning,
        'ng0': ng0, 'nSF6_0': nSF6_0, 'tau': tau,
        'ns': {'SF6': nSF6, 'SF5': nSF5, 'SF4': nSF4, 'SF3': nSF3,
               'SF2': nSF2, 'SF': nSF, 'S': nS, 'F': nF, 'F2': nF2},
    }
