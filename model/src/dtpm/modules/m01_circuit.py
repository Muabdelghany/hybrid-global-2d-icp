"""
Module 01 — RF Circuit Model (Lieberman Transformer-Coupled ICP)
================================================================
Self-consistent coupled-circuit model for an ICP reactor following
Lieberman & Lichtenberg (2005), Chapter 12.

Physical picture
----------------
The ICP coil is a primary with real resistance R_coil and inductance L_coil.
The plasma couples inductively through mutual inductance M, presenting an
equivalent reflected impedance Z_plasma = R_plasma + j omega L_plasma.  In
the coil circuit:

    V_rf = I_peak * (R_coil + R_plasma + j omega (L_coil + L_plasma))

For steady-state power delivery:

    P_rf = 0.5 * I_peak^2 * (R_coil + R_plasma)
    => I_peak = sqrt( 2 P_rf / (R_coil + R_plasma) )

Coupling efficiency emerges as:

    eta = R_plasma / (R_coil + R_plasma)

R_plasma is NOT prescribed.  It is obtained from the plasma conductivity
sigma(r,z) and the coil-coupling geometry through the Ohmic integral of
the FDTD-computed E-field (module m10).  R_plasma = 2 P_abs / I_peak^2 is
updated every Picard iteration as sigma(r,z) evolves.

Key properties
--------------
- No "matched 50 Ohm load" assumption.  Z = 50 Ohm (from config) is
  purely the RF supply's characteristic impedance and appears only in a
  diagnostic `V_rf_nominal` output.
- I_peak is physically determined by P_rf, R_coil, and the plasma-dependent
  R_plasma.  On the first Picard iteration we seed R_plasma with a
  literature guess (5 Ohm per Lieberman 2005); every subsequent iteration
  updates it from the converged FDTD integral.
- eta is a computed observable, not a dial.  A unit test in
  tests/test_selfconsistent_eta.py verifies that eta = 1 when R_coil = 0
  and eta = 0 when R_plasma = 0.

This module replaces the prior `V_rms = sqrt(P*Z)` matched-load formula
(see docs/CODE_REVIEW_ULTRAREVIEW.md section B for the legacy issue).
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Lieberman 2005, Ch. 12 typical value for the low-pressure plasma
# loading seen by a 3-turn industrial-class ICP coil at 40 MHz.  Used only as
# an initial seed; the Picard loop converges R_plasma to the physically
# correct value within ~3 iterations.
R_PLASMA_INITIAL_GUESS = 5.0   # [Ohm]


def compute_coil_current(P_rf, R_coil, R_plasma):
    """Compute the coil peak current from the RLC + transformer circuit.

    Parameters
    ----------
    P_rf : float
        Total RF power delivered by the source [W].
    R_coil : float
        Real resistance of the coil (loss term) [Ohm].
    R_plasma : float
        Plasma-reflected resistance [Ohm].  From the Ohmic integral,
        updated by m10 every Picard iteration.

    Returns
    -------
    I_peak : float
        Peak coil current [A].
    eta : float
        Power coupling efficiency = R_plasma / (R_coil + R_plasma).
    """
    R_total = R_coil + R_plasma
    if R_total <= 0:
        raise ValueError(f"R_coil + R_plasma = {R_total} must be positive")
    I_peak = float(np.sqrt(2.0 * P_rf / R_total))
    eta = float(R_plasma / R_total)
    return I_peak, eta


def compute_circuit_parameters(config, R_plasma=None):
    """Compute the coil circuit parameters for a given plasma loading.

    Parameters
    ----------
    config : SimulationConfig
        Reads `circuit.source_power`, `circuit.source_frequency`,
        `circuit.R_coil`, `circuit.L_coil`.
    R_plasma : float or None
        Plasma-reflected resistance [Ohm].  If None, uses the initial
        guess R_PLASMA_INITIAL_GUESS.

    Returns
    -------
    dict with P_rf, f, omega, R_coil, L_coil, R_plasma, I_peak, V_peak,
    eta (initial), V_rf_nominal (diagnostic only; = sqrt(P_rf * Z_nominal)
    with Z_nominal from config — has no effect on the simulation).
    """
    if hasattr(config, 'circuit'):
        circuit = config.circuit
    elif isinstance(config, dict) and 'circuit' in config:
        circuit = config['circuit']
    else:
        circuit = config

    P_rf = circuit['source_power']
    f = circuit['source_frequency']
    # LIT: Lieberman 2005 Eq 12.2.19 typical industrial-class 3-turn coil
    R_coil = circuit.get('R_coil', 0.8)
    # LIT: Lieberman 2005 §12.2 typical industrial ICP coil
    L_coil = circuit.get('L_coil', 2.0e-6)
    Z_nominal = circuit.get('impedance', 50.0)

    if R_plasma is None:
        R_plasma = R_PLASMA_INITIAL_GUESS

    omega = 2.0 * np.pi * f
    I_peak, eta = compute_coil_current(P_rf, R_coil, R_plasma)
    V_peak = I_peak * np.sqrt((R_coil + R_plasma)**2
                              + (omega * L_coil)**2)
    V_rf_nominal = float(np.sqrt(P_rf * Z_nominal) * np.sqrt(2.0))

    params = {
        'P_rf': float(P_rf),
        'frequency': float(f),
        'omega': float(omega),
        'R_coil': float(R_coil),
        'L_coil': float(L_coil),
        'R_plasma': float(R_plasma),
        'I_peak': float(I_peak),
        'V_peak': float(V_peak),
        'eta': float(eta),
        'V_rf_nominal': V_rf_nominal,
        'Z_nominal': float(Z_nominal),
    }
    logger.info(
        f"Circuit: P_rf={P_rf:.1f}W, f={f/1e6:.1f}MHz, "
        f"R_coil={R_coil:.2f}Ohm, L_coil={L_coil*1e6:.2f}uH"
    )
    logger.info(
        f"  seed R_plasma={R_plasma:.2f}Ohm -> I_peak={I_peak:.2f}A, "
        f"V_peak={V_peak:.1f}V, eta={eta:.3f}"
    )
    return params


def update_circuit_from_Pabs(P_rf, R_coil, I_peak_prev, P_abs):
    """Update the coil circuit given a new Ohmic-integral P_abs.

    This is the closed-form update used inside the Picard loop:

        R_plasma_new = 2 * P_abs / I_peak_prev^2
        I_peak_new   = sqrt( 2 P_rf / (R_coil + R_plasma_new) )
        eta_new      = R_plasma_new / (R_coil + R_plasma_new)

    Parameters
    ----------
    P_rf : float
        Total RF power [W].
    R_coil : float
        Coil resistance [Ohm].
    I_peak_prev : float
        Coil current used for the FDTD run that produced `P_abs` [A].
    P_abs : float
        Absorbed power from the Ohmic integral over the FDTD E-field [W].

    Returns
    -------
    dict with R_plasma, I_peak, eta (all updated from the latest P_abs).
    """
    if I_peak_prev <= 0:
        raise ValueError(f"I_peak_prev = {I_peak_prev} must be positive")
    R_plasma = 2.0 * P_abs / I_peak_prev**2
    I_peak, eta = compute_coil_current(P_rf, R_coil, max(R_plasma, 1e-6))
    return {
        'R_plasma': float(R_plasma),
        'I_peak': float(I_peak),
        'eta': float(eta),
    }


# --- Pipeline interface ---

def run(state, config):
    """Pipeline-compatible entry point for M01.

    Produces the initial circuit parameters with an R_plasma seed.  The
    Picard loop in m11_plasma_chemistry updates R_plasma and I_peak each
    iteration via `update_circuit_from_Pabs()`.
    """
    circuit_params = compute_circuit_parameters(config)
    return {
        'circuit': circuit_params,
        'I_peak': circuit_params['I_peak'],
        'V_peak': circuit_params['V_peak'],
        'omega': circuit_params['omega'],
        'R_coil': circuit_params['R_coil'],
        'L_coil': circuit_params['L_coil'],
        'R_plasma': circuit_params['R_plasma'],
        'eta_circuit': circuit_params['eta'],
    }
