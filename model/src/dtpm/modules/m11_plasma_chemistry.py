"""
Module 11 — Plasma Chemistry Coupling (Self-Consistent Picard Iteration)
=========================================================================
Implements the outer Picard iteration loop coupling:

  Circuit (m01)
    -> I_peak from P_rf, R_coil, R_plasma (Lieberman transformer coupling)
  FDTD (m06c)
    -> E_theta_rms(r,z) from I_peak and current sigma(r,z)
  Power deposition (m10)
    -> P(r,z) = 0.5 * sigma * |E_theta|^2  (no rescaling; physical magnitude)
    -> P_abs  = integral P(r,z) dV
  Circuit update (m01)
    -> R_plasma = 2 P_abs / I_peak^2
    -> I_peak   = sqrt(2 P_rf / (R_coil + R_plasma))
    -> eta      = R_plasma / (R_coil + R_plasma)
  Energy & transport
    -> Te(r,z) from local power balance
    -> ne(r,z) from ambipolar diffusion
    -> Neutral species transport (9-species SF6 chemistry)

Key architectural change from the prior Phase-1 version
--------------------------------------------------------
The FDTD E-field magnitude is NO LONGER rescaled to match an eta_initial
target.  Instead, I_peak is the physical input and eta emerges as an
observable.  Maxwell's equations are linear in the coil current J, so
when the circuit update changes I_peak by a factor alpha, the stored
E_theta_rms is rescaled by the same alpha without re-running the FDTD.
The FDTD is re-run only when the plasma conductivity sigma(r,z) drifts
significantly (governed by config parameter `rerun_fdtd_every`).

See docs/CODE_REVIEW_ULTRAREVIEW.md for the full legacy diagnosis.
"""

import numpy as np
import logging
from scipy.optimize import brentq
from scipy.constants import e as eC

from . import m06_fdtd_cylindrical
from . import m01_circuit
from . import m12_ccp_bias_sheath
from .m10_power_deposition import compute_power_deposition
from ..solvers.te_energy_transport import solve_Te_energy_transport
from ..chemistry.rate_table import RateTable
from ..solvers.ambipolar_diffusion import solve_ne_ambipolar, prescribe_bessel_cosine
from ..solvers.multispecies_transport import solve_multispecies_transport
from ..chemistry.global_model import solve_0D
from ..chemistry.sf6_rates import rates

logger = logging.getLogger(__name__)


def solve_Te_local_power_balance(P_rz, ne, nSF6_field, mesh, inside, config):
    """Solve for Te(r,z) from local power balance at each active cell.

    P(r,z) = ne(r,z) * nSF6(r,z) * nu_iz(Te) * eps_T(Te) * e
    """
    from scipy.constants import m_e, pi

    Nr, Nz = mesh.Nr, mesh.Nz
    Te = np.full((Nr, Nz), 3.0)

    for i in range(Nr):
        for j in range(Nz):
            if not inside[i, j] or ne[i, j] < 1e10 or P_rz[i, j] < 1e-3:
                continue

            ne_local = ne[i, j]
            nSF6_local = max(nSF6_field[i, j], 1e10)
            P_local = P_rz[i, j]

            def balance(T):
                k = rates(T)
                nu_iz = nSF6_local * k['iz_SF6_total']
                Riz = k['iz_SF6_total']
                Eloss = (16*k['iz18'] + 20*k['iz19'] + 20.5*k['iz20'] + 28*k['iz21']
                         + 37.5*k['iz22'] + 18*k['iz23'] + 29*k['iz24']
                         + 9.6*k['d1'] + 12.1*k['d2'] + 16*k['d3']
                         + 18.6*k['d4'] + 22.7*k['d5'])
                Ec = np.clip(Eloss / max(Riz, 1e-30), 80, 400)
                eiw = 0.5 * T * np.log(max(127 * 1.66e-27 / (2*pi*m_e), 1))
                eps_T = Ec + eiw + 2 * T
                return ne_local * nu_iz * eps_T * eC - P_local

            try:
                Te[i, j] = brentq(balance, 0.5, 15.0, xtol=0.01)
            except (ValueError, RuntimeError):
                Te[i, j] = 3.0

    return Te


def _run_fdtd(state, config, I_peak, sigma_plasma):
    """Invoke m06 FDTD with the given I_peak and plasma conductivity.

    Returns the updated E_theta_rms(r,z) on the full mesh.
    """
    state['I_peak'] = I_peak
    state['sigma_plasma'] = sigma_plasma
    m06_out = m06_fdtd_cylindrical.run(state, config)
    # m06 writes E_theta_rms into state via its return dict
    state.update(m06_out)
    return state['E_theta_rms']


def run(state, config):
    """Run the Picard-coupled EM + plasma chemistry loop with self-consistent eta.

    On each Picard iteration:
      1. Power deposition from current E_theta_rms and sigma -> P_abs.
      2. Close the coil circuit: R_plasma = 2 P_abs / I_peak^2,
         I_peak_new = sqrt(2 P_rf / (R_coil + R_plasma)),
         eta = R_plasma / (R_coil + R_plasma).
      3. Rescale E_theta_rms by I_peak_new / I_peak (Maxwell linearity).
      4. Solve for Te(r,z) from local power balance.
      5. Solve for ne(r,z) from ambipolar diffusion.
      6. Solve the 9-species SF6 chemistry.
      7. Optionally re-run FDTD if sigma has drifted (rerun_fdtd_every).
      8. Check convergence on ne.

    Returns dict with ne, Te, nF, nSF6, P_rz, eta_computed, R_plasma,
    I_peak, F_drop_pct, etc.
    """
    mesh = state['mesh']
    inside = state['inside']
    bc_type = state['bc_type']
    ij_to_flat = state['ij_to_flat']
    flat_to_ij = state['flat_to_ij']
    n_active = state['n_active']

    rgeom = config.reactor_geometry if hasattr(config, 'reactor_geometry') else config.get('reactor_geometry', {})
    oper = config.operating if hasattr(config, 'operating') else config.get('operating', {})
    coup = config.coupling if hasattr(config, 'coupling') else config.get('coupling', {})
    circ = config.circuit if hasattr(config, 'circuit') else config.get('circuit', {})

    R_icp = rgeom.get('R_icp', 0.038)
    L_proc = rgeom.get('L_proc', 0.050)
    L_apt = rgeom.get('L_apt', 0.002)
    L_icp = rgeom.get('L_icp', 0.1815)

    max_picard = coup.get('max_picard_iter', 20)
    picard_tol = coup.get('picard_tol', 0.02)
    w_ne = coup.get('under_relax_ne', 0.3)
    w_Te = coup.get('under_relax_Te', 0.3)
    # Te closure: 'local_balance' (legacy algebraic, per-cell, gated) or
    # 'energy_transport' (conduction + convection PDE over the whole mask).
    te_closure = coup.get('te_closure', 'local_balance')
    inner_iter = coup.get('inner_chem_iter', 60)
    inner_relax = coup.get('inner_chem_relax', 0.12)
    rerun_fdtd_every = coup.get('rerun_fdtd_every', 1)  # every iter by default

    # D4: Tier-2 PINN Boltzmann rates (opt-in via config.chemistry.use_boltzmann_rates)
    chem_cfg = config.chemistry if hasattr(config, 'chemistry') else config.get('chemistry', {})
    # C6/C2/C1 fix flags (all default off -> bit-exact baseline)
    _depleted_att = bool(chem_cfg.get('depleted_attachment', False))
    _ec_clip = (10.0, 2000.0) if bool(chem_cfg.get('ec_clip_wide', False)) else (80.0, 400.0)
    _ne_cap_0D = float(chem_cfg.get('ne_cap_0D', 1e19))
    # Fig-14 comparison hook: a fixed scalar alpha for the ambipolar closure,
    # set by run_ambipolar_comparison.py. When present it pins alpha_2D at both
    # the initial assignment and on every outer pass, and disables the Path-D
    # 2D-alpha refresh, so the two comparison arms differ only in this scalar.
    _alpha_fixed = chem_cfg.get('alpha_fixed', None)
    _alpha_fixed = None if _alpha_fixed is None else float(_alpha_fixed)
    # Diagnostic/refinement: include the bias expansion power in the ELECTRON
    # ENERGY equation (one-iteration Picard lag). Physically the bias source
    # creates electron-ion pairs whose energy flows through the electron
    # population; without this the Te equation never sees the ~320 W and the
    # wafer region stays cold, killing the volumetric bias->F channel.
    _bias_in_te = bool(coup.get('bias_in_te', False))
    # Level-2 emergent eta: constrain absorbed power to what the plasma can
    # dissipate (sink_frac from the Te energy solver). Default OFF.
    _pb_eta = bool(coup.get('power_balance_eta', False))
    _pb_relax = float(coup.get('power_balance_relax', 0.5))
    # GATED power-balance mode (pb_eta_gated): certify the outer loop on
    # the emergent absorbed fraction with inner-convergence + energy-
    # balance gates. Default OFF -> legacy/EEDF runs bit-unchanged.
    _pb_gated = bool(coup.get('pb_eta_gated', False))
    _pb_energy_tol = float(coup.get('pb_energy_tol', 0.05))
    _eta_eff = None
    _eta_eff_prev = None
    _sink_frac_prev = 1.0
    _P_diss_prev = None   # measured dissipatable power [W] from the Te solver
    # In the f_e_bias closure the energy channel IS the mechanism -- force it on.
    _bias_cfg = config.bias if hasattr(config, 'bias') else config.get('bias', {})
    if str(_bias_cfg.get('closure', 'lambda_exp')) == 'f_e_bias':
        _bias_in_te = True
    use_boltzmann = bool(chem_cfg.get('use_boltzmann_rates', False))
    from ..chemistry import tier2_interface as _tier2
    if use_boltzmann:
        _tier2.install_pinn()
        logger.info("M11: Tier-2 Boltzmann rates (PINN) ENABLED via config.chemistry")
    else:
        _tier2.clear()  # ensure any prior cache is flushed before tier-1 baseline

    P_rf = circ['source_power']
    R_coil = state.get('R_coil', circ.get('R_coil', 0.8))
    # m01 already seeded state['R_plasma'] and state['I_peak'] with initial guess
    R_plasma = state.get('R_plasma', m01_circuit.R_PLASMA_INITIAL_GUESS)
    I_peak, eta = m01_circuit.compute_coil_current(P_rf, R_coil, R_plasma)

    Nr, Nz = mesh.Nr, mesh.Nz

    # -- Initial guess from 0D global model --
    # The 0D model internally prescribes eta (no spatial E-field); use the
    # m01 circuit seed eta here so the 0D guess is consistent with the 2D
    # circuit model at iteration 0.
    logger.info("M11: Computing initial guess from 0D global model...")
    result_0D = solve_0D(
        P_rf=P_rf,
        p_mTorr=oper.get('pressure_mTorr', 10),
        frac_Ar=oper.get('frac_Ar', 0.0),
        Q_sccm=oper.get('Q_sccm', 100),
        Tgas=oper.get('Tgas', 313),
        eta=eta,  # from the self-consistent circuit seed
        R_icp=R_icp, L_icp=L_icp, ne_cap=_ne_cap_0D,
    )
    alpha_0D = float(result_0D.get('alpha', 0.0))
    logger.info(f"  0D: Te={result_0D['Te']:.2f}eV, ne={result_0D['ne']:.2e}m^-3, "
                f"alpha={alpha_0D:.2f}")
    # L1 correction: electronegative ambipolar diffusion (Lieberman 2005 §10.3).
    #
    # Path D (Phase-1b): optionally use a full 2D alpha(r,z) field rather
    # than the scalar alpha_0D. The 2D alpha is the spatial map
    # n_-(r,z) / n_e(r,z) already computed inside solve_multispecies_transport
    # via the local attachment/recombination balance. It is fed back into
    # solve_ne_ambipolar on the NEXT Picard iter — a one-iteration lag that
    # converges naturally with the rest of the self-consistent loop.
    #
    # Enabled by config.chemistry.use_2d_alpha (default False, preserving
    # the scalar-alpha baseline bit-for-bit).
    # use_2d_alpha: False | True ("renorm", default) | "raw"
    #   "renorm" preserves the local 2D shape but rescales the volume average
    #   to the 0D alpha (transport-corrected, physically correct)
    #   "raw" uses ions['alpha'] directly — diagnostic only, over-estimates
    #   electronegativity because it omits transport loss.
    _alpha_mode = chem_cfg.get('use_2d_alpha', False)
    if isinstance(_alpha_mode, bool):
        use_2d_alpha = _alpha_mode
        alpha_mode = 'renorm' if _alpha_mode else 'off'
    else:
        alpha_mode = str(_alpha_mode).lower()
        use_2d_alpha = alpha_mode in ('renorm', 'raw', 'true', '1')
    alpha_2D = alpha_0D  # scalar initial; may be replaced each iter if use_2d_alpha
    if _alpha_fixed is not None:
        alpha_2D = _alpha_fixed
        use_2d_alpha = False   # comparison mode: no Path-D refresh
    if use_2d_alpha:
        logger.info(f"M11: 2D alpha(r,z) field feedback ENABLED (mode={alpha_mode}, Path D).")

    ne = prescribe_bessel_cosine(result_0D['ne'], mesh, inside,
                                 R_icp, L_proc, L_apt, L_icp)
    Te = np.where(inside, result_0D['Te'], 0.0)

    from scipy.constants import k as kB
    p_Pa = oper.get('pressure_mTorr', 10) * 0.133322
    ng = p_Pa / (kB * oper.get('Tgas', 313))
    nSF6_feed = ng * (1 - oper.get('frac_Ar', 0.0))
    nSF6 = np.where(inside, nSF6_feed * 0.5, 0.0)
    nF = np.where(inside, 1e18, 0.0)

    # Local composition carried across Picard iterations for the Te energy sink.
    # Seeded to match solve_multispecies_transport's own initialisation so the
    # first Te solve sees the same composition the chemistry will start from;
    # replaced by the real fields as soon as step 6 has run once.
    _species_state = {
        'SF6': np.where(inside, nSF6_feed * 0.5, 0.0),
        'F':   np.where(inside, nSF6_feed * 0.1, 0.0),
    }
    for _sp in ('SF5', 'SF4', 'SF3', 'SF2', 'SF', 'S', 'F2'):
        _species_state[_sp] = np.where(inside, nSF6_feed * 0.01, 0.0)
    # Ar metastable density imported from the 0D. This is not only a warm start:
    # it is refreshed from the re-solved 0D on every outer pass, and sf6_rates
    # consumes it in three places, namely electron_source (stepwise ionization
    # and Penning), fluorine_source (Penning and quenching on SF6 and F2) and
    # energy_loss_density (the 12 + 4.95 eV stepwise channel). Ar* is the only
    # species imported from the 0D, because it is the only one appearing in the
    # 2D source terms that has neither a 2D transport equation nor an
    # operating-point expression. It arrives as a uniform scalar, so the Ar*
    # field is taken to be flat. Absent -> 0.0, dropping all three channels.
    # INTERFACE_CONTRACT.md quantifies each channel's share.
    _nArm = float(result_0D.get('nArm', 0.0) or 0.0)
    # NOTE: the RateTable is built INSIDE the Picard loop (after the tier-2
    # refresh), not here -- rates() is impure and tier2 mutates it per
    # iteration in lxcat mode. See the tier-2 refresh below and the solver docstring.
    _rate_table = None

    # -- Initial FDTD solve with seed I_peak and sigma(r,z) --
    # Compute initial sigma from the 0D ne and Te guess via m10 (no rescale).
    E_theta_rms = state.get('E_theta_rms')
    if E_theta_rms is None or not np.any(E_theta_rms > 0):
        # Must invoke FDTD first; initial sigma will be built by m10 inside
        # the first power_deposition call.  We pass sigma_plasma=None so
        # m06 uses the default free-space inside-mask behaviour.
        logger.info("M11: Running initial FDTD with seed I_peak...")
        E_theta_rms = _run_fdtd(state, config, I_peak, sigma_plasma=None)
    else:
        logger.info("M11: Using pre-existing FDTD E-field from state")

    convergence_history = []
    P_abs = 0.0
    P_rz = np.zeros((Nr, Nz))
    F_drop = 0.0

    logger.info(f"M11: Starting Picard iteration (max {max_picard} iter, "
                f"tol={picard_tol}, rerun_fdtd_every={rerun_fdtd_every})")
    print(f"\n{'='*78}")
    print("  Self-Consistent Picard Iteration: Circuit + EM + Chemistry")
    print(f"{'='*78}")
    print(f"  {'It':>3s} {'ne_avg':>10s} {'Te_avg':>6s} {'eta':>6s} "
          f"{'I_peak':>7s} {'Rp':>6s} {'P_abs':>7s} {'Fdrop':>6s} "
          f"{'sigma_max':>10s} {'ne_chg':>8s}")

    # -- Emergent-η OUTER loop (coupling.outer_loop, default OFF) ----------
    # Feeds the CONVERGED circuit η back into the 0D closure and re-solves the
    # electronegativity α_0D, replacing the hardcoded m01 seed η≈0.862 the warm
    # start is otherwise evaluated at.
    #
    # What crosses the interface, established by tracing the data flow:
    #   2D -> 0D : eta only. The init_Te/init_ne/init_alpha/init_ns arguments
    #              below come from result_0D itself, so they are the 0D warm-
    #              starting itself rather than 2D feedback.
    #   0D -> 2D : four quantities. <Te> and <ne> are seeds only, since the 2D
    #              solves both and overwrites them. alpha_0D and nArm are
    #              persistent coefficients and both are refreshed here on every
    #              outer pass.
    # The convergence residual monitors two of these (eta 2D->0D, alpha_0D
    # 0D->2D); nArm rides along unmonitored. nSF6 is not part of the interface
    # in either direction, as the 2D seeds SF6 from the feed density above.
    # alpha is not fed 2D->0D, because the local 2D alpha closure omits the
    # negative-ion wall loss the 0D carries. Measured loop gain 1.2e-5 (10 mT)
    # / 2.0e-4 (20 mT) -> contraction; 2 passes suffice; no damping required.
    outer_on = bool(coup.get('outer_loop', False))
    max_outer = int(coup.get('max_outer', 3)) if outer_on else 1
    outer_tol = float(coup.get('outer_tol', 1e-2))
    w_alpha_outer = float(coup.get('outer_relax_alpha', 1.0))
    outer_history = []
    _eta_prev_outer = None
    _alpha_prev_outer = None
    outer_converged = not outer_on   # trivially true when the loop is off
    _P_bias_prev = None   # previous-iteration bias deposition for bias_in_te

    for m_outer in range(max_outer):
        if m_outer > 0:
            # Re-solve the 0D closure at the EMERGENT η from the converged inner
            # loop, warm-started from its own previous state (0.11 s vs 0.27 s).
            result_0D = solve_0D(
                P_rf=P_rf,
                p_mTorr=oper.get('pressure_mTorr', 10),
                frac_Ar=oper.get('frac_Ar', 0.0),
                Q_sccm=oper.get('Q_sccm', 100),
                Tgas=oper.get('Tgas', 313),
                eta=float(_eta_eff if (_pb_gated and _eta_eff is not None) else eta),  # gated pb: absorbed fraction
                R_icp=R_icp, L_icp=L_icp, ne_cap=_ne_cap_0D,
                init_Te=result_0D.get('Te'), init_ne=result_0D.get('ne'),
                init_alpha=result_0D.get('alpha'), init_ns=result_0D.get('ns'),
                # Tightened 10x-100x vs the defaults: the outer gate certifies
                # |d alpha|/alpha < 1e-2, so the 0D's own alpha truncation must
                # sit well below that. Measured at the default dal<1e-3: the 0D
                # lands within ~2% of its root between warm starts, flooring the
                # outer residual at ~2e-2 (pass-2 alpha wobble +2.3% while eta
                # changed 1e-6 -- pure truncation, not feedback). Costs <0.3 s.
                tol_Te=5e-6, tol_ne=5e-5, tol_alpha=1e-5,
            )
            _alpha_new = float(result_0D.get('alpha', alpha_0D))
            alpha_0D = (1.0 - w_alpha_outer) * alpha_0D + w_alpha_outer * _alpha_new
            if not use_2d_alpha:
                alpha_2D = alpha_0D if _alpha_fixed is None else _alpha_fixed
            _nArm = float(result_0D.get('nArm', 0.0) or 0.0)
            logger.info(
                "OUTER %d: 0D re-solved at emergent eta=%.4f -> alpha_0D=%.5f "
                "(was computed at seed eta on pass 0)", m_outer, eta, alpha_0D)

        for k_iter in range(max_picard):
            ne_old = ne.copy()

            # ----------------------------------------------------------------
            # Step 1: Compute P_abs at CURRENT E_theta_rms (physical magnitude)
            # ----------------------------------------------------------------
            pd = compute_power_deposition(E_theta_rms, ne, Te, mesh, inside, config)
            P_abs = pd['P_abs']
            sigma_rz = pd['sigma_rz']

            # ----------------------------------------------------------------
            # Step 2: Close the coil circuit (Lieberman transformer coupling)
            # ----------------------------------------------------------------
            if P_abs > 1e-3:
                circuit_upd = m01_circuit.update_circuit_from_Pabs(
                    P_rf=P_rf, R_coil=R_coil,
                    I_peak_prev=I_peak, P_abs=P_abs,
                )
                R_plasma_new = circuit_upd['R_plasma']
                I_peak_new = circuit_upd['I_peak']
                eta_new = circuit_upd['eta']
            else:
                # Plasma hasn't ignited yet; retain the initial guess
                R_plasma_new, I_peak_new, eta_new = R_plasma, I_peak, eta

            # ----------------------------------------------------------------
            # Step 3: Rescale E_theta_rms linearly (Maxwell linearity in J)
            # ----------------------------------------------------------------
            # For FIXED sigma(r,z), the FDTD output scales linearly in I_peak.
            # So updating I_peak without re-running FDTD is valid as long as
            # sigma hasn't drifted too much; step 6 handles re-runs.
            if I_peak > 0:
                linear_scale = I_peak_new / I_peak
                E_theta_rms = E_theta_rms * linear_scale
                if _pb_eta and _P_diss_prev is not None:
                    # Rescale absorbed power TO the measured dissipatable level (P ~ E^2).
                    # P_now after the circuit rescale is eta_new*P_rf by construction.
                    _P_now = float(eta * P_rf)
                    if _P_diss_prev < _P_now:
                        _P_target = ((1.0 - _pb_relax) * _P_now
                                     + _pb_relax * max(_P_diss_prev, 1.0))
                        E_theta_rms = E_theta_rms * float(np.sqrt(_P_target / max(_P_now, 1e-30)))

            I_peak = I_peak_new
            R_plasma = R_plasma_new
            eta = eta_new

            # Recompute P_abs after the linear scale (should equal eta*P_rf)
            pd = compute_power_deposition(E_theta_rms, ne, Te, mesh, inside, config)
            P_rz = pd['P_rz']
            P_abs = pd['P_abs']
            sigma_rz = pd['sigma_rz']

            # D4 hook: refresh tier-2 PINN rates from the current EM field.
            # Runs once per outer Picard iter when use_boltzmann_rates is True.
            if use_boltzmann:
                E_over_N_Td = _tier2.compute_eff_E_over_N(
                    E_theta_rms, ng, inside, mesh
                )
                _tier2.refresh(
                    E_over_N_Td,
                    x_Ar=oper.get('frac_Ar', 0.0),
                    pressure_mTorr=oper.get('pressure_mTorr', 10),
                )

            # ----------------------------------------------------------------
            # Step 4: Te(r,z) from local power balance
            # ----------------------------------------------------------------
            if te_closure == 'energy_transport':
                # Rebuilt every iteration: rates() applies tier-2 PINN overrides
                # from mutable module state that _tier2.refresh() just updated,
                # so a table cached across iterations would be stale in lxcat mode
                # and would evaluate the energy sink with a different rate set
                # than the ne solve uses.
                _rate_table = RateTable()
                _P_for_te = P_rz
                if _bias_in_te and _P_bias_prev is not None:
                    _P_for_te = P_rz + _P_bias_prev
                Te_new, te_info = solve_Te_energy_transport(
                    _P_for_te, ne, _species_state, mesh, inside, config,
                    bc_type, ij_to_flat, flat_to_ij, n_active,
                    Te_init=Te, alpha=alpha_2D, nArm=_nArm,
                    rate_table=_rate_table)
                if _pb_eta and te_info is not None:
                    _sf = te_info.get('sink_frac')
                    if _sf is not None and np.isfinite(_sf):
                        _sink_frac_prev = float(_sf)
                    _pv = te_info.get('P_volloss_W'); _pw = te_info.get('P_wall_W')
                    if _pv is not None and _pw is not None and np.isfinite(_pv + _pw):
                        _P_diss_prev = float(_pv + _pw)
            else:
                Te_new = solve_Te_local_power_balance(
                    P_rz, ne, nSF6, mesh, inside, config)
                te_info = None

            # ----------------------------------------------------------------
            # Step 4b: Wafer-bias sheath (m12) — adds an expansion source in
            # the process chamber when bias is enabled.  Bypassed entirely
            # when config.bias.enabled = False or P_bias_W = 0.
            # ----------------------------------------------------------------
            bias = m12_ccp_bias_sheath.compute_bias_sheath(
                ne, Te_new, mesh, inside, config)
            if bias['enabled']:
                _P_bias_prev = bias['P_rz_bias']
                P_rz_total = P_rz + bias['P_rz_bias']
                P_abs_total = P_abs + float(np.sum(
                    bias['P_rz_bias'] * mesh.vol * inside))
            else:
                P_rz_total = P_rz
                P_abs_total = P_abs

            # ----------------------------------------------------------------
            # Step 5: ne(r,z) from ionization-source diffusion
            # ----------------------------------------------------------------
            ne_new, ne_avg = solve_ne_ambipolar(
                Te_new, P_abs_total, mesh, inside, bc_type,
                ij_to_flat, flat_to_ij, n_active, config, P_rz=P_rz_total,
                alpha=alpha_2D,
                nSF6_field=(nSF6 if _depleted_att else None), ec_clip=_ec_clip,
            )

            # ----------------------------------------------------------------
            # Step 6: Full 9-species neutral chemistry
            # ----------------------------------------------------------------
            chem_result = solve_multispecies_transport(
                mesh, inside, bc_type, ij_to_flat, flat_to_ij, n_active,
                ne_new, Te_new, config,
                n_iter=inner_iter, w=inner_relax, verbose=(k_iter == 0),
            )
            nF = chem_result['nF']
            nSF6 = chem_result['nSF6']
            F_drop = chem_result['F_drop_pct']

            # Refresh the composition the NEXT iteration's Te energy sink will use.
            _fields = chem_result.get('fields', {})
            for _sp, _val in _fields.items():
                _species_state[_sp] = np.asarray(_val)

            # Path D: update alpha_2D from the 2D ion fields for the NEXT Picard
            # iteration's ambipolar solve. The raw `ions['alpha']` from
            # solve_multispecies_transport uses a LOCAL attachment/recombination
            # quadratic closure that ignores transport loss, so its volume-
            # average is systematically too high. We preserve the relative
            # spatial SHAPE of that field but RENORMALISE the volume average
            # to the 0D alpha (which correctly accounts for wall-transport loss
            # in a finite reactor). This is the physically-honest 2D extension
            # of the scalar L1 correction.
            if use_2d_alpha and 'ions' in chem_result and 'alpha' in chem_result['ions']:
                alpha_local = np.asarray(chem_result['ions']['alpha'])
                if alpha_mode == 'raw':
                    # Diagnostic: use the local attachment/recombination quadratic
                    # directly. Over-estimates alpha because transport loss is
                    # ignored; NOT physically correct but useful for sensitivity.
                    alpha_field_new = np.where(inside, alpha_local, 0.0)
                    scale = 1.0
                elif np.any(inside):
                    a_in = alpha_local[inside]
                    a_mean_local = float(np.mean(a_in)) if a_in.size else 1.0
                    a_mean_local = max(a_mean_local, 1e-12)
                    scale = alpha_0D / a_mean_local  # renormalise to 0D volume avg
                    alpha_shape = np.where(inside, alpha_local, 0.0)
                    alpha_field_new = alpha_shape * scale
                else:
                    alpha_field_new = np.full_like(alpha_local, alpha_0D)
                    scale = 1.0
                if k_iter == 0:
                    alpha_2D = alpha_field_new
                else:
                    # Blend 70% new / 30% previous to dampen Picard oscillation
                    alpha_2D = 0.7 * alpha_field_new + 0.3 * np.asarray(alpha_2D)
                if k_iter in (0, max_picard - 1) or k_iter % 5 == 0:
                    a = alpha_2D[inside] if np.ndim(alpha_2D) > 0 else np.array([alpha_2D])
                    logger.info(
                        f"  Path-D alpha(r,z): min={a.min():.3e}, mean={a.mean():.3e}, "
                        f"max={a.max():.3e}, renorm-scale={scale:.3e} (alpha_0D={alpha_0D:.3e})"
                    )

            # ----------------------------------------------------------------
            # Step 7: Re-run FDTD periodically if sigma has drifted
            # ----------------------------------------------------------------
            if rerun_fdtd_every > 0 and ((k_iter + 1) % rerun_fdtd_every == 0):
                # Pass the latest sigma_rz into the FDTD so dispersion is consistent.
                # I_peak stays the same — circuit closure happens at the top of the
                # next iteration.
                logger.info(f"  Re-running FDTD with updated sigma (iter {k_iter+1})")
                E_theta_rms = _run_fdtd(state, config, I_peak, sigma_plasma=sigma_rz)

            # ----------------------------------------------------------------
            # Step 8: Convergence
            # ----------------------------------------------------------------
            ne_inside = ne_old[inside]
            ne_new_inside = ne_new[inside]
            rel_change = (np.linalg.norm(ne_new_inside - ne_inside)
                          / max(np.linalg.norm(ne_inside), 1e-30))

            Te_avg = np.mean(Te_new[inside]) if np.any(inside) else 3.0

            convergence_history.append({
                'iter': k_iter,
                'ne_avg': float(ne_avg),
                'Te_avg': float(Te_avg),
                'eta': float(eta),
                'I_peak': float(I_peak),
                'R_plasma': float(R_plasma),
                'P_abs': float(P_abs),
                'F_drop_pct': float(F_drop),
                'rel_change': float(rel_change),
            })

            print(f"  {k_iter:3d} {ne_avg*1e-6:10.2e} {Te_avg:6.2f} {eta:6.3f} "
                  f"{I_peak:7.2f} {R_plasma:6.2f} {P_abs:7.1f} {F_drop:5.1f}% "
                  f"{sigma_rz.max():10.2e} {rel_change:8.4f}")

            if rel_change < picard_tol and k_iter > 0:
                logger.info(f"  Picard converged at iteration {k_iter+1} "
                            f"(rel_change={rel_change:.4e})")
                break

            # Under-relax ne, Te to stabilise the outer iteration
            ne = (1 - w_ne) * ne + w_ne * ne_new
            Te = (1 - w_Te) * Te + w_Te * Te_new

        # -- outer-loop residual on the interface pair (η, α_0D) --
        _inner_conv = bool(rel_change < picard_tol)
        if _pb_gated:
            # The physically meaningful efficiency in power-balance mode is the
            # ABSORBED fraction P_abs/P_rf, not the circuit eta. Feed THAT
            # forward and certify the outer loop on it, plus inner convergence
            # and a saved-field energy balance -- alpha alone is insufficient.
            _eta_eff = float(P_abs / P_rf) if P_rf > 0 else 0.0
            _Ps = float(te_info.get('P_source_W', 0.0)) if te_info else 0.0
            _Pl = ((float(te_info.get('P_volloss_W', 0.0))
                    + float(te_info.get('P_wall_W', 0.0))) if te_info else 0.0)
            _r_energy = abs(_Ps - _Pl) / max(abs(_Ps), 1e-30)
            _r_eta = (abs(_eta_eff - _eta_eff_prev) / max(abs(_eta_eff), 1e-30)
                      if _eta_eff_prev is not None else float('inf'))
            _r_alpha = (abs(float(alpha_0D) - _alpha_prev_outer)
                        / max(abs(float(alpha_0D)), 1e-12)
                        if _alpha_prev_outer is not None else float('inf'))
            r_outer = max(_r_eta, _r_alpha)
            outer_history.append({
                'outer': int(m_outer),
                'eta_circuit': float(eta),
                'eta_eff': float(_eta_eff),
                'alpha_0D': float(alpha_0D),
                'residual': (None if r_outer == float('inf') else float(r_outer)),
                'r_eta_eff': (None if _r_eta == float('inf') else float(_r_eta)),
                'r_alpha': (None if _r_alpha == float('inf') else float(_r_alpha)),
                'energy_residual': float(_r_energy),
                'inner_converged': _inner_conv,
                'inner_rel_change': float(rel_change),
                'inner_iters': int(k_iter + 1),
            })
            _eta_eff_prev = float(_eta_eff)
            _alpha_prev_outer = float(alpha_0D)
            if (outer_on and _inner_conv and _r_energy < _pb_energy_tol
                    and r_outer < outer_tol):
                outer_converged = True
                logger.info("OUTER (gated) converged pass %d: r=%.2e "
                            "energy_resid=%.2e inner_conv=%s",
                            m_outer, r_outer, _r_energy, _inner_conv)
                break
        else:
            if _eta_prev_outer is not None:
                # GATE ON alpha_0D ONLY (measured design correction, Stage 1):
                # eta wobbles ~1% from inner truncation, so an eta-residual never
                # falls below ~2e-2; alpha_0D is the smoothly fed-forward quantity.
                r_outer = (abs(float(alpha_0D) - _alpha_prev_outer)
                           / max(abs(float(alpha_0D)), 1e-12))
            else:
                r_outer = float('inf')
            outer_history.append({
                'outer': int(m_outer),
                'eta': float(eta),
                'alpha_0D': float(alpha_0D),
                'residual': (None if r_outer == float('inf') else float(r_outer)),
                'inner_iters': int(k_iter + 1),
            })
            _eta_prev_outer = float(eta)
            _alpha_prev_outer = float(alpha_0D)
            if outer_on and r_outer < outer_tol:
                outer_converged = True
                logger.info("OUTER loop converged at pass %d (residual=%.2e < %.0e)",
                            m_outer, r_outer, outer_tol)
                break
    if outer_on and not outer_converged:
        logger.warning("OUTER loop hit max_outer=%d without meeting tol %.0e "
                       "(last residual on record in outer_history)", max_outer, outer_tol)


    print(f"{'='*78}")
    print(f"  Picard complete: {k_iter+1} iterations, "
          f"eta={eta:.3f}, I_peak={I_peak:.2f} A, R_plasma={R_plasma:.2f} Ohm")
    print(f"  [F] drop={F_drop:.1f}%, P_abs={P_abs:.1f} W (P_rf={P_rf} W)")
    print(f"{'='*78}\n")

    species_fields = chem_result.get('fields', {})
    ions = chem_result.get('ions', {})

    # Operating voltages at the coil driven port (Lieberman Eq 12.2.19
    # at matched-resonance condition: reactive part cancelled by the
    # matching network, so V across coil is I * (R_coil + R_plasma)).
    V_peak_final = float(I_peak * (R_coil + R_plasma))
    V_rms_final = V_peak_final / np.sqrt(2.0)

    # Re-evaluate the Te energy balance on the FINAL SAVED field. The per-iteration
    # `te_info` describes the solver's raw output, but the saved `Te` is the
    # under-relaxed Picard blend; for a converged solve they match, but for a
    # non-converged one (e.g. lxcat clamping every iteration) the raw value
    # over-states the saved field, so the summary would otherwise mis-report the
    # converged Te and its sink_frac. This makes the reported diagnostic honest.
    if te_closure == 'energy_transport' and te_info is not None:
        from ..solvers.te_energy_transport import energy_balance
        _saved = energy_balance(Te, ne, _species_state, mesh, inside, P_rz, config,
                                nArm=_nArm, rate_table=_rate_table)
        # keep the solver's convergence fields; overwrite the field-dependent
        # balance with the values for the field we actually save.
        te_info = {**te_info, **_saved, 'te_info_on': 'saved_field'}

    # Capture tier-2 cache snapshot for diagnostics, then clear to avoid
    # polluting subsequent tier-1 runs in the same Python process.
    tier2_snapshot = dict(_tier2.current_cache()) if _tier2.is_active() else None
    _tier2.clear()

    result = {
        'ne': ne,
        'Te': Te,
        # Te closure diagnostics (sink_frac ~1 means the electron energy
        # budget closes; far from 1 means the sink is mis-specified).
        'te_info': te_info if te_closure == 'energy_transport' else None,
        # Emergent-η outer-loop diagnostics (coupling.outer_loop). With the flag
        # off this is a single trivial pass and outer_converged is True.
        'power_balance_eta_on': bool(_pb_eta),
        'eta_emergent_power_balance': (float(P_abs / P_rf) if _pb_eta and P_rf > 0 else None),
        'sink_frac_final': (float(_sink_frac_prev) if _pb_eta else None),
        'pb_eta_gated_on': bool(_pb_gated),
        'eta_eff_final': (float(_eta_eff) if _pb_gated and _eta_eff is not None else None),
        'inner_converged_final': bool(rel_change < picard_tol),
        'inner_rel_change_final': float(rel_change),
        'outer_loop_on': bool(outer_on),
        'outer_history': outer_history,
        'outer_iters': len(outer_history),
        'outer_converged': bool(outer_converged),
        'nF': nF,
        'nSF6': nSF6,
        'P_rz': P_rz,
        'P_abs': float(P_abs),
        'P_abs_final': float(P_abs),
        'eta_computed': float(eta),
        'I_peak_final': float(I_peak),
        'R_plasma_final': float(R_plasma),
        'V_peak_final': V_peak_final,
        'V_rms_final': V_rms_final,
        'R_coil': float(R_coil),
        'E_theta_rms': E_theta_rms,
        'F_drop_pct': float(F_drop),
        'ne_avg': float(ne_avg),
        'picard_iterations': k_iter + 1,
        'convergence_history': convergence_history,
        'result_0D': result_0D,
        'species_fields': species_fields,
        'ions': ions,
        # Bias diagnostics (zero-valued when bias disabled)
        'bias_enabled': bool(bias.get('enabled', False)),
        'bias_V_dc': float(bias.get('V_dc', 0.0)),
        'bias_Gamma_i': float(bias.get('Gamma_i', 0.0)),
        'bias_P_ion_to_gas': float(bias.get('P_ion_to_gas', 0.0)),
        'bias_lambda_exp': float(bias.get('lambda_exp', 0.0)),
        'bias_P_bias_total': float(np.sum(
            bias.get('P_rz_bias', np.zeros((Nr, Nz))) * mesh.vol * inside
        )) if bias.get('enabled', False) else 0.0,
        # D4 tier-2 snapshot (None when use_boltzmann_rates=False)
        'tier2_boltzmann_cache': tier2_snapshot,
        'use_boltzmann_rates': use_boltzmann,
    }
    for sp_name, sp_field in species_fields.items():
        result[f'n{sp_name}'] = sp_field
    return result
