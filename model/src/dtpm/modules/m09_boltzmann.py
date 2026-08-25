"""
Module 09 — Boltzmann Solver (PINN-based)
==========================================
Computes the Electron Energy Distribution Function (EEDF) and derived
transport/rate coefficients using a Physics-Informed Neural Network.

Physics:
- Solves the steady-state, spatially homogeneous two-term Boltzmann equation
  for electrons in a weakly ionized gas.
- Input: reduced electric field E/N [Td]
- Output: EEDF f(epsilon), transport coefficients (mu_e, D_e), rate
  coefficients k_j, electron temperature Te

Modes:
- inference: loads a pre-trained PINN model and evaluates at given E/N
- bolos: uses the bolos two-term Boltzmann solver directly (fallback)
- train: triggers PINN training pipeline (not used in normal simulation)

References:
- Kim 2023: PINNs for Boltzmann with variable E/N
- Kawaguchi 2022: PINNs for electron velocity distribution
- Hagelaar & Pitchford, PSST 14, 2005 (two-term Boltzmann equation)
"""

import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Physical constants
KB = 1.380649e-23   # Boltzmann constant [J/K]
QE = 1.602176634e-19  # elementary charge [C]


def compute_gas_density(config):
    """Compute neutral gas density from pressure and temperature.

    Uses ideal gas law: N = p / (kB * T)

    Parameters
    ----------
    config : SimulationConfig
        Must have chemistry.pressure_Pa and chemistry.gas_temperature_K.

    Returns
    -------
    N : float
        Gas number density [m^-3].
    """
    chem = config.chemistry if hasattr(config, 'chemistry') else config.get('chemistry', {})
    pressure = chem.get('pressure_Pa', 1.0)
    temperature = chem.get('gas_temperature_K', 300.0)
    return pressure / (KB * temperature)


def run_pinn_inference(state, config):
    """Run M09 using pre-trained PINN model.

    Parameters
    ----------
    state : dict
        Pipeline state (may contain E_mag from upstream modules).
    config : SimulationConfig
        Configuration with boltzmann section.

    Returns
    -------
    dict
        EEDF, transport coefficients, rate coefficients.
    """
    from ..solvers.boltzmann_pinn import BoltzmannPINNTrainer, compute_transport_coefficients
    from ..chemistry.cross_sections import CrossSectionSet

    boltz_cfg = config.get('boltzmann', {})
    inference_cfg = boltz_cfg.get('inference', {})

    # Load model
    model_path = inference_cfg.get('model_path', 'data/boltzmann_models/ar_pinn.pt')
    # Resolve relative paths from project root
    if not Path(model_path).is_absolute():
        # Try relative to config file or current directory
        for base in [Path('.'), Path(config.get('_config_dir', '.'))]:
            candidate = base / model_path
            if candidate.exists():
                model_path = str(candidate)
                break

    logger.info(f"Loading PINN model from {model_path}")
    model = BoltzmannPINNTrainer.load_model(model_path)

    # Load cross-sections for transport coefficient computation
    cs_dir = boltz_cfg.get('cross_section_dir', 'data/cross_sections/ar')
    gas = boltz_cfg.get('gas_system', 'ar')
    cs = CrossSectionSet(gas, cs_dir)

    # Determine E/N
    N_gas = compute_gas_density(config)
    energy_max = inference_cfg.get('energy_max_eV', 60.0)
    n_energy = 200
    eps_grid = np.linspace(0.01, energy_max, n_energy)

    spatially_resolved = inference_cfg.get('spatially_resolved', False)

    if spatially_resolved and 'E_mag' in state:
        # Per-grid-point evaluation
        E_mag = state['E_mag']
        EN = E_mag / N_gas / 1e-21  # Convert to Td
        EN = np.clip(EN, 0.1, 1000.0)  # Clamp to training range

        # For now, use volume-averaged E/N
        EN_mean = np.mean(EN[EN > 0]) if np.any(EN > 0) else 10.0
        logger.info(f"Spatially resolved: EN_mean = {EN_mean:.2f} Td")
        en_td = EN_mean
    else:
        # 0-D mode: use config value or default
        en_td = boltz_cfg.get('EN_Td', 50.0)
        logger.info(f"0-D mode: E/N = {en_td:.2f} Td")

    # Predict EEDF
    eedf = model.predict_eedf(en_td, eps_grid)

    # Compute transport and rate coefficients
    tc = compute_transport_coefficients(eedf, eps_grid, cs)

    logger.info(f"M09 results: Te={tc['Te']:.3f} eV, <eps>={tc['mean_energy']:.3f} eV, "
                f"mu*N={tc['mobility_N']:.3e}")
    for name, rate in tc['rate_coefficients'].items():
        logger.info(f"  k_{name} = {rate:.3e} m^3/s")

    return {
        'eedf': eedf,
        'eedf_energy_grid': eps_grid,
        'Te': tc['Te'],
        'mean_energy': tc['mean_energy'],
        'mu_e_N': tc['mobility_N'],
        'D_e_N': tc['diffusion_N'],
        'rate_coefficients': tc['rate_coefficients'],
        'EN_Td': en_td,
        'N_gas': N_gas,
    }


def run_bolos_fallback(state, config):
    """Run M09 using bolos two-term Boltzmann solver directly (fallback).

    Parameters
    ----------
    state : dict
        Pipeline state.
    config : SimulationConfig
        Configuration.

    Returns
    -------
    dict
        Same output format as run_pinn_inference.
    """
    from ..chemistry.cross_sections import CrossSectionSet
    from ..solvers.boltzmann_pinn import compute_transport_coefficients

    boltz_cfg = config.get('boltzmann', {})
    cs_dir = boltz_cfg.get('cross_section_dir', 'data/cross_sections/ar')
    gas = boltz_cfg.get('gas_system', 'ar')
    cs = CrossSectionSet(gas, cs_dir)

    N_gas = compute_gas_density(config)
    en_td = boltz_cfg.get('EN_Td', 50.0)

    # Use bolos solver
    import warnings
    from bolos import solver as bsolver, grid as bgrid

    ME = 9.10938e-31
    MAR = 39.948 * 1.6605e-27
    mass_ratio = ME / MAR

    gr = bgrid.QuadraticGrid(0, 60, 200)
    slv = bsolver.BoltzmannSolver(gr)

    cs_elastic = cs.elastic
    slv.add_process(
        kind='ELASTIC', target='Ar', mass_ratio=mass_ratio,
        data=np.column_stack([cs_elastic.energy, cs_elastic.sigma]),
    )
    for name, csi in cs.inelastic.items():
        kind = 'IONIZATION' if csi.process_type == 'ionization' else 'EXCITATION'
        slv.add_process(
            kind=kind, target='Ar', mass_ratio=mass_ratio,
            threshold=csi.threshold,
            data=np.column_stack([csi.energy, csi.sigma]),
        )

    slv.set_density('Ar', 1.0)
    slv.kT = 0.025
    slv.EN = en_td * 1e-21
    slv.init()

    f0 = slv.maxwell(max(0.5, 0.05 * en_td))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = slv.converge(f0, maxn=300, rtol=1e-4)

    eps_grid = slv.grid.c
    eedf = f

    # Compute transport coefficients using our function
    tc = compute_transport_coefficients(eedf, eps_grid, cs)

    logger.info(f"M09 (bolos): Te={tc['Te']:.3f} eV, <eps>={tc['mean_energy']:.3f} eV")

    return {
        'eedf': eedf,
        'eedf_energy_grid': eps_grid,
        'Te': tc['Te'],
        'mean_energy': tc['mean_energy'],
        'mu_e_N': tc['mobility_N'],
        'D_e_N': tc['diffusion_N'],
        'rate_coefficients': tc['rate_coefficients'],
        'EN_Td': en_td,
        'N_gas': N_gas,
    }


# --- Pipeline interface ---

def run(state, config):
    """Pipeline-compatible entry point for M09 Boltzmann Solver.

    Dispatches to PINN inference or bolos fallback based on config.

    Config section:
        boltzmann:
          backend: pinn | bolos     # solver backend (default: pinn)
          mode: inference           # always inference in pipeline
          gas_system: ar            # gas species
          EN_Td: 50.0              # E/N for 0-D mode [Td]
          cross_section_dir: data/cross_sections/ar/
          inference:
            model_path: data/boltzmann_models/ar_pinn.pt
            energy_max_eV: 60.0
            spatially_resolved: false
    """
    boltz_cfg = config.get('boltzmann', {})
    backend = boltz_cfg.get('backend', 'pinn')

    logger.info(f"M09 Boltzmann Solver: backend={backend}")

    if backend == 'pinn':
        try:
            return run_pinn_inference(state, config)
        except Exception as e:
            logger.warning(f"PINN inference failed ({e}), falling back to bolos")
            return run_bolos_fallback(state, config)
    elif backend == 'bolos':
        return run_bolos_fallback(state, config)
    else:
        raise ValueError(f"Unknown Boltzmann backend: {backend}")
