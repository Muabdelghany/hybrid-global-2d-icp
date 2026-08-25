"""Configuration management for DTPM simulations.

Extended for Phase 1: includes IPI ICP reactor geometry, wall chemistry,
operating conditions, and Picard coupling parameters.
"""

import os
import yaml
import numpy as np
from copy import deepcopy


def _coerce_numerics(obj):
    """Recursively convert string values that look like numbers to float/int.
    PyYAML safe_load misparses scientific notation like '4.0e7' as strings.
    """
    if isinstance(obj, dict):
        return {k: _coerce_numerics(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_numerics(v) for v in obj]
    if isinstance(obj, str):
        try:
            f = float(obj)
            return int(f) if f == int(f) and 'e' not in obj.lower() and '.' not in obj else f
        except ValueError:
            return obj
    return obj


class SimulationConfig:
    """Load and manage simulation configuration from YAML files."""

    def __init__(self, config_path=None):
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self._config = _coerce_numerics(yaml.safe_load(f))
        else:
            self._config = self._default_config()

    def _default_config(self):
        """Return default configuration for the merged EM + chemistry model."""
        return {
            'grid': {
                'x_min': 0,
                'x_max': 0.105,   # R_proc = 105 mm
                'y_min': 0,
                'y_max': 0.220,   # Domain height = 220 mm
                'dx': 2e-3,       # 2 mm
                'dy': 3e-3,       # 3 mm
            },
            'mesh_refinement': {
                'enabled': False,
                'gradient_threshold': 2.0,
                'max_refinement_level': 2,
                'min_dx': 0.25e-3,
                'min_dy': 0.25e-3,
            },
            'circuit': {
                'source_power': 700,       # Watts
                'source_frequency': 40.0e6, # Hz (IPI ICP = 40 MHz)
                'impedance': 50,           # Ohms
            },
            'geometry': {
                'coil_radius': 1e-3,       # 1 mm wire radius (2 mm diameter)
                'num_coils': 6,
                'coil_start_z': 0.145,     # z position of first coil (y=145mm)
                'coil_spacing': 0.010,     # 10 mm pitch
                'coil_r_position': 0.0405, # coil at r=40.5mm
                'coil_start_y': 0.145,     # alias for M01-M04
                'left_coil_x': 0.0405,     # r = R_coil
                'right_coil_x': 0.0405,    # same (axisymmetric)
            },
            'reactor_geometry': {
                'R_icp': 0.038,            # ICP source radius (m)
                'R_proc': 0.105,           # Processing chamber radius (m)
                'L_icp': 0.150,            # ICP source height (m) (150 mm)
                'L_proc': 0.050,           # Processing chamber height (m)
                'L_apt': 0.002,            # Aperture gap height (m)
                'Nr': 50,                  # Radial cells
                'Nz': 80,                  # Axial cells
                'beta_r': 1.2,             # Radial mesh stretching
                'beta_z': 1.0,             # Axial stretching (uniform)
            },
            'operating': {
                'pressure_mTorr': 10,
                'frac_Ar': 0.0,
                'Q_sccm': 100,
                'Tgas': 313,
                # eta is computed self-consistently from the m01 circuit model
                # (see docs/CODE_REVIEW_ULTRAREVIEW.md) — no eta_initial here.
            },
            'wall_chemistry': {
                'gamma_quartz': 0.001,
                'gamma_Al': 0.18,          # Calibrated to 74% [F] drop
                'gamma_wafer': 0.025,
                'gamma_window': 0.001,
            },
            'coupling': {
                'max_picard_iter': 20,
                'picard_tol': 0.02,
                'under_relax_ne': 0.3,
                'under_relax_Te': 0.3,
                'inner_chem_iter': 60,
                'inner_chem_relax': 0.12,
                'rerun_fdtd_every': 5,
            },
            'simulation': {
                'fdtd_rf_cycles': 2,
                'max_iter': 10000,
                'tolerance': 1e-2,
                'relaxation_omega': 1.8,
                'dt': 1e-10,
                'Np_e': 5000,
                'Np_i': 5000,
                'pic_steps': 200,
                'testing_mode': False,
            },
            'chemistry': {
                'gas_system': 'sf6',
                'pressure_Pa': 1.333,
                'gas_temperature_K': 313,
            },
            'boundary_conditions': {
                'dirichlet_lines': [],
            },
        }

    @property
    def grid(self):
        return self._config['grid']

    @property
    def circuit(self):
        return self._config['circuit']

    @property
    def geometry(self):
        return self._config['geometry']

    @property
    def simulation(self):
        return self._config['simulation']

    @property
    def mesh_refinement(self):
        return self._config.get('mesh_refinement', {})

    @property
    def chemistry(self):
        return self._config.get('chemistry', {})

    @property
    def boundary_conditions(self):
        return self._config.get('boundary_conditions', {})

    # Fields that identify the reactor-geometry section regardless of what it is called.
    _GEOM_SIGNATURE = frozenset(('R_icp', 'R_proc'))

    @property
    def reactor_geometry(self):
        sec = self._config.get('reactor_geometry')
        if sec:
            return sec
        # Configs written before this section was renamed use a different key. Find it by
        # its contents rather than by name, so the old name need not be recorded here.
        for value in self._config.values():
            if isinstance(value, dict) and self._GEOM_SIGNATURE <= set(value):
                return value
        return {}

    @property
    def wall_chemistry(self):
        return self._config.get('wall_chemistry', {})

    @property
    def operating(self):
        return self._config.get('operating', {})

    @property
    def coupling(self):
        return self._config.get('coupling', {})

    def get(self, key, default=None):
        return self._config.get(key, default)

    def to_simulation_params(self):
        """Convert to legacy simulation_params dict format."""
        from .units import PhysicalConstants as PC
        params = deepcopy(self._config)
        params['physics'] = {
            'mu_0': PC.mu_0, 'epsilon_0': PC.epsilon_0,
            'permittivity': PC.epsilon_0, 'q_e': PC.q_e,
            'm_e': PC.m_e, 'k_B': PC.k_B,
            'q_i': PC.e, 'm_i': PC.m_p,
        }
        return params

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

    def __repr__(self):
        rgeom = self.reactor_geometry
        return (f"SimulationConfig(IPI ICP {rgeom.get('Nr',50)}x{rgeom.get('Nz',80)}, "
                f"P={self.circuit['source_power']}W)")
