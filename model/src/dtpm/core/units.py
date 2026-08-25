"""Centralized physical constants for DTPM simulations. All SI units."""

import numpy as np


class PhysicalConstants:
    """Physical constants used across all DTPM modules."""

    # Fundamental constants
    mu_0 = 4e-7 * np.pi          # Vacuum permeability [H/m]
    epsilon_0 = 8.854187817e-12   # Vacuum permittivity [F/m]
    c = 2.99792458e8              # Speed of light [m/s]
    e = 1.602176634e-19           # Elementary charge [C]
    q_e = -1.602176634e-19        # Electron charge [C]
    m_e = 9.1093837015e-31        # Electron mass [kg]
    m_p = 1.67262192369e-27       # Proton mass [kg]
    k_B = 1.380649e-23            # Boltzmann constant [J/K]
    h = 6.62607015e-34            # Planck constant [J·s]

    # Derived constants
    eV = 1.602176634e-19          # 1 eV in Joules
    amu = 1.66053906660e-27       # Atomic mass unit [kg]

    @classmethod
    def ion_mass(cls, species='Ar'):
        """Return ion mass for common species."""
        masses = {
            'H': 1.008 * cls.amu,
            'H2': 2.016 * cls.amu,
            'He': 4.003 * cls.amu,
            'N': 14.007 * cls.amu,
            'N2': 28.014 * cls.amu,
            'O': 15.999 * cls.amu,
            'O2': 31.998 * cls.amu,
            'Ar': 39.948 * cls.amu,
            'SF6': 146.055 * cls.amu,
            'SF5': 127.055 * cls.amu,
            'SF4': 108.055 * cls.amu,
            'SF3': 89.055 * cls.amu,
            'F': 18.998 * cls.amu,
            'F2': 37.997 * cls.amu,
        }
        return masses.get(species, 39.948 * cls.amu)
