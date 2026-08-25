"""Numerical solvers for DTPM (Poisson, FDTD, PIC, Boltzmann, Fluid)."""
from .poisson import solve_poisson_sor, solve_poisson_fft

# Boltzmann PINN is an optional dependency (requires torch). Load lazily so
# the rest of the solvers package remains usable without PyTorch installed.
try:
    from .boltzmann_pinn import (
        BoltzmannPINN,
        BoltzmannPINNTrainer,
        compute_transport_coefficients,
    )
except ImportError:  # torch not available
    BoltzmannPINN = None
    BoltzmannPINNTrainer = None
    compute_transport_coefficients = None
