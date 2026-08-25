"""
Module 07 — Particle Kinetics (2D3V PIC)
==========================================
Simulates charged particle dynamics in the ICP reactor using a
Particle-in-Cell (PIC) scheme with Boris pusher.

Physics:
- 2D3V: 2D spatial positions (r, z), 3D velocity components (vr, vz, vtheta)
- Lorentz force: F = q(E + v x B)
- Boris leap-frog integrator (symplectic, energy-conserving in pure B)
- Cloud-in-Cell (CIC) for charge deposition and field gathering
- Self-consistent space-charge via FFT Poisson solver
- External fields from EM pipeline (M02/M04/M05)
- Reflecting boundary conditions (ICP reactor is bounded)
- Maxwellian velocity initialization at configurable Te

Electron and ion species are tracked independently:
- Electrons: small mass, high thermal speed, fast gyration
- Ions (Ar+/SF6+ etc.): large mass, slow response

ML Interface:
- Particle phase-space data for kinetic neural operator training
- KE history for energy conservation monitoring
- Charge density rho(r,z) for fluid model comparison

References:
    - Birdsall & Langdon, Plasma Physics via Computer Simulation (2004)
    - Verboncoeur, J. Comp. Phys. 174, 421-427 (2005)
"""

import os
import numpy as np
import logging

logger = logging.getLogger(__name__)


def initialize_particles(Np, x_min, x_max, y_min, y_max, Te_eV, mass, charge):
    """
    Initialize particle positions (uniform) and velocities (Maxwellian).

    Args:
        Np: Number of macroparticles.
        x_min, x_max, y_min, y_max: Domain boundaries [m].
        Te_eV: Initial temperature [eV].
        mass: Particle mass [kg].
        charge: Particle charge [C] (sign matters).

    Returns:
        x, y: Positions [m].
        vx, vy, vz: Velocities [m/s].
    """
    from ..core.units import PhysicalConstants as PC

    x = np.random.uniform(x_min, x_max, Np)
    y = np.random.uniform(y_min, y_max, Np)

    # Thermal speed: v_th = sqrt(kB * T / m)
    v_th = np.sqrt(PC.eV * Te_eV / mass)

    vx = np.random.normal(0, v_th, Np)
    vy = np.random.normal(0, v_th, Np)
    vz = np.random.normal(0, v_th, Np)

    logger.info(f"  Initialized {Np} particles: Te={Te_eV:.1f} eV, "
                f"v_th={v_th:.4e} m/s, mass={mass:.4e} kg")

    return x, y, vx, vy, vz


def run_pic(nx, ny, dx, dy, dt, n_steps,
            Ex_ext, Ey_ext, Bx_ext, By_ext,
            Np_e, Np_i, Te_init_eV,
            x_min, x_max, y_min, y_max,
            particle_bc='reflecting',
            gas_species='Ar'):
    """
    Run 2D3V electrostatic PIC simulation.

    Args:
        nx, ny: Grid dimensions.
        dx, dy: Grid spacing [m].
        dt: Time step [s].
        n_steps: Number of PIC steps.
        Ex_ext, Ey_ext: External E-field from M02/M05 (nx, ny) [V/m].
        Bx_ext, By_ext: External B-field from M04/M05 (nx, ny) [T].
        Np_e, Np_i: Number of electron/ion macroparticles.
        Te_init_eV: Initial electron temperature [eV].
        x_min, x_max, y_min, y_max: Domain boundaries [m].
        particle_bc: 'reflecting', 'periodic', or 'absorbing'.
        gas_species: Ion species name for mass lookup.

    Returns:
        dict with particle data and diagnostics.
    """
    from ..core.units import PhysicalConstants as PC
    from ..solvers.pic_solver import (
        cic_deposit, cic_gather, boris_push_2d3v,
        apply_periodic_bc, apply_reflecting_bc
    )
    from ..solvers.poisson import solve_poisson_fft

    Lx = x_max - x_min
    Ly = y_max - y_min
    m_i = PC.ion_mass(gas_species)

    logger.info(f"PIC: {Np_e} electrons, {Np_i} ions, {n_steps} steps, dt={dt:.4e} s")

    # Initialize particles
    xe, ye, vxe, vye, vze = initialize_particles(
        Np_e, x_min, x_max, y_min, y_max, Te_init_eV, PC.m_e, PC.q_e)
    xi, yi, vxi, vyi, vzi = initialize_particles(
        Np_i, x_min, x_max, y_min, y_max, Te_init_eV * PC.m_e / m_i, m_i, PC.e)

    # Macroparticle charge (ensure quasi-neutrality)
    q_macro_e = PC.q_e  # charge per macro-electron
    q_macro_i = PC.e    # charge per macro-ion

    # Diagnostics
    KE_history = []
    work_history = []

    for step in range(n_steps):
        # --- Charge deposition ---
        rho_e = cic_deposit(xe, ye, q_macro_e, nx, ny, dx, dy)
        rho_i = cic_deposit(xi, yi, q_macro_i, nx, ny, dx, dy)
        rho = rho_e + rho_i

        # --- Self-consistent E-field from space charge ---
        Ex_sc, Ey_sc = solve_poisson_fft(rho, dx, dy, PC.epsilon_0)

        # --- Total fields ---
        Ex_total = Ex_ext + Ex_sc
        Ey_total = Ey_ext + Ey_sc

        # --- Gather fields at electron positions ---
        Ex_e = cic_gather(xe, ye, Ex_total, nx, ny, dx, dy)
        Ey_e = cic_gather(xe, ye, Ey_total, nx, ny, dx, dy)
        Bx_e = cic_gather(xe, ye, Bx_ext, nx, ny, dx, dy)
        By_e = cic_gather(xe, ye, By_ext, nx, ny, dx, dy)
        Bz_e = np.zeros(Np_e)  # No out-of-plane B in this geometry

        # --- Gather fields at ion positions ---
        Ex_i = cic_gather(xi, yi, Ex_total, nx, ny, dx, dy)
        Ey_i = cic_gather(xi, yi, Ey_total, nx, ny, dx, dy)
        Bx_i = cic_gather(xi, yi, Bx_ext, nx, ny, dx, dy)
        By_i = cic_gather(xi, yi, By_ext, nx, ny, dx, dy)
        Bz_i = np.zeros(Np_i)

        # --- Track kinetic energy before push ---
        KE_e = 0.5 * PC.m_e * np.sum(vxe**2 + vye**2 + vze**2)
        KE_i = 0.5 * m_i * np.sum(vxi**2 + vyi**2 + vzi**2)
        KE_history.append(float(KE_e + KE_i))

        # --- Track work done by E-field ---
        dW_e = q_macro_e * np.sum(Ex_e * vxe + Ey_e * vye) * dt
        dW_i = q_macro_i * np.sum(Ex_i * vxi + Ey_i * vyi) * dt
        work_history.append(float(dW_e + dW_i))

        # --- Boris push ---
        boris_push_2d3v(xe, ye, vxe, vye, vze, PC.q_e, PC.m_e,
                        Ex_e, Ey_e, Bx_e, By_e, Bz_e, dt)
        boris_push_2d3v(xi, yi, vxi, vyi, vzi, PC.e, m_i,
                        Ex_i, Ey_i, Bx_i, By_i, Bz_i, dt)

        # --- Boundary conditions ---
        if particle_bc == 'periodic':
            apply_periodic_bc(xe, ye, Lx, Ly)
            apply_periodic_bc(xi, yi, Lx, Ly)
        elif particle_bc == 'reflecting':
            apply_reflecting_bc(xe, ye, vxe, vye, x_min, x_max, y_min, y_max)
            apply_reflecting_bc(xi, yi, vxi, vyi, x_min, x_max, y_min, y_max)
        # 'absorbing': particles that leave are lost (no action needed for now)

    # Final kinetic energy
    KE_e_final = 0.5 * PC.m_e * np.sum(vxe**2 + vye**2 + vze**2)
    KE_i_final = 0.5 * m_i * np.sum(vxi**2 + vyi**2 + vzi**2)
    KE_history.append(float(KE_e_final + KE_i_final))

    logger.info(f"PIC complete: KE_initial={KE_history[0]:.4e} J, "
                f"KE_final={KE_history[-1]:.4e} J")

    return {
        # Electron data
        'xe': xe, 'ye': ye, 'vxe': vxe, 'vye': vye, 'vze': vze,
        # Ion data
        'xi': xi, 'yi': yi, 'vxi': vxi, 'vyi': vyi, 'vzi': vzi,
        # Charge density
        'rho_e': rho_e, 'rho_i': rho_i,
        # Diagnostics
        'KE_history': np.array(KE_history),
        'work_history': np.array(work_history),
        'Np_e': Np_e, 'Np_i': Np_i,
    }


# --- Pipeline interface ---

def run(state, config):
    """Pipeline-compatible entry point for M07."""
    from ..core.units import PhysicalConstants as PC

    sim = config.simulation if hasattr(config, 'simulation') else config.get('simulation', {})
    grid_cfg = config.grid if hasattr(config, 'grid') else config.get('grid', {})

    nx, ny = state['nx'], state['ny']
    dx, dy = state['dx'], state['dy']

    # External fields from upstream modules
    Ex_ext = state.get('Ex', np.zeros((nx, ny)))
    Ey_ext = state.get('Ey', np.zeros((nx, ny)))
    Bx_ext = state.get('Bs_x', np.zeros((nx, ny)))
    By_ext = state.get('Bs_y', np.zeros((nx, ny)))

    result = run_pic(
        nx=nx, ny=ny, dx=dx, dy=dy,
        dt=sim.get('dt', 1e-10),
        n_steps=sim.get('pic_steps', 200),
        Ex_ext=Ex_ext, Ey_ext=Ey_ext,
        Bx_ext=Bx_ext, By_ext=By_ext,
        Np_e=sim.get('Np_e', 5000),
        Np_i=sim.get('Np_i', 5000),
        Te_init_eV=sim.get('Te_init_eV', 3.0),
        x_min=grid_cfg.get('x_min', 0.0),
        x_max=grid_cfg.get('x_max', 0.32),
        y_min=grid_cfg.get('y_min', 0.0),
        y_max=grid_cfg.get('y_max', 0.22),
        particle_bc=sim.get('particle_bc', 'reflecting'),
        gas_species=config.chemistry.get('gas_system', 'Ar')
        if hasattr(config, 'chemistry') else 'Ar',
    )

    return result
