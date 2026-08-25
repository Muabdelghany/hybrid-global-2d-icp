"""
Particle-in-Cell (PIC) solver primitives.

Provides:
- cic_deposit: Cloud-in-Cell charge deposition
- cic_gather: Cloud-in-Cell field interpolation to particles
- boris_push_2d3v: Boris leap-frog particle pusher (2D positions, 3D velocities)
- apply_periodic_bc: Periodic boundary conditions
- apply_reflecting_bc: Specular reflection boundary conditions

These are the core PIC building blocks used by m07_particle_kinetics.
The Boris pusher handles arbitrary 3D magnetic fields (Bx, By, Bz),
enabling correct gyration in the in-plane B-field of the ICP reactor.

References:
    - Birdsall & Langdon, Plasma Physics via Computer Simulation (2004)
    - Hockney & Eastwood, Computer Simulation Using Particles (1988)
"""

import numpy as np


def cic_deposit(x, y, q, nx, ny, dx, dy):
    """
    Deposit particle charges onto a 2D grid using Cloud-in-Cell (CIC) weighting.

    Args:
        x, y: Particle positions (arrays of length Np).
        q: Charge per macroparticle.
        nx, ny: Grid dimensions.
        dx, dy: Grid spacing.

    Returns:
        rho: Charge density on the grid (nx, ny).
    """
    rho = np.zeros((nx, ny))
    ix = x / dx - 0.5
    iy = y / dy - 0.5
    i = np.floor(ix).astype(int) % nx
    j = np.floor(iy).astype(int) % ny
    wx = ix - np.floor(ix)
    wy = iy - np.floor(iy)

    i1 = (i + 1) % nx
    j1 = (j + 1) % ny

    np.add.at(rho, (i, j), q * (1 - wx) * (1 - wy))
    np.add.at(rho, (i1, j), q * wx * (1 - wy))
    np.add.at(rho, (i, j1), q * (1 - wx) * wy)
    np.add.at(rho, (i1, j1), q * wx * wy)

    rho /= (dx * dy)
    return rho


def cic_gather(x, y, field, nx, ny, dx, dy):
    """
    Gather field values at particle positions using CIC weights.

    Args:
        x, y: Particle positions.
        field: 2D field array (nx, ny).
        nx, ny: Grid dimensions.
        dx, dy: Grid spacing.

    Returns:
        f_p: Field values at each particle position.
    """
    ix = x / dx - 0.5
    iy = y / dy - 0.5
    i = np.floor(ix).astype(int) % nx
    j = np.floor(iy).astype(int) % ny
    wx = ix - np.floor(ix)
    wy = iy - np.floor(iy)

    i1 = (i + 1) % nx
    j1 = (j + 1) % ny

    f_p = (
        field[i, j] * (1 - wx) * (1 - wy)
        + field[i1, j] * wx * (1 - wy)
        + field[i, j1] * (1 - wx) * wy
        + field[i1, j1] * wx * wy
    )
    return f_p


def boris_push_2d3v(x, y, vx, vy, vz, q, m,
                    Ex_p, Ey_p, Bx_p, By_p, Bz_p, dt):
    """
    Boris leap-frog particle pusher for 2D3V simulations.

    Handles full 3D magnetic field (Bx, By, Bz) with 3D velocities
    while keeping positions in 2D. This correctly captures gyration
    in arbitrary B-field orientations.

    The Boris algorithm is symplectic: it exactly conserves energy
    in a uniform magnetic field (no E-field).

    Updates positions and velocities in-place.

    Args:
        x, y: Particle positions (2D, modified in-place).
        vx, vy, vz: Particle velocities (3D, modified in-place).
        q: Particle charge [C].
        m: Particle mass [kg].
        Ex_p, Ey_p: Electric field at particle positions [V/m].
        Bx_p, By_p, Bz_p: Magnetic field at particle positions [T].
        dt: Time step [s].
    """
    qmdt2 = q / m * dt * 0.5

    # --- Half electric kick ---
    vx += qmdt2 * Ex_p
    vy += qmdt2 * Ey_p
    # No Ez component in source-free 2D

    # --- Magnetic rotation (Boris rotation) ---
    # t = (q*dt / 2m) * B
    tx = qmdt2 * Bx_p
    ty = qmdt2 * By_p
    tz = qmdt2 * Bz_p
    t2 = tx * tx + ty * ty + tz * tz

    # s = 2t / (1 + |t|^2)
    sx = 2 * tx / (1 + t2)
    sy = 2 * ty / (1 + t2)
    sz = 2 * tz / (1 + t2)

    # v_minus = v after first half kick (already done above)
    vx_minus = vx.copy()
    vy_minus = vy.copy()
    vz_minus = vz.copy()

    # v' = v_minus + v_minus x t
    vx_prime = vx_minus + (vy_minus * tz - vz_minus * ty)
    vy_prime = vy_minus + (vz_minus * tx - vx_minus * tz)
    vz_prime = vz_minus + (vx_minus * ty - vy_minus * tx)

    # v_plus = v_minus + v' x s
    vx[:] = vx_minus + (vy_prime * sz - vz_prime * sy)
    vy[:] = vy_minus + (vz_prime * sx - vx_prime * sz)
    vz[:] = vz_minus + (vx_prime * sy - vy_prime * sx)

    # --- Second half electric kick ---
    vx += qmdt2 * Ex_p
    vy += qmdt2 * Ey_p

    # --- Position update (drift) — 2D only ---
    x += vx * dt
    y += vy * dt


# Legacy API compatibility
def boris_push(x, y, vx, vy, q, m, Ex_p, Ey_p, Bz_p, dt):
    """Legacy 2D Boris pusher (Bz only). Use boris_push_2d3v for full 3D B."""
    vz = np.zeros_like(vx)
    Bx_p = np.zeros_like(Bz_p)
    By_p = np.zeros_like(Bz_p)
    boris_push_2d3v(x, y, vx, vy, vz, q, m, Ex_p, Ey_p, Bx_p, By_p, Bz_p, dt)


def apply_periodic_bc(x, y, Lx, Ly):
    """Apply periodic boundary conditions."""
    x[:] = np.mod(x, Lx)
    y[:] = np.mod(y, Ly)


def apply_reflecting_bc(x, y, vx, vy, x_min, x_max, y_min, y_max):
    """
    Apply specular reflection boundary conditions.

    Particles hitting a wall are reflected: position mirrored and
    normal velocity component reversed.

    Args:
        x, y: Particle positions (modified in-place).
        vx, vy: Particle velocities (modified in-place).
        x_min, x_max, y_min, y_max: Domain boundaries [m].
    """
    # Left wall
    mask = x < x_min
    x[mask] = 2 * x_min - x[mask]
    vx[mask] *= -1

    # Right wall
    mask = x > x_max
    x[mask] = 2 * x_max - x[mask]
    vx[mask] *= -1

    # Bottom wall
    mask = y < y_min
    y[mask] = 2 * y_min - y[mask]
    vy[mask] *= -1

    # Top wall
    mask = y > y_max
    y[mask] = 2 * y_max - y[mask]
    vy[mask] *= -1
