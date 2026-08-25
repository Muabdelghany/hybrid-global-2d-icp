"""
Reactor geometry — coil positions, IPI ICP T-shaped geometry mask.

Combines:
- Coil geometry computation (from DTPM EM pipeline)
- IPI ICP reactor geometry mask and boundary classification (from Stage 10)
"""
import numpy as np

# -- Boundary type constants (IPI ICP reactor) --
BC_INTERIOR = 0
BC_AXIS     = 1
BC_QUARTZ   = 2
BC_WINDOW   = 3
BC_AL_SIDE  = 4
BC_AL_TOP   = 5
BC_WAFER    = 6
BC_SHOULDER = 7
BC_INACTIVE = -1


# ===============================================================
# Coil geometry (from EM pipeline, used by M01-M04)
# ===============================================================

def compute_coil_geometry(config, grid):
    """
    Compute coil center positions in both physical and grid-index coordinates.

    Args:
        config: SimulationConfig with geometry parameters.
        grid: Grid instance (Cartesian) or Mesh2D (cylindrical).

    Returns:
        dict with keys:
            coil_centers_m: List of (x, y) in meters.
            coil_centers_idx: List of (i, j) grid indices.
            coil_radius: Physical coil radius [m].
            coil_radius_idx: Coil radius in grid points.
    """
    geom = config.geometry if hasattr(config, 'geometry') else config.get('geometry', {})
    dx = grid.dx if hasattr(grid, 'dx') else grid.dr.mean()
    dy = grid.dy if hasattr(grid, 'dy') else grid.dz.mean()

    coil_radius = geom['coil_radius']
    coil_radius_idx = int(coil_radius / dx)
    num_coils = geom['num_coils']
    start_y = geom['coil_start_y']
    spacing = geom['coil_spacing']
    left_x = geom['left_coil_x']
    right_x = geom['right_coil_x']

    coil_centers_m = []
    for i in range(num_coils):
        y_pos = start_y + i * spacing
        coil_centers_m.append((left_x, y_pos))
        coil_centers_m.append((right_x, y_pos))

    coil_centers_idx = []
    for x_m, y_m in coil_centers_m:
        coil_centers_idx.append((int(x_m / dx), int(y_m / dy)))

    return {
        'coil_centers_m': coil_centers_m,
        'coil_centers_idx': coil_centers_idx,
        'coil_radius': coil_radius,
        'coil_radius_idx': coil_radius_idx,
    }


def compute_coil_positions_cylindrical(config, mesh, inside=None):
    """Compute coil positions at the physical coil radius r = R_coil.

    The physical ICP coils are wound outside the quartz tube at r = R_coil
    (typically 40.5 mm for the IPI ICP reactor). The FDTD source is placed at
    the mesh cell closest to this physical coil radius.

    For this to work, the mesh must extend to at least R_coil, and the
    FDTD must be run on an EM-active mask that includes the quartz wall
    and coil region (see build_em_active_mask).

    Returns:
        list of (i, j) index pairs for coil positions on the mesh.
    """
    geom = config.geometry if hasattr(config, 'geometry') else config.get('geometry', {})
    rgeom = config.reactor_geometry if hasattr(config, 'reactor_geometry') else config.get('reactor_geometry', {})

    R_coil = geom.get('coil_r_position', 0.0405)  # Physical coil radius [m]
    num_coils = geom.get('num_coils', 6)
    coil_start_z = geom.get('coil_start_z', rgeom.get('L_proc', 0.050) + rgeom.get('L_apt', 0.002) + 0.020)
    coil_spacing = geom.get('coil_spacing', 0.025)

    # Find mesh cell closest to the physical coil radius
    i_coil = int(np.argmin(np.abs(mesh.rc - R_coil)))

    coil_positions = []
    z_apt_top = rgeom.get('L_proc', 0.050) + rgeom.get('L_apt', 0.002)
    for n in range(num_coils):
        z_coil = coil_start_z + n * coil_spacing
        j_coil = int(np.argmin(np.abs(mesh.zc - z_coil)))
        if mesh.zc[j_coil] >= z_apt_top:
            coil_positions.append((i_coil, j_coil))

    return coil_positions


def build_em_active_mask(mesh, inside, bc_type, R_icp, R_outer, z_apt_top, z_top):
    """Build an extended mask for EM (FDTD) that includes the quartz wall and coil region.

    The gas volume mask (inside) only covers r < R_icp in the ICP region.
    The EM solver needs to propagate fields through the quartz wall and
    up to the coil position. This function extends the active region to
    include cells from R_icp to R_outer in the ICP source z-range.

    Parameters
    ----------
    mesh : Mesh2D
    inside : ndarray (Nr, Nz), bool — gas volume mask
    bc_type : ndarray (Nr, Nz), int
    R_icp : float — inner quartz wall radius [m]
    R_outer : float — outer extent for EM (beyond coil) [m]
    z_apt_top : float — bottom of ICP source [m]
    z_top : float — top of ICP source [m]

    Returns
    -------
    em_active : ndarray (Nr, Nz), bool
        True for cells where EM fields should be computed.
    eps_r_map : ndarray (Nr, Nz), float
        Relative permittivity: 1.0 in gas/air, 3.8 in quartz.
    """
    Nr, Nz = mesh.Nr, mesh.Nz
    em_active = inside.copy()
    eps_r_map = np.ones((Nr, Nz))

    qz_thick = 0.0005  # 0.5 mm quartz wall (from reactor specification)
    R_qz_outer = R_icp + qz_thick

    for i in range(Nr):
        r = mesh.rc[i]
        for j in range(Nz):
            z = mesh.zc[j]
            if z >= z_apt_top and z <= z_top:
                # Quartz wall region: r in [R_icp, R_icp + 0.5mm]
                if R_icp <= r <= R_qz_outer:
                    em_active[i, j] = True
                    eps_r_map[i, j] = 3.8
                # Air gap + coil region: r in [R_qz_outer, R_outer]
                elif R_qz_outer < r <= R_outer:
                    em_active[i, j] = True
                    eps_r_map[i, j] = 1.0  # vacuum/air

    return em_active, eps_r_map


# ===============================================================
# IPI ICP reactor T-shaped geometry (from Stage 10)
# ===============================================================

def build_geometry_mask(mesh, R_icp, R_proc, z_apt_bot, z_apt_top, z_top,
                        R_wafer=0.075):
    """Build boolean mask and boundary classification for the IPI ICP geometry.

    Parameters
    ----------
    mesh : Mesh2D
        Structured (r,z) mesh covering [0, R_proc] x [0, z_top].
    R_icp : float
        ICP source radius (m).
    R_proc : float
        Processing region radius (m).
    z_apt_bot : float
        Bottom of aperture plate = top of processing region (m).
    z_apt_top : float
        Top of aperture plate = bottom of ICP source (m).
    z_top : float
        Top of ICP source = dielectric window (m).
    R_wafer : float, optional
        Wafer radius (m). Default 0.075 (6-inch Si wafer, reactor specification).
        Splits the j=0 bottom boundary into r <= R_wafer (BC_WAFER, Si) and
        R_wafer < r <= R_proc (BC_AL_TOP, Al pedestal-top / chamber-floor
        annulus). An earlier revision labelled the entire j=0 row BC_WAFER,
        under-counting the Al F-sink by ~50 % of the bottom surface area.

    Returns
    -------
    inside : ndarray (Nr, Nz), bool
        True for cells inside the gas volume.
    bc_type : ndarray (Nr, Nz), int
        Boundary classification for each cell.
    """
    Nr, Nz = mesh.Nr, mesh.Nz
    inside = np.zeros((Nr, Nz), dtype=bool)
    bc_type = np.full((Nr, Nz), BC_INACTIVE, dtype=int)

    # Step 1: Mark active cells
    for i in range(Nr):
        r = mesh.rc[i]
        for j in range(Nz):
            z = mesh.zc[j]
            in_icp = (r <= R_icp) and (z >= z_apt_top)
            in_apt = (r <= R_icp) and (z_apt_bot <= z < z_apt_top)
            in_proc = (r <= R_proc) and (z < z_apt_bot)
            if in_icp or in_apt or in_proc:
                inside[i, j] = True
                bc_type[i, j] = BC_INTERIOR

    # Step 2: Classify boundary cells
    for i in range(Nr):
        for j in range(Nz):
            if not inside[i, j]:
                continue
            r, z = mesh.rc[i], mesh.zc[j]

            # Axis of symmetry
            if i == 0:
                bc_type[i, j] = BC_AXIS

            # Radial outer boundary
            if i == Nr - 1 and z < z_apt_bot:
                bc_type[i, j] = BC_AL_SIDE
            elif i < Nr - 1 and not inside[i + 1, j]:
                if z >= z_apt_top:
                    bc_type[i, j] = BC_QUARTZ
                elif z >= z_apt_bot:
                    bc_type[i, j] = BC_SHOULDER
                else:
                    bc_type[i, j] = BC_AL_SIDE

            # Top boundary
            if j == Nz - 1:
                bc_type[i, j] = BC_WINDOW

            # Bottom boundary: split wafer (Si) vs chamber-floor annulus (Al)
            if j == 0:
                if r <= R_wafer:
                    bc_type[i, j] = BC_WAFER
                else:
                    bc_type[i, j] = BC_AL_TOP

            # Aperture plate underside
            if z < z_apt_bot and r > R_icp:
                if j < Nz - 1 and not inside[i, j + 1]:
                    bc_type[i, j] = BC_AL_TOP

    return inside, bc_type


def build_index_maps(inside):
    """Build mapping between (i,j) grid positions and active-cell flat indices.

    Returns
    -------
    ij_to_flat : ndarray (Nr, Nz), int
        Maps (i,j) -> flat index. -1 for inactive cells.
    flat_to_ij : list of (i, j) tuples
        Maps flat index -> (i,j).
    n_active : int
        Number of active cells.
    """
    Nr, Nz = inside.shape
    ij_to_flat = np.full((Nr, Nz), -1, dtype=int)
    flat_to_ij = []
    k = 0
    for i in range(Nr):
        for j in range(Nz):
            if inside[i, j]:
                ij_to_flat[i, j] = k
                flat_to_ij.append((i, j))
                k += 1
    return ij_to_flat, flat_to_ij, k


def count_boundary_cells(bc_type):
    """Count cells of each boundary type for diagnostics."""
    names = {
        BC_INTERIOR: 'Interior', BC_AXIS: 'Axis', BC_QUARTZ: 'Quartz',
        BC_WINDOW: 'Window', BC_AL_SIDE: 'Al side', BC_AL_TOP: 'Al top',
        BC_WAFER: 'Wafer', BC_SHOULDER: 'Shoulder', BC_INACTIVE: 'Inactive'
    }
    counts = {}
    for bc_val, name in names.items():
        n = int(np.sum(bc_type == bc_val))
        if n > 0:
            counts[name] = n
    return counts
