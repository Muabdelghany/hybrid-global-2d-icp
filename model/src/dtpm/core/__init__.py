"""Core infrastructure for DTPM simulation framework."""
from .config import SimulationConfig
from .units import PhysicalConstants
from .grid import Grid
from .mesh import Mesh2D
from .geometry import (
    compute_coil_geometry,
    compute_coil_positions_cylindrical,
    build_geometry_mask,
    build_em_active_mask,
    build_index_maps,
    count_boundary_cells,
    BC_INTERIOR, BC_AXIS, BC_QUARTZ, BC_WINDOW,
    BC_AL_SIDE, BC_AL_TOP, BC_WAFER, BC_SHOULDER, BC_INACTIVE,
)
