"""Grid creation and management for DTPM simulations."""

import numpy as np


class Grid:
    """Spatial grid for the simulation domain."""

    def __init__(self, config):
        """
        Initialize grid from configuration.

        Args:
            config: SimulationConfig or dict with grid parameters.
        """
        if isinstance(config, dict):
            grid_cfg = config.get('grid', config)
        else:
            grid_cfg = config.grid

        self.x_min = grid_cfg['x_min']
        self.x_max = grid_cfg['x_max']
        self.y_min = grid_cfg['y_min']
        self.y_max = grid_cfg['y_max']
        self.dx = grid_cfg['dx']
        self.dy = grid_cfg['dy']

        # Compute grid dimensions
        self.nx = int((self.x_max - self.x_min) / self.dx) + 1
        self.ny = int((self.y_max - self.y_min) / self.dy) + 1

        # Create coordinate arrays
        self.x = np.linspace(self.x_min, self.x_max, self.nx)
        self.y = np.linspace(self.y_min, self.y_max, self.ny)

        # Recompute dx/dy from linspace for consistency
        if self.nx > 1:
            self.dx = self.x[1] - self.x[0]
        if self.ny > 1:
            self.dy = self.y[1] - self.y[0]

        # Create meshgrid (r,z coordinates for cylindrical ICP)
        self.r_grid, self.z_grid = np.meshgrid(self.x, self.y, indexing='ij')

    @property
    def Lx(self):
        return self.x_max - self.x_min

    @property
    def Ly(self):
        return self.y_max - self.y_min

    def refine(self, factor=2):
        """Return a refined grid with resolution increased by factor."""
        config = {
            'x_min': self.x_min, 'x_max': self.x_max,
            'y_min': self.y_min, 'y_max': self.y_max,
            'dx': self.dx / factor, 'dy': self.dy / factor,
        }
        return Grid(config)
