"""Data management for DTPM simulation results."""

import os
import json
import numpy as np
from datetime import datetime


class DataManager:
    """Manages simulation result I/O with organized directory structure."""

    def __init__(self, base_dir='results'):
        """Initialize with base results directory."""
        self.base_dir = base_dir
        self.run_dir = None

    def create_run_directory(self, config=None):
        """Create a timestamped directory for this simulation run."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(self.base_dir, 'runs', timestamp)

        # Create subdirectories
        for subdir in ['data', 'plots', 'logs']:
            os.makedirs(os.path.join(self.run_dir, subdir), exist_ok=True)

        # Save config if provided
        if config is not None:
            config_path = os.path.join(self.run_dir, 'config.yaml')
            config.save(config_path)

        return self.run_dir

    def save_array(self, data, name, subdir='data'):
        """Save numpy array to the run directory."""
        save_dir = os.path.join(self.run_dir or self.base_dir, subdir)
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f'{name}.npy'), data)

    def load_array(self, name, subdir='data'):
        """Load numpy array from the run directory."""
        load_dir = os.path.join(self.run_dir or self.base_dir, subdir)
        return np.load(os.path.join(load_dir, f'{name}.npy'))

    def save_dict(self, data, name, subdir='data'):
        """Save dictionary (JSON-serializable) to the run directory."""
        save_dir = os.path.join(self.run_dir or self.base_dir, subdir)
        os.makedirs(save_dir, exist_ok=True)

        # Convert numpy arrays to lists for JSON
        serializable = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                np.save(os.path.join(save_dir, f'{name}_{k}.npy'), v)
                serializable[k] = f'{name}_{k}.npy'
            else:
                serializable[k] = v

        with open(os.path.join(save_dir, f'{name}.json'), 'w') as f:
            json.dump(serializable, f, indent=2, default=str)

    @property
    def plots_dir(self):
        """Return the plots directory for the current run."""
        d = os.path.join(self.run_dir or self.base_dir, 'plots')
        os.makedirs(d, exist_ok=True)
        return d
