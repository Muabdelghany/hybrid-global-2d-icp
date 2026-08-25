"""
Standardized plotting and I/O utilities for DTPM simulations.
Provides publication-quality plotting functions for field data.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import logging

logger = logging.getLogger(__name__)

# Publication-quality defaults
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

# I/O flags
SAVE_NPY = True
SAVE_TXT = False


def save_results(data, filename, base_dir='results'):
    """Save simulation results to npy and/or txt format."""
    if SAVE_NPY:
        npy_dir = os.path.join(base_dir, 'data')
        os.makedirs(npy_dir, exist_ok=True)
        np.save(os.path.join(npy_dir, f'{filename}.npy'), data)

    if SAVE_TXT:
        txt_dir = os.path.join(base_dir, 'data')
        os.makedirs(txt_dir, exist_ok=True)
        if isinstance(data, np.ndarray):
            np.savetxt(os.path.join(txt_dir, f'{filename}.txt'), data.flatten())
        elif isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    np.savetxt(os.path.join(txt_dir, f'{filename}_{key}.txt'), value.flatten())
                else:
                    with open(os.path.join(txt_dir, f'{filename}_{key}.txt'), 'w') as f:
                        f.write(str(value))


def create_directories(sweep_potentials, base_dir='results'):
    """Create directory structure for simulation outputs."""
    for subdir in ['data', 'plots', 'plots/Coil_Geometry', 'plots/Coil_Potentials']:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
    for potential in sweep_potentials:
        pot_dir = os.path.join(base_dir, 'plots', 'Coil_Potentials', f'{potential:.2f}V')
        for sub in ['1.Initial_Condition', '2.Electrostatic', '3.Magnetostatic',
                     '4.Electromagnetic', '4.Electromagnetic/animations']:
            os.makedirs(os.path.join(pot_dir, sub), exist_ok=True)


def plot_coil_geometry(coil_centers, coil_radius, Lx, Ly, save_folder='results/plots/Coil_Geometry',
                       geometry_lines=None):
    """Plot the ICP coil geometry with reactor walls."""
    os.makedirs(save_folder, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal')

    for (xc, yc) in coil_centers:
        circle = plt.Circle((xc * 1e3, yc * 1e3), coil_radius * 1e3,
                             color='orange', fill=True, zorder=5)
        ax.add_patch(circle)

    add_geometry_lines(ax)

    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title('ICP Reactor Coil Geometry')
    ax.grid(True, alpha=0.1)
    ax.set_xlim(0, Lx * 1e3)
    ax.set_ylim(0, Ly * 1e3)
    plt.savefig(os.path.join(save_folder, 'coil_geometry.pdf'), bbox_inches='tight', dpi=300)
    plt.close()


def add_geometry_lines(ax):
    """Add ICP reactor wall geometry lines to a plot axis."""
    walls = [
        [(3, 50), (141, 50)],
        [(141, 50), (141, 200)],
        [(141, 200), (179, 200)],
        [(179, 200), (179, 50)],
        [(179, 50), (317, 50)],
    ]
    for line in walls:
        xs, ys = zip(*line)
        ax.plot(xs, ys, linewidth=0.8, color='blue')

    # Substrate line
    ax.plot([85, 235], [25, 25], linewidth=2, color='lightgreen')
    # Centerline
    ax.plot([160, 160], [10, 210], '-.', color='red', linewidth=0.5)


def plot_results(data, filename, save_folder='results/plots', xlabel='', ylabel='',
                 title='', x_data=None, y_data=None, cmap='jet', colorbar_label='',
                 show_geometry=True):
    """Generate and save a publication-quality plot of simulation data."""
    os.makedirs(save_folder, exist_ok=True)
    filepath = os.path.join(save_folder, f'{filename}.pdf')
    plt.figure(figsize=(10, 8))

    if isinstance(data, np.ndarray) and data.ndim == 2:
        data_plot = np.abs(data.T)
        if x_data is not None and y_data is not None:
            extent = [x_data[0], x_data[-1], y_data[0], y_data[-1]]
            im = plt.imshow(data_plot, extent=extent, origin='lower', cmap=cmap, aspect='auto')
            if show_geometry:
                add_geometry_lines(plt.gca())
        else:
            im = plt.imshow(data_plot, origin='lower', cmap=cmap, aspect='auto')
        cbar = plt.colorbar(im)
        if colorbar_label:
            cbar.set_label(colorbar_label, fontsize=14)
    elif isinstance(data, np.ndarray) and data.ndim == 1:
        plt.plot(x_data, data) if x_data is not None else plt.plot(data)
    else:
        plt.plot(x_data, data) if x_data is not None else plt.plot(data)

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=15)
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close()


def plot_vector_field(field_x, field_y, filename, title, x_data, y_data,
                      dx, dy, xlabel='x [mm]', ylabel='y [mm]',
                      field_type=None, save_folder=None, show_geometry=True):
    """Plot a vector field with arrows colored by magnitude."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes([0.1, 0.1, 0.75, 0.8])

    magnitude = np.sqrt(field_x**2 + field_y**2)
    x = np.linspace(x_data[0], x_data[-1], field_x.shape[0])
    y = np.linspace(y_data[0], y_data[-1], field_x.shape[1])
    X, Y = np.meshgrid(x, y, indexing='ij')

    skip = max(1, min(field_x.shape) // 25)
    xi = np.arange(0, field_x.shape[0], skip)
    yi = np.arange(0, field_x.shape[1], skip)
    Xq, Yq = np.meshgrid(x[xi], y[yi], indexing='ij')
    U = field_x[xi][:, yi]
    V = field_y[xi][:, yi]
    M = magnitude[xi][:, yi]

    scale = 20 if (field_type and 'electro' in field_type.lower()) else 15
    p99 = np.percentile(M, 99) if M.max() > 0 else 1.0
    norm = plt.Normalize(vmin=0, vmax=p99)

    ax.quiver(Xq, Yq, U, V, M, cmap='jet', norm=norm, scale=scale,
              scale_units='inches', width=0.002, headwidth=4, headlength=5,
              headaxislength=4.5, minshaft=1, pivot='mid')

    cax = fig.add_axes([0.87, 0.1, 0.03, 0.8])
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='jet'), cax=cax)
    if field_type and 'electro' in field_type.lower():
        cbar.set_label('Electric Field Magnitude [V/m]', fontsize=12)
    elif field_type and 'magneto' in field_type.lower():
        cbar.set_label('Magnetic Field Magnitude [T]', fontsize=12)
    else:
        cbar.set_label('Field Magnitude', fontsize=12)

    if show_geometry:
        add_geometry_lines(ax)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    ax.set_xlim(x_data[0], x_data[-1])
    ax.set_ylim(y_data[0], y_data[-1])

    full_path = os.path.join(save_folder, filename) if save_folder else filename
    os.makedirs(os.path.dirname(full_path) if os.path.dirname(full_path) else '.', exist_ok=True)
    plt.savefig(f"{full_path}.pdf", bbox_inches='tight', dpi=300)
    plt.close()


def plot_equipotential_lines(Phi, coil_potential, save_folder, dx, dy,
                              use_db_scale=False, show_geometry=True):
    """Plot equipotential contour lines."""
    os.makedirs(save_folder, exist_ok=True)
    nx, ny = Phi.shape
    x_pos = np.linspace(0, (nx - 1) * dx * 1e3, nx)
    y_pos = np.linspace(0, (ny - 1) * dy * 1e3, ny)
    X, Y = np.meshgrid(x_pos, y_pos, indexing='ij')

    Phi_plot = np.log10(np.abs(Phi) + 1e-10) if use_db_scale else Phi
    label = 'Log|Electric Potential| [V]' if use_db_scale else 'Electric Potential [V]'

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes([0.1, 0.1, 0.75, 0.8])

    n_levels = 75
    vmin, vmax = np.min(Phi_plot), np.max(Phi_plot)
    levels = np.linspace(vmin + (vmax - vmin) * 0.01, vmax - (vmax - vmin) * 0.01, n_levels)
    colors = plt.cm.jet(np.linspace(0, 1, len(levels)))
    cs = ax.contour(X, Y, Phi_plot, levels=levels, colors=colors, linewidths=1)

    cax = fig.add_axes([0.87, 0.2, 0.03, 0.6])
    cbar = plt.colorbar(cs, cax=cax)
    cbar.set_label(label, fontsize=14)

    if show_geometry:
        add_geometry_lines(ax)

    ax.set_title(f'Equipotential Lines (V_coil = {coil_potential:.2f}V)', fontsize=15)
    ax.set_xlabel('x [mm]', fontsize=14)
    ax.set_ylabel('y [mm]', fontsize=14)
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    plt.savefig(os.path.join(save_folder, 'equipotential_lines.pdf'),
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()


def plot_field_streamlines(field_x, field_y, filename, title, x_data, y_data,
                           field_type=None, save_folder=None, show_geometry=True):
    """Plot field streamlines for better visualization of field topology."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes([0.1, 0.1, 0.75, 0.8])

    magnitude = np.sqrt(field_x**2 + field_y**2)
    X, Y = np.meshgrid(x_data, y_data, indexing='ij')
    vmax = np.percentile(magnitude, 99) if magnitude.max() > 0 else 1.0
    norm = plt.Normalize(vmin=0, vmax=vmax)

    strm = ax.streamplot(X.T, Y.T, field_x.T, field_y.T,
                         density=1.8, color=magnitude.T, cmap='jet',
                         norm=norm, linewidth=1.5, arrowsize=1.2)

    cax = fig.add_axes([0.87, 0.1, 0.03, 0.8])
    cbar = plt.colorbar(strm.lines, cax=cax, norm=norm)
    if field_type and 'electro' in field_type.lower():
        cbar.set_label('Electric Field Magnitude [V/m]', fontsize=12)
    elif field_type and 'magneto' in field_type.lower():
        cbar.set_label('Magnetic Field Magnitude [T]', fontsize=12)
    else:
        cbar.set_label('Field Magnitude', fontsize=12)

    if show_geometry:
        add_geometry_lines(ax)

    ax.set_xlabel('x [mm]', fontsize=12)
    ax.set_ylabel('y [mm]', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    ax.set_xlim(x_data[0], x_data[-1])
    ax.set_ylim(y_data[0], y_data[-1])

    full_path = os.path.join(save_folder, filename) if save_folder else filename
    os.makedirs(os.path.dirname(full_path) if os.path.dirname(full_path) else '.', exist_ok=True)
    plt.savefig(f"{full_path}.pdf", bbox_inches='tight', dpi=300)
    plt.close()


def create_field_animation(field_history, filename, x_data, y_data, title,
                           field_type='electromagnetic', save_folder=None,
                           xlabel='x [mm]', ylabel='y [mm]', show_geometry=True):
    """Create GIF animation from field evolution data."""
    if len(field_history) < 2:
        return
    if len(field_history) > 50:
        field_history = field_history[::len(field_history) // 50]

    fig, ax = plt.subplots(figsize=(10, 8))
    vmin = min(f.min() for f in field_history)
    vmax = max(f.max() for f in field_history)

    def update(frame):
        ax.clear()
        ax.imshow(field_history[frame].T, origin='lower',
                  extent=[x_data[0], x_data[-1], y_data[0], y_data[-1]],
                  cmap='jet', vmin=vmin, vmax=vmax)
        if show_geometry:
            add_geometry_lines(ax)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{title} — Frame {frame}', fontsize=14)

    save_dir = save_folder or os.path.join('results', 'plots', field_type)
    os.makedirs(save_dir, exist_ok=True)

    anim = FuncAnimation(fig, update, frames=len(field_history), interval=100)
    anim.save(os.path.join(save_dir, f'{filename}.gif'), writer=PillowWriter(fps=10))
    plt.close()
    logger.info(f"Animation saved: {filename}.gif")
