"""
Physics-Informed Neural Network (PINN) for solving the electron Boltzmann equation.

Solves the steady-state, spatially homogeneous two-term Boltzmann equation
in energy space to predict the Electron Energy Distribution Function (EEDF)
as a function of the reduced electric field E/N.

References
----------
- Kim 2023: "Numerical strategy for solving the Boltzmann equation with
  variable E/N using physics-informed neural networks"
- Kawaguchi 2022: "Physics-informed neural networks for solving the Boltzmann
  equation of the electron velocity distribution in weakly ionized plasmas"
- Hagelaar & Pitchford, PSST 14, 2005 (two-term Boltzmann equation)
"""

import numpy as np
import torch
import torch.nn as nn
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Physical constants
ME = 9.10938e-31      # electron mass [kg]
QE = 1.602176634e-19  # elementary charge [C]
GAMMA = np.sqrt(2 * QE / ME)  # ~5.93e5 sqrt(eV) -> m/s conversion


def get_device():
    """Select best available device (CUDA on GPU, else CPU).

    Note: MPS (Apple Silicon) is intentionally skipped due to compatibility
    issues with PyTorch < 2.0 and NumPy 2.x tensor conversion. CPU is fast
    enough for the small networks used here (~13k params).
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class BoltzmannPINN(nn.Module):
    """Neural network that predicts log(EEDF) from (log10(E/N), epsilon).

    Architecture: 2 inputs -> [n_layers x n_neurons, tanh] -> 1 output (log f).

    Parameters
    ----------
    n_hidden : int
        Number of hidden layers (default: 4).
    n_neurons : int
        Neurons per hidden layer (default: 64).
    en_range : tuple
        (log10(EN_min_Td), log10(EN_max_Td)) for input normalization.
    eps_max : float
        Maximum energy [eV] for epsilon normalization to [0, 1].
    """

    def __init__(self, n_hidden=4, n_neurons=64,
                 en_range=(0.0, 2.7), eps_max=60.0):
        super().__init__()
        self.en_range = en_range
        self.eps_max = eps_max

        # Build network
        layers = [nn.Linear(2, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers.extend([nn.Linear(n_neurons, n_neurons), nn.Tanh()])
        layers.append(nn.Linear(n_neurons, 1))
        self.net = nn.Sequential(*layers)

        # Initialize weights (Xavier)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, log10_en, eps):
        """Forward pass. Returns log(f(epsilon)).

        Parameters
        ----------
        log10_en : Tensor (N,) or (N,1)
            log10(E/N in Td).
        eps : Tensor (N,) or (N,1)
            Electron energy [eV].

        Returns
        -------
        log_f : Tensor (N, 1)
            log of EEDF value.
        """
        # Normalize inputs to ~[-1, 1]
        en_norm = 2.0 * (log10_en.view(-1, 1) - self.en_range[0]) / \
                  (self.en_range[1] - self.en_range[0]) - 1.0
        eps_norm = 2.0 * eps.view(-1, 1) / self.eps_max - 1.0

        x = torch.cat([en_norm, eps_norm], dim=1)
        return self.net(x)

    def predict_eedf(self, en_td, eps_ev):
        """Predict EEDF at given E/N [Td] and energy grid [eV].

        Parameters
        ----------
        en_td : float or array
            Reduced electric field [Td].
        eps_ev : array
            Energy grid [eV].

        Returns
        -------
        f : ndarray
            EEDF values [eV^{-3/2}].
        """
        self.eval()
        # Determine model device
        device = next(self.parameters()).device

        with torch.no_grad():
            en_td = np.atleast_1d(np.float64(en_td))
            eps_ev = np.asarray(eps_ev, dtype=np.float64)

            # If single E/N, broadcast to match eps grid
            if len(en_td) == 1:
                log_en = torch.full((len(eps_ev),), np.log10(en_td[0]),
                                    dtype=torch.float32, device=device)
            else:
                log_en = torch.tensor(np.log10(en_td), dtype=torch.float32,
                                      device=device)

            eps = torch.tensor(eps_ev, dtype=torch.float32, device=device)
            log_f = self.forward(log_en, eps)
            f_tensor = torch.exp(log_f).squeeze().cpu()

            # Workaround for PyTorch/NumPy version mismatch:
            # use .tolist() + np.array instead of .numpy()
            try:
                f = f_tensor.numpy()
            except RuntimeError:
                f = np.array(f_tensor.tolist(), dtype=np.float64)

        return f


class BoltzmannPINNTrainer:
    """Training loop for the Boltzmann PINN.

    Implements the physics-informed loss:
    L = w_data * L_data + w_norm * L_norm + w_bc * L_bc

    For the initial implementation, we focus on data-driven training with
    normalization and boundary constraints. The full PDE residual loss
    (L_pde) can be added as a refinement.

    Parameters
    ----------
    model : BoltzmannPINN
        The neural network model.
    reference_data : dict
        Reference data from bolos (.npz file contents).
    cross_sections : CrossSectionSet
        Ar cross-section data (for future PDE loss).
    device : torch.device
        Computation device.
    """

    def __init__(self, model, reference_data, cross_sections=None,
                 device=None):
        self.model = model
        self.device = device or get_device()
        self.model.to(self.device)
        self.cross_sections = cross_sections

        # Unpack reference data
        self.en_td_ref = reference_data['EN_Td']       # (N_en,)
        self.eps_ref = reference_data['energy_grid']     # (N_eps,)
        self.eedf_ref = reference_data['eedf']           # (N_en, N_eps)

        # Precompute log of reference EEDF (clip to avoid log(0))
        self.log_eedf_ref = np.log(np.clip(self.eedf_ref, 1e-30, None))

        # Prepare training tensors
        self._prepare_training_data()

        # Adaptive loss weights (learnable)
        self.log_w_data = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.log_w_norm = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.log_w_bc = nn.Parameter(torch.tensor(0.0, device=self.device))

    def _prepare_training_data(self):
        """Create meshgrid tensors from reference data for training."""
        n_en = len(self.en_td_ref)
        n_eps = len(self.eps_ref)

        # Create (E/N, epsilon) pairs for all reference points
        log_en_grid = np.log10(self.en_td_ref)
        en_mesh, eps_mesh = np.meshgrid(log_en_grid, self.eps_ref, indexing='ij')

        self.train_log_en = torch.tensor(
            en_mesh.ravel(), dtype=torch.float32, device=self.device
        )
        self.train_eps = torch.tensor(
            eps_mesh.ravel(), dtype=torch.float32, device=self.device
        )
        self.train_log_f = torch.tensor(
            self.log_eedf_ref.ravel(), dtype=torch.float32, device=self.device
        )

        # Boundary condition points: f(eps_max) -> 0 for all E/N
        self.bc_log_en = torch.tensor(
            log_en_grid, dtype=torch.float32, device=self.device
        )
        self.bc_eps_max = torch.full(
            (n_en,), self.eps_ref[-1],
            dtype=torch.float32, device=self.device
        )

        # For normalization: quadrature weights (trapezoidal rule)
        eps_t = torch.tensor(self.eps_ref, dtype=torch.float32, device=self.device)
        de = eps_t[1:] - eps_t[:-1]
        # Trapezoidal weights
        w = torch.zeros_like(eps_t)
        w[0] = de[0] / 2
        w[-1] = de[-1] / 2
        w[1:-1] = (de[:-1] + de[1:]) / 2
        self.quad_weights = w  # (N_eps,)
        self.quad_sqrt_eps = torch.sqrt(eps_t)  # (N_eps,)

        logger.info(f"Training data: {n_en} E/N points x {n_eps} energies = "
                    f"{n_en * n_eps} total points")

    def compute_data_loss(self):
        """MSE between predicted and reference log(EEDF)."""
        log_f_pred = self.model(self.train_log_en, self.train_eps).squeeze()
        # Weight by significance: points with larger EEDF matter more
        return torch.mean((log_f_pred - self.train_log_f) ** 2)

    def compute_normalization_loss(self):
        """Enforce integral(sqrt(eps) * f(eps) d_eps) = 1 for each E/N.

        This is the standard EEDF normalization in the Boltzmann equation.
        """
        n_en = len(self.en_td_ref)
        n_eps = len(self.eps_ref)
        log_en_grid = np.log10(self.en_td_ref)

        total_loss = torch.tensor(0.0, device=self.device)

        for i, log_en in enumerate(log_en_grid):
            log_en_t = torch.full((n_eps,), log_en,
                                  dtype=torch.float32, device=self.device)
            eps_t = torch.tensor(self.eps_ref, dtype=torch.float32,
                                 device=self.device)

            log_f = self.model(log_en_t, eps_t).squeeze()
            f = torch.exp(log_f)

            # integral(sqrt(eps) * f(eps) d_eps)
            integrand = self.quad_sqrt_eps * f
            integral = torch.sum(integrand * self.quad_weights)
            total_loss += (integral - 1.0) ** 2

        return total_loss / n_en

    def compute_bc_loss(self):
        """Boundary: f(eps_max) should be ~0 (very small)."""
        log_f_bc = self.model(self.bc_log_en, self.bc_eps_max).squeeze()
        # f at boundary should be very small -> log(f) should be very negative
        # Penalize if log(f) > -20 (i.e., f > ~2e-9)
        return torch.mean(torch.relu(log_f_bc + 20.0) ** 2)

    def train(self, epochs=50000, lr=1e-3, lr_min=1e-6,
              log_every=1000, validate_every=5000):
        """Train the PINN model.

        Parameters
        ----------
        epochs : int
            Number of training epochs.
        lr : float
            Initial learning rate.
        lr_min : float
            Minimum learning rate for cosine annealing.
        log_every : int
            Print loss every N epochs.
        validate_every : int
            Run validation every N epochs.

        Returns
        -------
        history : dict
            Training history with loss values.
        """
        # Optimizer: model params + adaptive weights
        all_params = list(self.model.parameters()) + \
                     [self.log_w_data, self.log_w_norm, self.log_w_bc]
        optimizer = torch.optim.Adam(all_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr_min
        )

        history = {'epoch': [], 'loss': [], 'data': [], 'norm': [], 'bc': []}

        logger.info(f"Starting PINN training: {epochs} epochs, lr={lr}, "
                    f"device={self.device}")

        for epoch in range(1, epochs + 1):
            self.model.train()
            optimizer.zero_grad()

            # Compute losses
            l_data = self.compute_data_loss()
            l_norm = self.compute_normalization_loss()
            l_bc = self.compute_bc_loss()

            # Adaptive weights
            w_data = torch.exp(self.log_w_data)
            w_norm = torch.exp(self.log_w_norm)
            w_bc = torch.exp(self.log_w_bc)

            loss = w_data * l_data + w_norm * l_norm + w_bc * l_bc

            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % log_every == 0 or epoch == 1:
                history['epoch'].append(epoch)
                history['loss'].append(loss.item())
                history['data'].append(l_data.item())
                history['norm'].append(l_norm.item())
                history['bc'].append(l_bc.item())

                current_lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"[{epoch}/{epochs}] loss={loss.item():.4e} "
                    f"(data={l_data.item():.4e}, norm={l_norm.item():.4e}, "
                    f"bc={l_bc.item():.4e}) lr={current_lr:.2e}"
                )

            if epoch % validate_every == 0:
                self._validate(epoch)

        return history

    def _validate(self, epoch):
        """Quick validation: check EEDF shape and normalization."""
        self.model.eval()
        with torch.no_grad():
            # Test at a few E/N values
            test_en = [5.0, 50.0, 200.0]
            for en_td in test_en:
                eps = np.array(self.eps_ref)
                f = self.model.predict_eedf(en_td, eps)

                # Normalization integral
                de = eps[1:] - eps[:-1]
                norm = np.sum(np.sqrt(eps[:-1]) * f[:-1] * de)
                mean_e = np.sum(eps[:-1] ** 1.5 * f[:-1] * de) / max(norm, 1e-30)

                logger.info(
                    f"  Validate E/N={en_td:.0f} Td: norm={norm:.4f}, "
                    f"<eps>={mean_e:.2f} eV, f(0)={f[0]:.3e}, f(max)={f[-1]:.3e}"
                )

    def save_model(self, filepath):
        """Save trained model checkpoint."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'en_range': self.model.en_range,
            'eps_max': self.model.eps_max,
            'n_hidden': len([m for m in self.model.net if isinstance(m, nn.Linear)]) - 1,
            'n_neurons': self.model.net[0].out_features,
        }, filepath)
        logger.info(f"Model saved to {filepath}")

    @staticmethod
    def load_model(filepath, device=None):
        """Load a trained model from checkpoint.

        Parameters
        ----------
        filepath : str or Path
            Path to .pt checkpoint file.
        device : torch.device, optional
            Device to load model onto.

        Returns
        -------
        model : BoltzmannPINN
            Loaded model in eval mode.
        """
        device = device or get_device()
        checkpoint = torch.load(filepath, map_location=device)

        model = BoltzmannPINN(
            n_hidden=checkpoint['n_hidden'],
            n_neurons=checkpoint['n_neurons'],
            en_range=checkpoint['en_range'],
            eps_max=checkpoint['eps_max'],
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        logger.info(f"Model loaded from {filepath}")
        return model


def compute_transport_coefficients(eedf, energy_grid, cross_sections):
    """Compute transport and rate coefficients from EEDF.

    Parameters
    ----------
    eedf : ndarray (N_eps,)
        Electron energy distribution function [eV^{-3/2}].
    energy_grid : ndarray (N_eps,)
        Energy grid [eV].
    cross_sections : CrossSectionSet
        Gas cross-section data.

    Returns
    -------
    dict with keys:
        Te : float - electron temperature [eV]
        mean_energy : float - mean electron energy [eV]
        mobility_N : float - reduced mobility mu*N [1/(m·V·s)]
        diffusion_N : float - reduced diffusion D*N [1/(m·s)]
        rate_coefficients : dict - {process_name: rate [m³/s]}
    """
    eps = energy_grid
    f = eedf
    de = np.diff(eps)

    # Normalization
    norm = np.sum(np.sqrt(eps[:-1]) * f[:-1] * de)
    if norm < 1e-30:
        norm = 1.0

    # Mean energy
    mean_energy = np.sum(eps[:-1] ** 1.5 * f[:-1] * de) / norm

    # Electron temperature (from mean energy, assuming Te = 2/3 <eps>)
    Te = (2.0 / 3.0) * mean_energy

    # Reduced mobility: mu*N = -gamma/3 * integral(eps/sigma_m * df/deps d_eps)
    sigma_m = cross_sections.sigma_m(eps)
    # Use simpler formula: mu*N = -gamma/3 * integral(eps * f / sigma_m d_eps)
    # More robust: mu*N = gamma/3 * integral(eps / sigma_m(eps) * (-df/deps) d_eps)
    # Approximate df/deps with central differences
    df_deps = np.gradient(f, eps)
    integrand_mu = eps / np.maximum(sigma_m, 1e-30) * (-df_deps)
    mobility_N = GAMMA / 3.0 * np.sum(integrand_mu[:-1] * de)

    # Reduced diffusion: D*N = gamma/3 * integral(eps / sigma_m(eps) * f d_eps)
    integrand_D = eps / np.maximum(sigma_m, 1e-30) * f
    diffusion_N = GAMMA / 3.0 * np.sum(integrand_D[:-1] * de)

    # Rate coefficients for inelastic processes
    rate_coefficients = {}
    for name in cross_sections.inelastic_names:
        sigma_inel = cross_sections.sigma_inelastic(name, eps)
        # k = gamma * integral(eps * sigma(eps) * f(eps) d_eps)
        integrand_k = eps * sigma_inel * f
        rate = GAMMA * np.sum(integrand_k[:-1] * de)
        rate_coefficients[name] = rate

    return {
        'Te': Te,
        'mean_energy': mean_energy,
        'mobility_N': mobility_N,
        'diffusion_N': diffusion_N,
        'rate_coefficients': rate_coefficients,
    }
