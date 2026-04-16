"""
Tandem Network and CVAE baselines for thin-film inverse design.

Both use fixed-length representations:
  - Structure: [20 material IDs (0-16, padded with -1), 20 thicknesses (padded with 0)]
  - Spectrum: [142 floats]

Tandem Network:
  Inverse MLP (spectrum → structure) + Forward MLP (structure → spectrum)
  Trained jointly with reconstruction + consistency loss.

CVAE:
  Conditional VAE: encoder(spectrum, structure) → z, decoder(spectrum, z) → structure
  At inference, sample z ~ N(0,1) and decode.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from prism.constants import MAX_LAYERS, N_SPECTRUM

N_MATERIALS = 17


# ── Tandem Network ────────────────────────────────────────────────────────────

class ForwardNet(nn.Module):
    """Structure → Spectrum MLP."""

    def __init__(self, d_hidden: int = 512):
        super().__init__()
        # Input: 20 material one-hots (20*17=340) + 20 thicknesses + 1 length = 361
        input_dim = MAX_LAYERS * N_MATERIALS + MAX_LAYERS + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, N_SPECTRUM), nn.Sigmoid(),
        )

    def forward(self, mat_onehot: Tensor, thk: Tensor, length: Tensor) -> Tensor:
        """
        Args:
            mat_onehot: [B, 20, 17] one-hot material encoding
            thk: [B, 20] normalized thicknesses
            length: [B, 1] normalized layer count
        Returns: [B, 142] predicted spectrum
        """
        x = torch.cat([mat_onehot.flatten(1), thk, length], dim=-1)
        return self.net(x)


class InverseNet(nn.Module):
    """Spectrum → Structure MLP."""

    def __init__(self, d_hidden: int = 512):
        super().__init__()
        self.d_hidden = d_hidden
        self.shared = nn.Sequential(
            nn.Linear(N_SPECTRUM, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
        )
        self.mat_head = nn.Linear(d_hidden, MAX_LAYERS * N_MATERIALS)  # logits
        self.thk_head = nn.Linear(d_hidden, MAX_LAYERS)
        self.len_head = nn.Linear(d_hidden, MAX_LAYERS)  # classify layer count 1-20

    def forward(self, spectrum: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            mat_logits: [B, 20, 17]
            thk_pred: [B, 20] (normalized)
            len_logits: [B, 20] (layer count classification)
        """
        h = self.shared(spectrum)
        mat_logits = self.mat_head(h).view(-1, MAX_LAYERS, N_MATERIALS)
        thk_pred = torch.sigmoid(self.thk_head(h))
        len_logits = self.len_head(h)
        return mat_logits, thk_pred, len_logits


class TandemNetwork(nn.Module):
    """Joint inverse + forward network with consistency loss."""

    def __init__(self, d_hidden: int = 512):
        super().__init__()
        self.inverse = InverseNet(d_hidden)
        self.forward_net = ForwardNet(d_hidden)

    def forward(self, spectrum: Tensor):
        mat_logits, thk_pred, len_logits = self.inverse(spectrum)
        # Use soft one-hot for differentiable forward pass
        mat_soft = F.softmax(mat_logits, dim=-1)
        length = len_logits.argmax(dim=-1, keepdim=True).float() / MAX_LAYERS
        spec_recon = self.forward_net(mat_soft, thk_pred, length)
        return mat_logits, thk_pred, len_logits, spec_recon


# ── CVAE ──────────────────────────────────────────────────────────────────────

class CVAEEncoder(nn.Module):
    """Encode (spectrum, structure) → (mu, logvar)."""

    def __init__(self, d_hidden: int = 512, d_latent: int = 64):
        super().__init__()
        struct_dim = MAX_LAYERS * N_MATERIALS + MAX_LAYERS + 1
        self.net = nn.Sequential(
            nn.Linear(N_SPECTRUM + struct_dim, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(d_hidden, d_latent)
        self.logvar = nn.Linear(d_hidden, d_latent)

    def forward(self, spectrum: Tensor, mat_onehot: Tensor, thk: Tensor, length: Tensor):
        struct = torch.cat([mat_onehot.flatten(1), thk, length], dim=-1)
        h = self.net(torch.cat([spectrum, struct], dim=-1))
        return self.mu(h), self.logvar(h)


class CVAEDecoder(nn.Module):
    """Decode (spectrum, z) → structure."""

    def __init__(self, d_hidden: int = 512, d_latent: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(N_SPECTRUM + d_latent, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
        )
        self.mat_head = nn.Linear(d_hidden, MAX_LAYERS * N_MATERIALS)
        self.thk_head = nn.Linear(d_hidden, MAX_LAYERS)
        self.len_head = nn.Linear(d_hidden, MAX_LAYERS)

    def forward(self, spectrum: Tensor, z: Tensor):
        h = self.shared(torch.cat([spectrum, z], dim=-1))
        mat_logits = self.mat_head(h).view(-1, MAX_LAYERS, N_MATERIALS)
        thk_pred = torch.sigmoid(self.thk_head(h))
        len_logits = self.len_head(h)
        return mat_logits, thk_pred, len_logits


class CVAE(nn.Module):
    def __init__(self, d_hidden: int = 512, d_latent: int = 64):
        super().__init__()
        self.d_latent = d_latent
        self.encoder = CVAEEncoder(d_hidden, d_latent)
        self.decoder = CVAEDecoder(d_hidden, d_latent)

    def forward(self, spectrum, mat_onehot, thk, length):
        mu, logvar = self.encoder(spectrum, mat_onehot, thk, length)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        mat_logits, thk_pred, len_logits = self.decoder(spectrum, z)
        return mat_logits, thk_pred, len_logits, mu, logvar

    def sample(self, spectrum: Tensor, n_samples: int = 1):
        """Sample designs at inference."""
        B = spectrum.shape[0]
        z = torch.randn(B, n_samples, self.d_latent, device=spectrum.device)
        spec_exp = spectrum.unsqueeze(1).expand(-1, n_samples, -1)
        # Flatten for decoder
        z_flat = z.reshape(B * n_samples, -1)
        spec_flat = spec_exp.reshape(B * n_samples, -1)
        mat_logits, thk_pred, len_logits = self.decoder(spec_flat, z_flat)
        return (
            mat_logits.view(B, n_samples, MAX_LAYERS, N_MATERIALS),
            thk_pred.view(B, n_samples, MAX_LAYERS),
            len_logits.view(B, n_samples, MAX_LAYERS),
        )
