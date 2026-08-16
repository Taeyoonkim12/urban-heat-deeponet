"""DeepONet architectures for the model ladder M0-M5.

All variants share the same backbone (6-layer residual MLPs with
LayerNorm/GELU, latent dim 256). They differ only in the trunk input
features, the branch inputs and the branch/trunk coupling:

  M0       canonical DeepONet (inner-product coupling, raw coordinates)
  M1       + adaptive dual-path branch-trunk combiner
  M2       + signed building-proximity feature d_BP
  M3       + multiscale Fourier features
  M4       + learnable surface-category embedding (primary model)
  M5       + solar branch re-encoding (RH, T, altitude, azimuth, BHI)
             and transient SEB regularization

Supplementary controls: D1 (inner coupling + d_BP) and the solar
re-encoding control (M4 architecture with solar branch, no SEB).
"""

import numpy as np
import torch
import torch.nn as nn

from .config import (LATENT_DIM, HIDDEN_DIM, DEPTH, DROPOUT, NUM_FREQUENCIES,
                     FOURIER_SEED, N_CATEGORIES, CAT_EMBED_DIM)

# Model registry. Keys follow the paper's model labels (M0-M5)
# plus two supplementary controls (D1, solar control).
MODEL_SPECS = {
    # M0: canonical DeepONet - inner-product coupling, raw coordinates.
    'm0':     dict(branch='basic', use_dbp=False, use_fourier=False,
                   use_category=False,
                   use_seb=False, coupling='inner'),
    # M1: M0 backbone + adaptive dual-path branch-trunk combiner.
    'm1':     dict(branch='basic', use_dbp=False, use_fourier=False,
                   use_category=False,
                   use_seb=False, coupling='dual'),
    # M2: M1 + signed building-proximity feature d_BP.
    'm2':     dict(branch='basic', use_dbp=True, use_fourier=False,
                   use_category=False,
                   use_seb=False, coupling='dual'),
    # M3: M2 + multiscale Fourier features.
    'm3':     dict(branch='basic', use_dbp=True, use_fourier=True,
                   use_category=False,
                   use_seb=False, coupling='dual'),
    # M4: M3 + learnable surface-category embedding (primary model).
    'm4':     dict(branch='basic', use_dbp=True, use_fourier=True,
                   use_category=True,
                   use_seb=False, coupling='dual'),
    # M5: physics-consistent extension of M4 - solar branch re-encoding
    # [RH, T, altitude, azimuth, BHI] + transient SEB regularization.
    # Identical architecture to the solar control; only the loss differs.
    'm5':     dict(branch='solar', use_dbp=True, use_fourier=True,
                   use_category=True,
                   use_seb=True, seb_mode='transient', coupling='dual'),
    # (trains on the original + surface-modification configurations)
    # --- supplementary controls (not part of the M0-M5 progression) ---
    # D1: canonical inner-product coupling + d_BP.
    'd1':     dict(branch='basic', use_dbp=True, use_fourier=False,
                   use_category=False,
                   use_seb=False, coupling='inner'),
    # Solar re-encoding control: M4 with the solar branch, no SEB loss.
    'solar_control': dict(branch='solar', use_dbp=True, use_fourier=True,
                   use_category=True,
                   use_seb=False, coupling='dual'),
}


def _init_linear(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class BranchNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=HIDDEN_DIM, output_dim=LATENT_DIM,
                 depth=DEPTH, dropout=DROPOUT):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                          nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
            for _ in range(depth - 2)])
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim))
        _init_linear(self)

    def forward(self, params):
        x = self.input_layer(params)
        for layer in self.hidden_layers:
            x = layer(x) + x
        return self.output_layer(x)


class MultiScaleFourierFeatures(nn.Module):
    def __init__(self, input_dim=4, num_frequencies=NUM_FREQUENCIES, seed=FOURIER_SEED):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        scales = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        per_scale = num_frequencies // len(scales)
        B = [torch.randn(input_dim, per_scale) * s for s in scales]
        rem = num_frequencies - per_scale * len(scales)
        if rem > 0:
            B.append(torch.randn(input_dim, rem) * 5.0)
        self.register_buffer('B', torch.cat(B, dim=1))
        self.output_dim = num_frequencies * 2

    def forward(self, x):
        proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class TrunkNet(nn.Module):
    """Trunk with optional Fourier embedding and category embedding."""

    def __init__(self, coord_dim, use_fourier=False,
                 use_category=False, hidden_dim=HIDDEN_DIM, output_dim=LATENT_DIM,
                 depth=DEPTH, dropout=DROPOUT):
        super().__init__()
        self.coord_dim = coord_dim

        input_dim = coord_dim
        if use_fourier:
            self.fourier = MultiScaleFourierFeatures(coord_dim)
            input_dim += self.fourier.output_dim
        else:
            self.fourier = None
        if use_category:
            self.cat_embed = nn.Embedding(N_CATEGORIES, CAT_EMBED_DIM)
            nn.init.normal_(self.cat_embed.weight, mean=0.0, std=0.1)
            input_dim += CAT_EMBED_DIM
        else:
            self.cat_embed = None

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                          nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
            for _ in range(depth - 2)])
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim))
        _init_linear(self)

    def forward(self, trunk, cat_idx=None):
        coords = trunk[:, :self.coord_dim]
        parts = [coords]
        if self.fourier is not None:
            parts.append(self.fourier(coords))
        if self.cat_embed is not None:
            parts.append(self.cat_embed(cat_idx))
        x = self.input_layer(torch.cat(parts, dim=-1))
        for layer in self.hidden_layers:
            x = layer(x) + x
        return self.output_layer(x)


class PureDeepONet(nn.Module):
    """M0/D1: canonical DeepONet, branch and trunk coupled by the inner
    product of Eq. (7)."""

    def __init__(self, branch_dim=3, trunk_dim=3):
        super().__init__()
        self.branch_net = BranchNet(input_dim=branch_dim)
        self.trunk_net = TrunkNet(coord_dim=trunk_dim)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, branch_in, trunk_in, cat_idx=None):
        b = self.branch_net(branch_in)
        t = self.trunk_net(trunk_in)
        inner = torch.sum(b.float() * t.float(), dim=1, keepdim=True)
        return inner + self.bias


class DualPathDeepONet(nn.Module):
    """M1 onwards: learnable blend of the inner product and an MLP head
    over the concatenated latents. The inner product is computed in fp32
    (outside autocast) and normalised by sqrt(latent dim)."""

    def __init__(self, branch_dim, trunk_kwargs):
        super().__init__()
        self.branch_net = BranchNet(input_dim=branch_dim)
        self.trunk_net = TrunkNet(**trunk_kwargs)
        self.combiner = nn.Sequential(
            nn.Linear(LATENT_DIM * 2, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 1))
        self.bias = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.inner_scale = LATENT_DIM ** 0.5

    def forward(self, branch_in, trunk_in, cat_idx=None):
        b = self.branch_net(branch_in)
        t = self.trunk_net(trunk_in, cat_idx) if self.trunk_net.cat_embed is not None \
            else self.trunk_net(trunk_in)
        if b.is_cuda:
            with torch.amp.autocast('cuda', enabled=False):
                inner = torch.sum(b.float() * t.float(), dim=1, keepdim=True) / self.inner_scale
        else:
            inner = torch.sum(b.float() * t.float(), dim=1, keepdim=True) / self.inner_scale
        mlp_out = self.combiner(torch.cat([b, t], dim=-1))
        a = torch.sigmoid(self.alpha)
        return a * inner + (1 - a) * mlp_out + self.bias


def build_model(name):
    spec = MODEL_SPECS[name]
    branch_dim = 5 if spec['branch'] == 'solar' else 3
    coord_dim = 4 if spec['use_dbp'] else 3

    if spec['coupling'] == 'inner':
        return PureDeepONet(branch_dim=branch_dim, trunk_dim=coord_dim), spec

    trunk_kwargs = dict(coord_dim=coord_dim,
                        use_fourier=spec['use_fourier'],
                        use_category=spec['use_category'])
    return DualPathDeepONet(branch_dim, trunk_kwargs), spec
