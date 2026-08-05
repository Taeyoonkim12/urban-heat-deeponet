"""Surface energy balance (SEB) soft constraint (paper Sec. 2.3.4,
Eqs. 10-12).

The residual per surface point is
    R = Q_sw + eps * (L_down - sigma * Ts^4) - h * (Ts - Ta) - G_cat
with learnable effective parameters: convective coefficient h, sky
temperature depression dT_sky (L_down = sigma * (Ta - dT_sky)^4) and a
per-category ground/storage flux G_cat. Green and Water points are
excluded via W_PHYS_CAT.
"""

import torch
import torch.nn as nn

from .config import N_CATEGORIES, RESID_SCALE, W_PHYS_CAT
from .solar import EPS_CAT

SIGMA_SB = 5.670374419e-8


class SEBHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_h = nn.Parameter(torch.tensor(2.3))    # softplus*4 -> ~10 W/m2K range
        self.raw_dt = nn.Parameter(torch.tensor(2.7))   # softplus*5 -> ~15 K range
        self.g_cat = nn.Parameter(torch.zeros(N_CATEGORIES))

    @property
    def h(self):
        return torch.nn.functional.softplus(self.raw_h) * 4.0

    @property
    def dt(self):
        return torch.nn.functional.softplus(self.raw_dt) * 5.0


class SEBLoss(nn.Module):
    def __init__(self, seb_head, output_scaler, device):
        super().__init__()
        self.head = seb_head
        self.out_mean = float(output_scaler.mean_[0])
        self.out_scale = float(output_scaler.scale_[0])
        self.eps = torch.tensor(EPS_CAT, device=device)
        self.w_phys = torch.tensor(W_PHYS_CAT, dtype=torch.float32, device=device)

    def forward(self, pred_scaled, qsw, cat, tamb):
        T = pred_scaled.float().flatten() * self.out_scale + self.out_mean
        Tk = T + 273.15
        Tsky = tamb + 273.15 - self.head.dt
        Ldn = SIGMA_SB * Tsky ** 4
        resid = qsw + self.eps[cat] * (Ldn - SIGMA_SB * Tk ** 4) \
            - self.head.h * (T - tamb) - self.head.g_cat[cat]
        return (self.w_phys[cat] * (resid / RESID_SCALE) ** 2).mean()
