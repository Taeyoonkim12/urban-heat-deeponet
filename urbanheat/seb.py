"""Fixed-coefficient ray-resolved surface-energy-balance loss for M5.

For each Building receiver:

    R = Q_sw + eps_B * (G_lw - sigma * Ts**4) + G0
        - h * (Ts - Ta) - C_A * dTs/dt

The loss penalizes only the residual excess above the closure threshold
`R_floor` (R0): the reduced balance omits physics (reflected shortwave,
multi-bounce longwave, resolved wall conduction), so even the CFD
ground-truth temperature field leaves a nonzero residual. `R_floor = 0`
recovers the plain squared residual.

`G_lw` is the hemispheric mean of emissive radiance from ray-hit surfaces.
Sky rays have zero incoming radiance, matching the completely transparent
upper radiative boundary in the source CFD. Unresolved rays retain the
original ray-count denominator and receive zero numerical contribution; their
effect is represented by the independent QC confidence weight.
"""

import math

import torch
import torch.nn as nn

from .config import ROOF_NORMAL_Z
from .solar import EPS_CAT

SIGMA_SB = 5.670374419e-8


class SurfaceEnergyBalanceLoss(nn.Module):
    """Weight-normalized squared SEB residual with prescribed coefficients."""

    def __init__(self, output_scaler, h_roof, h_wall, c_areal,
                 resid_scale=100.0, residual_floor=0.0,
                 g0_roof=0.0, g0_wall=0.0):
        super().__init__()
        values = (h_roof, h_wall, c_areal, resid_scale)
        if not all(math.isfinite(float(value)) and value > 0
                   for value in values):
            raise ValueError('SEB coefficients and residual scale must be positive')
        if not math.isfinite(float(residual_floor)) or residual_floor < 0:
            raise ValueError('residual floor must be finite and nonnegative')
        for g0 in (g0_roof, g0_wall):
            if not math.isfinite(float(g0)) or abs(float(g0)) > 1000.0:
                raise ValueError('prescribed baseline flux G0 must be finite '
                                 'and within +/-1000 W/m2')
        self.out_mean = float(output_scaler.mean_[0])
        self.out_scale = float(output_scaler.scale_[0])
        if (not math.isfinite(self.out_mean) or
                not math.isfinite(self.out_scale) or self.out_scale <= 0):
            raise ValueError('output scaler must have finite positive scale')
        self.register_buffer('h_roof', torch.tensor(float(h_roof)))
        self.register_buffer('h_wall', torch.tensor(float(h_wall)))
        self.register_buffer('c_areal', torch.tensor(float(c_areal)))
        self.register_buffer('resid_scale', torch.tensor(float(resid_scale)))
        self.register_buffer('residual_floor', torch.tensor(float(residual_floor)))
        self.register_buffer('g0_roof', torch.tensor(float(g0_roof)))
        self.register_buffer('g0_wall', torch.tensor(float(g0_wall)))
        self.register_buffer('receiver_eps', torch.tensor(float(EPS_CAT[0])))

    def _temperature(self, pred_scaled):
        return pred_scaled.float().flatten() * self.out_scale + self.out_mean

    def forward(self, pred_receiver, pred_emitter, pred_prev, pred_next,
                ray_index, emitter_emissivity, qsw, receiver_normal_z,
                ambient_temperature, dt_seconds, confidence,
                return_details=False):
        Ts = self._temperature(pred_receiver)
        Te = self._temperature(pred_emitter)
        Tp = self._temperature(pred_prev)
        Tn = self._temperature(pred_next)
        qsw = qsw.float().flatten()
        confidence = confidence.float().flatten()
        normal_z = receiver_normal_z.float().flatten()
        emitter_emissivity = emitter_emissivity.float().flatten()
        ray_index = ray_index.long()

        if ray_index.ndim != 2 or ray_index.shape[0] != len(Ts):
            raise ValueError('ray_index must have shape (receivers, rays)')
        if (len(Tp) != len(Ts) or len(Tn) != len(Ts) or
                len(qsw) != len(Ts) or len(confidence) != len(Ts) or
                len(normal_z) != len(Ts)):
            raise ValueError('receiver flux/weight shape mismatch')
        if not torch.isfinite(confidence).all() or (confidence < 0).any():
            raise ValueError('confidence must be finite and nonnegative')
        if (not torch.isfinite(qsw).all() or not torch.isfinite(normal_z).all()
                or (normal_z.abs() > 1.001).any()):
            raise ValueError('receiver flux/normals are invalid')

        incoming_lw = torch.zeros_like(Ts)
        positive = ray_index >= 0
        if positive.any():
            if (len(Te) == 0 or len(emitter_emissivity) != len(Te) or
                    not torch.isfinite(emitter_emissivity).all() or
                    (emitter_emissivity < 0).any() or
                    (emitter_emissivity > 1).any()):
                raise ValueError('ray hits require matching emitter predictions')
            if int(ray_index[positive].max()) >= len(Te):
                raise ValueError('ray_index exceeds emitter prediction array')
            radiance = (emitter_emissivity * SIGMA_SB *
                        (Te + 273.15).pow(4))
            gathered = torch.zeros(ray_index.shape, dtype=Ts.dtype, device=Ts.device)
            gathered[positive] = radiance[ray_index[positive]]
            incoming_lw = gathered.sum(dim=1) / ray_index.shape[1]

        outgoing_lw = SIGMA_SB * (Ts + 273.15).pow(4)
        q_lw_net = self.receiver_eps * (incoming_lw - outgoing_lw)
        h = torch.where(normal_z > ROOF_NORMAL_Z, self.h_roof, self.h_wall)
        if ambient_temperature.numel() != 1:
            raise ValueError('ambient_temperature must be one scalar')
        Ta = ambient_temperature.float().to(Ts.device)
        if not torch.isfinite(Ta).all():
            raise ValueError('ambient_temperature must be finite')
        q_conv = h * (Ts - Ta)

        if dt_seconds.numel() != 1 or not torch.isfinite(dt_seconds).all():
            raise ValueError('dt_seconds must be one positive scalar')
        dt_value = float(dt_seconds.detach().cpu())
        if dt_value <= 0:
            raise ValueError('dt_seconds must be one positive scalar')
        dTdt = (Tn - Tp) / dt_seconds.float()
        q_storage = self.c_areal * dTdt

        # Fixed roof/wall closure flux G0 (mean substrate/interior heat
        # exchange not resolved by the reduced balance).
        g0 = torch.where(normal_z > ROOF_NORMAL_Z, self.g0_roof, self.g0_wall)
        residual = qsw + q_lw_net + g0 - q_conv - q_storage
        # Penalize only the excess above the closure threshold; below it
        # the physics gradient is exactly zero.
        excess = (residual.abs() - self.residual_floor).clamp_min(0.0)

        weight_sum = confidence.sum()
        if float(weight_sum.detach().cpu()) <= 0:
            raise ValueError('physics confidence has zero total weight')
        loss = (confidence * (excess / self.resid_scale).pow(2)).sum() / weight_sum
        if not torch.isfinite(loss):
            raise FloatingPointError('non-finite SEB loss')

        if not return_details:
            return loss
        rms = torch.sqrt((confidence * residual.pow(2)).sum() / weight_sum)
        return loss, {
            'residual': residual,
            'excess': excess,
            'qsw': qsw,
            'incoming_lw': incoming_lw,
            'q_lw_net': q_lw_net,
            'q_conv': q_conv,
            'q_storage': q_storage,
            'dTdt': dTdt,
            'rms': rms,
        }
