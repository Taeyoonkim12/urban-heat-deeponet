"""Geometry-only ray operator for the M5 surface-energy-balance loss.

The operator stores receiver geometry, fixed shadow visibility, ray-to-emitter
indices and QC metadata. It never reads a CFD temperature field. During
training, all receiver and emitter temperatures used by the physics loss are
evaluated by the neural network.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import config
from .data import build_trunk_array
from .geometry import compute_dbp, load_building_points
from .seb import SurfaceEnergyBalanceLoss
from .solar import (EPS_CAT, HOUR_TO_ROW, RHO_CAT, SUN_DIF, SUN_DNI,
                    SUN_VECS, solar_branch)


def _read_xyz(path):
    df = pd.read_csv(path)
    named = ['X (m)', 'Y (m)', 'Z (m)']
    if all(c in df.columns for c in named):
        return df[named].values.astype(np.float64)
    if df.shape[1] < 4:
        raise ValueError(f'cannot identify XYZ columns in {path}')
    # float64: digests must match scripts/prepare_physics_operator.py exactly.
    return df.iloc[:, [1, 2, 3]].values.astype(np.float64)


def _validate_domain_xyz(xyz, label, strict=True):
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz):
        raise ValueError(f'{label} must be a nonempty (N, 3) array')
    if not np.isfinite(xyz).all():
        raise ValueError(f'{label} contains non-finite coordinates')
    g = config.GLOBAL_COORD_RANGES
    valid = ((xyz[:, 0] >= g['x_min']) & (xyz[:, 0] <= g['x_max']) &
             (xyz[:, 1] >= g['y_min']) & (xyz[:, 1] <= g['y_max']) &
             (xyz[:, 2] >= g['z_min']) & (xyz[:, 2] <= g['z_max']))
    outside = int((~valid).sum())
    if outside and strict:
        raise ValueError(
            f'{label} has {outside} points outside the model domain')
    if outside:
        # Emitters may legitimately extend beyond the crop (e.g. water
        # surfaces): the diagnostic ray cache was built on the full CSV
        # point set, so the operator must keep the identical geometry.
        print(f'[report] {label}: {outside} points outside the crop domain '
              '(kept; diagnostic-identical emitter set)')
    return outside


def _array_digest(array, decimals=6):
    value = np.round(np.asarray(array, dtype=np.float64), decimals)
    value = np.ascontiguousarray(value)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(value.shape).encode())
    digest.update(str(value.dtype).encode())
    digest.update(memoryview(value).cast('B'))
    return digest.hexdigest()


def _file_sha256(path, chunk_bytes=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            block = stream.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def absorbed_shortwave(shortwave_normal, shadow_lit, sky_view, hour):
    """Diagnostic-consistent absorbed shortwave for Building receivers."""
    normal = np.asarray(shortwave_normal, dtype=np.float64)
    lit = np.asarray(shadow_lit, dtype=np.float64).reshape(-1)
    sky_view = np.asarray(sky_view, dtype=np.float64).reshape(-1)
    if normal.ndim != 2 or normal.shape[1] != 3:
        raise ValueError('shortwave_normal must have shape (receivers, 3)')
    if len(lit) != len(normal) or len(sky_view) != len(normal):
        raise ValueError('shortwave arrays have inconsistent receiver counts')
    if (not np.isfinite(normal).all() or not np.isfinite(lit).all() or
            not np.isfinite(sky_view).all()):
        raise ValueError('shortwave inputs must be finite')
    if not np.allclose(np.linalg.norm(normal, axis=1), 1.0, atol=2e-3):
        raise ValueError('shortwave normals must be unit vectors')
    if ((lit < 0) | (lit > 1)).any() or (
            (sky_view < 0) | (sky_view > 1)).any():
        raise ValueError('shadow visibility and sky view must be within [0, 1]')
    h = int(round(float(hour)))
    if h not in HOUR_TO_ROW or not np.isclose(float(hour), h):
        raise ValueError(f'hour {hour} is outside the verified solar table')
    row = HOUR_TO_ROW[h]
    cos_inc = np.maximum(0.0, normal @ SUN_VECS[row])
    qsw = ((1.0 - RHO_CAT[0]) *
           (SUN_DNI[row] * cos_inc * lit + SUN_DIF[row] * sky_view))
    return qsw.astype(np.float32)


def _weighted_stats(values, weights, mask=None):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        values, weights = values[mask], weights[mask]
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[valid], weights[valid]
    if not len(values) or weights.sum() <= 0:
        return None
    weights = weights / weights.sum()
    absolute = np.abs(values)
    order = np.argsort(absolute, kind='stable')
    cumulative = np.cumsum(weights[order])

    def quantile(probability):
        index = min(np.searchsorted(cumulative, probability, side='left'),
                    len(order) - 1)
        return float(absolute[order[index]])

    return {
        'n': int(len(values)),
        'rms_wm2': float(np.sqrt(np.sum(weights * values ** 2))),
        'mae_wm2': float(np.sum(weights * absolute)),
        'p90_abs_wm2': quantile(0.90),
        'p95_abs_wm2': quantile(0.95),
        'bias_wm2': float(np.sum(weights * values)),
    }


def _companion(manifest_path, prefix, tag, suffix='.npy'):
    path = manifest_path.parent / f'{prefix}_{tag}{suffix}'
    if not path.exists():
        raise FileNotFoundError(f'ray-operator companion is missing: {path}')
    return path


class RayPhysicsOperator:
    """Validated memory-mapped ray geometry and target-independent QC."""

    def __init__(self, manifest_path, data_root, meta_path=None):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.data_root = Path(data_root).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        if not self.manifest_path.name.startswith('ray_manifest_'):
            raise ValueError('ray manifest filename must start with ray_manifest_')

        with self.manifest_path.open(encoding='utf-8') as f:
            manifest = json.load(f)
        if manifest.get('complete') is not True:
            raise ValueError('ray manifest is not marked complete')
        self.manifest = manifest
        self.tag = self.manifest_path.stem[len('ray_manifest_'):]
        self.geometry_signature = str(manifest.get('geometry_signature', ''))
        self.n_ray = int(manifest['n_ray'])
        if not self.geometry_signature or self.n_ray <= 0:
            raise ValueError('manifest geometry signature/ray count is invalid')
        self.sky_code = int(manifest.get('sky_code', config.RAY_SKY))
        self.unresolved_code = int(
            manifest.get('unresolved_code', config.RAY_UNRESOLVED))
        if self.sky_code != config.RAY_SKY or self.unresolved_code != config.RAY_UNRESOLVED:
            raise ValueError('ray sentinel values do not match the experiment contract')

        self.ray_eid_path = _companion(
            self.manifest_path, 'ray_eid', self.tag)
        self.ray_hit_category_path = _companion(
            self.manifest_path, 'ray_hit_category', self.tag)
        self.ray_eid = np.load(self.ray_eid_path, mmap_mode='r')
        self.ray_hit_category = np.load(
            self.ray_hit_category_path, mmap_mode='r')
        if self.ray_eid.shape != self.ray_hit_category.shape:
            raise ValueError('ray ID/category arrays have different shapes')
        if self.ray_eid.ndim != 2 or self.ray_eid.shape[1] != self.n_ray:
            raise ValueError('ray arrays do not match manifest n_ray')
        manifest_shape = tuple(manifest.get('shape', self.ray_eid.shape))
        if manifest_shape != self.ray_eid.shape:
            raise ValueError('ray arrays do not match manifest shape')

        shadow_path = self.manifest_path.parent / (
            f'shadow_public_mapped_{self.geometry_signature}.npz')
        if not shadow_path.exists():
            raise FileNotFoundError(f'mapped receiver/shadow file is missing: {shadow_path}')
        shadow = np.load(shadow_path, allow_pickle=False)
        receiver_xyz64 = np.asarray(shadow['receiver_xyz'], dtype=np.float64)
        self.receiver_xyz = receiver_xyz64.astype(np.float32)
        self.receiver_shortwave_normal = shadow[
            'receiver_normal_public'].astype(np.float32)
        raw_lit = np.asarray(shadow['lit'])
        if (not np.isfinite(raw_lit).all() or
                not np.isin(raw_lit, (0, 1, False, True)).all()):
            raise ValueError('shadow visibility must contain only 0/1 values')
        self.shadow_lit = raw_lit.astype(bool)
        n_receiver = self.ray_eid.shape[0]
        if self.receiver_xyz.shape != (n_receiver, 3):
            raise ValueError('receiver coordinate shape mismatch')
        _validate_domain_xyz(self.receiver_xyz, 'receiver geometry')
        if self.receiver_shortwave_normal.shape != (n_receiver, 3):
            raise ValueError('shortwave receiver normal shape mismatch')
        if self.shadow_lit.shape != (12, n_receiver):
            raise ValueError('shadow visibility must have shape (12, receivers)')
        shortwave_norm = np.linalg.norm(
            self.receiver_shortwave_normal, axis=1)
        if (not np.isfinite(self.receiver_shortwave_normal).all() or
                not np.allclose(shortwave_norm, 1.0, atol=2e-3)):
            raise ValueError('shortwave receiver normals are not normalized')

        xyz_parts, cat_parts = [], []
        for cid, name in enumerate(config.CATEGORY_NAMES):
            xyz = _read_xyz(self.data_root / f'{name}.csv')
            _validate_domain_xyz(xyz, f'{name} geometry', strict=False)
            xyz_parts.append(xyz)
            cat_parts.append(np.full(len(xyz), cid, dtype=np.int64))
        emitter_xyz64 = np.concatenate(xyz_parts)
        self.emitter_xyz = emitter_xyz64.astype(np.float32)
        self.emitter_category = np.concatenate(cat_parts)

        # Validate memory-mapped arrays in bounded chunks. Full boolean advanced
        # indexing can otherwise materialize the entire ray cache in RAM.
        ray_resolved_fraction = np.empty(n_receiver, dtype=np.float32)
        ray_coarse_fraction = np.empty(n_receiver, dtype=np.float32)
        ray_category_consistency = np.empty(n_receiver, dtype=np.float32)
        for start in range(0, n_receiver, 50_000):
            stop = min(start + 50_000, n_receiver)
            op = np.asarray(self.ray_eid[start:stop])
            hcat = np.asarray(self.ray_hit_category[start:stop])
            invalid_negative = ((op < 0) & (op != self.sky_code) &
                                (op != self.unresolved_code))
            if invalid_negative.any():
                raise ValueError('ray array contains an unknown negative sentinel')
            if not np.all(hcat[op == self.sky_code] == self.sky_code):
                raise ValueError('sky ray/category sentinels disagree')
            positive = op >= 0
            consistency = np.ones(op.shape, dtype=np.float32)
            if positive.any():
                emitter_id = op[positive]
                if int(emitter_id.max()) >= len(self.emitter_xyz):
                    raise ValueError(
                        'ray emitter ID exceeds concatenated category geometry')
                raw_category = hcat[positive].astype(np.int64)
                if ((raw_category < 0) | (raw_category >= 20)).any():
                    raise ValueError('resolved ray has an invalid hit category')
                hit_cat = raw_category % 10
                expected_category = self.emitter_category[emitter_id]
                consistency[positive] = (hit_cat == expected_category)
                if not np.array_equal(hit_cat, expected_category):
                    raise ValueError(
                        'ray-hit category disagrees with emitter ID category')
            ray_resolved_fraction[start:stop] = (
                op != self.unresolved_code).mean(axis=1)
            ray_coarse_fraction[start:stop] = (
                (hcat >= 10) & positive).mean(axis=1)
            ray_category_consistency[start:stop] = consistency.mean(axis=1)

        if meta_path is None:
            meta_path = self.manifest_path.parent / f'physics_operator_meta_{self.tag}.npz'
        self.meta_path = Path(meta_path).expanduser().resolve()
        if not self.meta_path.exists():
            raise FileNotFoundError(
                f'physics-operator metadata is missing: {self.meta_path}; '
                'run scripts/prepare_physics_operator.py')
        meta = np.load(self.meta_path, allow_pickle=False)
        if str(meta['geometry_signature'].item()) != self.geometry_signature:
            raise ValueError('operator metadata geometry signature mismatch')
        if int(meta['n_ray'].item()) != self.n_ray:
            raise ValueError('operator metadata ray count mismatch')
        emitter_digest = str(meta['emitter_geometry_digest'].item())
        receiver_digest = str(meta['receiver_geometry_digest'].item())
        # Digests are defined on the float64 on-disk values (same as the
        # prepare script); float32 runtime copies would round differently at
        # real coordinate magnitudes (~3000 m).
        if emitter_digest != _array_digest(emitter_xyz64):
            raise ValueError('category point geometry differs from operator metadata')
        if receiver_digest != _array_digest(receiver_xyz64):
            raise ValueError('receiver geometry differs from operator metadata')
        if 'artifact_sha256_json' not in meta:
            raise ValueError('operator metadata has no artifact SHA-256 record')
        recorded_hashes = json.loads(str(meta['artifact_sha256_json'].item()))
        artifact_paths = {
            'manifest': self.manifest_path,
            'ray_eid': self.ray_eid_path,
            'ray_hit_category': self.ray_hit_category_path,
            'shadow': shadow_path,
        }
        if set(recorded_hashes) != set(artifact_paths):
            raise ValueError('operator artifact SHA-256 record is incomplete')
        for name, path in artifact_paths.items():
            if _file_sha256(path) != recorded_hashes[name]:
                raise ValueError(f'physics artifact changed after QC: {path}')
        self.artifact_sha256 = recorded_hashes
        self.meta_sha256 = _file_sha256(self.meta_path)
        if 'receiver_normal' not in meta:
            raise ValueError('operator metadata has no verified receiver normals')
        self.receiver_normal = meta['receiver_normal'].astype(np.float32)
        if self.receiver_normal.shape != (n_receiver, 3):
            raise ValueError('verified receiver normal shape mismatch')
        normal_norm = np.linalg.norm(self.receiver_normal, axis=1)
        if (not np.isfinite(self.receiver_normal).all() or
                not np.allclose(normal_norm, 1.0, atol=2e-3)):
            raise ValueError('verified receiver normals are not normalized')
        if 'receiver_shortwave_normal' not in meta:
            raise ValueError(
                'operator metadata has no diagnostic shortwave normals')
        meta_shortwave_normal = meta[
            'receiver_shortwave_normal'].astype(np.float32)
        if (meta_shortwave_normal.shape != (n_receiver, 3) or
                not np.allclose(meta_shortwave_normal,
                                self.receiver_shortwave_normal, atol=1e-6)):
            raise ValueError(
                'shadow and metadata shortwave normals disagree')

        required = ('receiver_area_weight', 'receiver_confidence',
                    'resolved_fraction', 'coarse_fraction',
                    'category_consistency', 'mapping_quality',
                    'normal_quality')
        for name in required:
            arr = np.asarray(meta[name])
            if arr.shape != (n_receiver,):
                raise ValueError(f'{name} shape mismatch')
            setattr(self, name, arr.astype(np.float32))
        for name in ('receiver_confidence', 'resolved_fraction',
                     'coarse_fraction', 'category_consistency',
                     'mapping_quality', 'normal_quality'):
            arr = getattr(self, name)
            if not (np.isfinite(arr).all() and (arr >= 0).all() and (arr <= 1).all()):
                raise ValueError(f'{name} must be finite and within [0, 1]')
        for name, recomputed in (
                ('resolved_fraction', ray_resolved_fraction),
                ('coarse_fraction', ray_coarse_fraction),
                ('category_consistency', ray_category_consistency)):
            if not np.allclose(
                    getattr(self, name), recomputed, rtol=0.0, atol=1e-6):
                raise ValueError(f'stored {name} disagrees with ray arrays')
        expected_formula = (
            'resolved_fraction*(1-coarse_fraction)*category_consistency*'
            'normal_quality*mapping_quality')
        if ('formula' not in meta or
                str(meta['formula'].item()) != expected_formula):
            raise ValueError('operator confidence formula is not recognized')
        expected_confidence = (
            self.resolved_fraction * (1.0 - self.coarse_fraction) *
            self.category_consistency * self.normal_quality *
            self.mapping_quality)
        if not np.allclose(
                self.receiver_confidence, expected_confidence,
                rtol=1e-5, atol=1e-6):
            raise ValueError(
                'stored receiver confidence does not match its QC components')
        if not (np.isfinite(self.receiver_area_weight).all() and
                (self.receiver_area_weight >= 0).all() and
                (self.receiver_area_weight > 0).any()):
            raise ValueError('receiver area weights are invalid')

        if 'receiver_dbp' in meta and 'emitter_dbp' in meta:
            self.receiver_dbp = meta['receiver_dbp'].astype(np.float32)
            self.emitter_dbp = meta['emitter_dbp'].astype(np.float32)
        else:
            building_tree, building_z = load_building_points(
                self.data_root / 'Building.csv')
            self.receiver_dbp = compute_dbp(
                self.receiver_xyz, building_tree, building_z)
            self.emitter_dbp = compute_dbp(
                self.emitter_xyz, building_tree, building_z)
        if self.receiver_dbp.shape != (n_receiver,):
            raise ValueError('receiver d_BP shape mismatch')
        if self.emitter_dbp.shape != (len(self.emitter_xyz),):
            raise ValueError('emitter d_BP shape mismatch')

        p = self.receiver_area_weight.astype(np.float64)
        p[~np.isfinite(p) | (p <= 0)] = 0.0
        self.sampling_probability = p / p.sum()

    def describe(self):
        c = self.receiver_confidence
        return (f'receivers={len(self.receiver_xyz):,}, rays={self.n_ray}, '
                f'confidence mean/p05={c.mean():.3f}/'
                f'{np.percentile(c, 5):.3f}, tag={self.tag}')

    def sample(self, n_points, rng, hour, weighting):
        """Area-sample receivers; confidence never changes sampling."""
        if weighting not in ('uniform', 'confidence'):
            raise ValueError(f'unknown physics weighting: {weighting}')
        positive_area = int((self.sampling_probability > 0).sum())
        n = min(int(n_points), positive_area)
        if n <= 0:
            raise ValueError('physics sample size must be positive')
        receiver_id = rng.choice(
            len(self.receiver_xyz), n, replace=False,
            p=self.sampling_probability)
        op = np.asarray(self.ray_eid[receiver_id], dtype=np.int64)

        ray_index = op.copy()
        positive = op >= 0
        unique_emitters, inverse = np.unique(op[positive], return_inverse=True)
        ray_index[positive] = inverse

        h = int(round(float(hour)))
        if h not in HOUR_TO_ROW or not np.isclose(float(hour), h):
            raise ValueError(f'hour {hour} is outside the verified solar table')
        row = HOUR_TO_ROW[h]
        normal = self.receiver_normal[receiver_id]
        shortwave_normal = self.receiver_shortwave_normal[receiver_id]
        sky_view = (op == self.sky_code).mean(axis=1).astype(np.float32)
        lit = self.shadow_lit[row, receiver_id].astype(np.float32)
        qsw = absorbed_shortwave(
            shortwave_normal, lit, sky_view, hour)

        confidence = (np.ones(n, dtype=np.float32) if weighting == 'uniform'
                      else self.receiver_confidence[receiver_id])
        return {
            'receiver_id': receiver_id.astype(np.int64),
            'receiver_xyz': self.receiver_xyz[receiver_id],
            'receiver_normal': normal,
            'receiver_shortwave_normal': shortwave_normal,
            'receiver_dbp': self.receiver_dbp[receiver_id],
            'receiver_category': np.zeros(n, dtype=np.int64),
            'emitter_id': unique_emitters.astype(np.int64),
            'emitter_xyz': self.emitter_xyz[unique_emitters],
            'emitter_dbp': self.emitter_dbp[unique_emitters],
            'emitter_category': self.emitter_category[unique_emitters],
            'ray_index': ray_index.astype(np.int64),
            'qsw': qsw.astype(np.float32),
            'sky_view': sky_view,
            'sunlit': lit.astype(bool),
            'is_roof': (normal[:, 2] > config.ROOF_NORMAL_Z),
            'confidence': confidence.astype(np.float32),
            'resolved_fraction': self.resolved_fraction[receiver_id],
            'coarse_fraction': self.coarse_fraction[receiver_id],
        }


def resolve_operator_manifest(path):
    """Resolve an explicit manifest or a directory containing exactly one."""
    path = Path(path).expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(path)
    candidates = sorted(path.glob('ray_manifest_*.json'))
    if len(candidates) != 1:
        raise ValueError(
            f'expected exactly one ray manifest in {path}, found {len(candidates)}')
    return candidates[0]


def _branch_array(params, scalers, n):
    raw = solar_branch(params).reshape(1, -1)
    row = scalers['branch'].transform(raw).astype(np.float32)
    return np.repeat(row, n, axis=0)


def _forward_chunks(model, branch, trunk, category, chunk):
    if len(trunk) == 0:
        return torch.empty((0, 1), dtype=trunk.dtype, device=trunk.device)
    out = []
    for start in range(0, len(trunk), chunk):
        stop = min(start + chunk, len(trunk))
        out.append(model(branch[start:stop], trunk[start:stop],
                         None if category is None else category[start:stop]))
    return torch.cat(out, dim=0)


def adjacent_hours(hour):
    """Return the verified finite-difference stencil and interval in seconds."""
    hour = float(hour)
    if hour < 7.0 or hour > 18.0 or not np.isclose(hour, round(hour)):
        raise ValueError(f'hour {hour} is outside the integer 07-18 sequence')
    previous = max(7.0, hour - 1.0)
    following = min(18.0, hour + 1.0)
    return previous, following, (following - previous) * 3600.0


class RayPhysicsRegularizer:
    """Build leakage-safe M5 physics minibatches and evaluate the SEB loss."""

    def __init__(self, operator, scalers, spec, device, weighting,
                 n_points, h_roof, h_wall, c_areal, resid_scale,
                 residual_floor=0.0, g0_roof=0.0, g0_wall=0.0,
                 chunk_size=config.CHUNK_SIZE):
        if spec.get('branch') != 'solar':
            raise ValueError('ray physics requires the solar branch encoding')
        if weighting not in ('uniform', 'confidence'):
            raise ValueError(f'unknown physics weighting: {weighting}')
        if int(n_points) <= 0 or int(chunk_size) <= 0:
            raise ValueError('physics point and chunk counts must be positive')
        self.operator = operator
        self.scalers = scalers
        self.spec = spec
        self.device = device
        self.weighting = weighting
        self.n_points = int(n_points)
        self.chunk_size = int(chunk_size)
        self.loss_module = SurfaceEnergyBalanceLoss(
            scalers['output'], h_roof=h_roof, h_wall=h_wall,
            c_areal=c_areal, resid_scale=resid_scale,
            residual_floor=residual_floor,
            g0_roof=g0_roof, g0_wall=g0_wall).to(device)

    def _inputs(self, xyz, dbp, category, params):
        trunk, category = build_trunk_array(
            xyz, dbp, category, self.scalers, self.spec)
        branch = _branch_array(params, self.scalers, len(trunk))
        branch = torch.from_numpy(branch).to(self.device)
        trunk = torch.from_numpy(trunk).to(self.device)
        cat = (torch.from_numpy(category).to(self.device)
               if self.spec['use_category'] else None)
        return branch, trunk, cat, category

    def __call__(self, model, params, rng):
        batch = self.operator.sample(
            self.n_points, rng, params['hour'], self.weighting)
        hour = float(params['hour'])
        h_prev, h_next, dt_seconds = adjacent_hours(hour)
        if dt_seconds <= 0:
            raise ValueError('invalid adjacent-hour interval')

        recv = self._inputs(
            batch['receiver_xyz'], batch['receiver_dbp'],
            batch['receiver_category'], params)
        emit = self._inputs(
            batch['emitter_xyz'], batch['emitter_dbp'],
            batch['emitter_category'], params)
        p_prev = dict(params, hour=h_prev)
        p_next = dict(params, hour=h_next)
        b_prev = torch.from_numpy(
            _branch_array(p_prev, self.scalers, len(batch['receiver_xyz']))).to(self.device)
        b_next = torch.from_numpy(
            _branch_array(p_next, self.scalers, len(batch['receiver_xyz']))).to(self.device)

        was_training = model.training
        model.eval()  # deterministic finite difference; gradients remain enabled
        try:
            pred_recv = _forward_chunks(model, recv[0], recv[1], recv[2], self.chunk_size)
            pred_emit = _forward_chunks(model, emit[0], emit[1], emit[2], self.chunk_size)
            pred_prev = _forward_chunks(model, b_prev, recv[1], recv[2], self.chunk_size)
            pred_next = _forward_chunks(model, b_next, recv[1], recv[2], self.chunk_size)
        finally:
            if was_training:
                model.train()

        emitter_eps = torch.from_numpy(EPS_CAT[emit[3]].astype(np.float32)).to(self.device)
        loss, details = self.loss_module(
            pred_recv, pred_emit, pred_prev, pred_next,
            torch.from_numpy(batch['ray_index']).to(self.device),
            emitter_eps,
            torch.from_numpy(batch['qsw']).to(self.device),
            torch.from_numpy(batch['receiver_normal'][:, 2]).to(self.device),
            torch.tensor(float(params['temperature']), dtype=torch.float32,
                         device=self.device),
            torch.tensor(dt_seconds, dtype=torch.float32, device=self.device),
            torch.from_numpy(batch['confidence']).to(self.device),
            return_details=True)
        residual = details['residual'].detach().cpu().numpy()
        stats = _weighted_stats(residual, batch['confidence'])
        if stats is None:
            raise RuntimeError('physics sample has no positive finite weight')
        summary = {
            'loss': float(loss.detach().cpu()),
            **stats,
            'confidence_mean': float(np.mean(batch['confidence'])),
            'resolved_mean': float(np.mean(batch['resolved_fraction'])),
            'coarse_mean': float(np.mean(batch['coarse_fraction'])),
            'groups': {
                'roof': _weighted_stats(
                    residual, batch['confidence'], batch['is_roof']),
                'wall': _weighted_stats(
                    residual, batch['confidence'], ~batch['is_roof']),
                'sunlit': _weighted_stats(
                    residual, batch['confidence'], batch['sunlit']),
                'shaded': _weighted_stats(
                    residual, batch['confidence'], ~batch['sunlit']),
            },
        }
        return loss, summary
