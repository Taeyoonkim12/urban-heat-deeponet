"""Build target-independent area and QC metadata for an existing ray cache.

This script consumes only geometry, ray IDs/categories, receiver normals and
shadow visibility. It never opens a CFD case file or temperature column.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urbanheat import config
from urbanheat.geometry import compute_dbp, load_building_points


def read_xyz(path):
    df = pd.read_csv(path)
    named = ['X (m)', 'Y (m)', 'Z (m)']
    if all(c in df.columns for c in named):
        return df[named].values.astype(np.float64)
    if df.shape[1] < 4:
        raise ValueError(f'cannot identify XYZ columns in {path}')
    return df.iloc[:, [1, 2, 3]].values.astype(np.float64)


def digest_array(array, decimals=None):
    value = np.asarray(array)
    if decimals is not None:
        value = np.round(value.astype(np.float64), decimals)
    value = np.ascontiguousarray(value)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(value.shape).encode())
    digest.update(str(value.dtype).encode())
    digest.update(memoryview(value).cast('B'))
    return digest.hexdigest()


def file_sha256(path, chunk_bytes=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            block = stream.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def geometry_signature(mesh, category_points):
    digest = hashlib.blake2b(digest_size=16)
    digest.update(digest_array(mesh.vertices, 6).encode())
    digest.update(digest_array(mesh.faces).encode())
    digest.update(digest_array(category_points, 6).encode())
    return digest.hexdigest()


def require_domain_xyz(xyz, label, strict=True):
    g = config.GLOBAL_COORD_RANGES
    valid = ((xyz[:, 0] >= g['x_min']) & (xyz[:, 0] <= g['x_max']) &
             (xyz[:, 1] >= g['y_min']) & (xyz[:, 1] <= g['y_max']) &
             (xyz[:, 2] >= g['z_min']) & (xyz[:, 2] <= g['z_max']))
    outside = int((~valid).sum())
    if outside and strict:
        raise SystemExit(
            f'{label} has {outside} points outside the model domain')
    if outside:
        print(f'[report] {label}: {outside} points outside the crop domain '
              '(kept; diagnostic-identical emitter set)')


def companion(manifest_path, prefix, tag):
    path = manifest_path.parent / f'{prefix}_{tag}.npy'
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--stl', default=None)
    parser.add_argument('--shadow', default=None)
    parser.add_argument('--out', default=None)
    parser.add_argument('--face-map-max', type=float, default=1.0)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if not np.isfinite(args.face_map_max) or args.face_map_max <= 0:
        raise SystemExit('--face-map-max must be finite and positive')

    manifest_path = Path(args.manifest).expanduser().resolve()
    if manifest_path.is_dir():
        candidates = sorted(manifest_path.glob('ray_manifest_*.json'))
        if len(candidates) != 1:
            raise SystemExit(f'expected exactly one ray manifest in '
                             f'{manifest_path}, found {len(candidates)}')
        manifest_path = candidates[0]
    data_root = Path(args.data_root).expanduser().resolve()
    with manifest_path.open(encoding='utf-8') as f:
        manifest = json.load(f)
    if manifest.get('complete') is not True:
        raise SystemExit('ray manifest is not marked complete')
    if not manifest_path.name.startswith('ray_manifest_'):
        raise SystemExit('manifest filename must start with ray_manifest_')
    tag = manifest_path.stem[len('ray_manifest_'):]
    n_ray = int(manifest['n_ray'])
    if not str(manifest.get('geometry_signature', '')) or n_ray <= 0:
        raise SystemExit('manifest geometry signature/ray count is invalid')
    sky_code = int(manifest.get('sky_code', config.RAY_SKY))
    unresolved_code = int(manifest.get('unresolved_code', config.RAY_UNRESOLVED))
    if sky_code != config.RAY_SKY or unresolved_code != config.RAY_UNRESOLVED:
        raise SystemExit('ray sentinel values do not match the code contract')

    ray_eid_path = companion(manifest_path, 'ray_eid', tag)
    hit_category_path = companion(manifest_path, 'ray_hit_category', tag)
    ray_eid = np.load(ray_eid_path, mmap_mode='r')
    hit_category = np.load(hit_category_path, mmap_mode='r')
    manifest_shape = tuple(manifest.get('shape', ()))
    if (ray_eid.ndim != 2 or ray_eid.shape != hit_category.shape or
            ray_eid.shape[1] != n_ray or manifest_shape != ray_eid.shape):
        raise SystemExit('ray companion arrays have inconsistent shapes')

    stl_path = Path(args.stl).expanduser().resolve() if args.stl else data_root / 'building.stl'
    mesh = trimesh.load(str(stl_path), force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        raise SystemExit('building STL did not load as a single Trimesh')
    try:
        mesh.fix_normals()
    except Exception:
        pass
    if (not np.isfinite(mesh.face_normals).all() or
            not np.isfinite(mesh.area_faces).all() or
            (mesh.area_faces <= 0).any()):
        raise SystemExit('building STL has invalid normals or face areas')

    xyz_parts, cat_parts = [], []
    for cid, name in enumerate(config.CATEGORY_NAMES):
        xyz = read_xyz(data_root / f'{name}.csv')
        if not len(xyz) or not np.isfinite(xyz).all():
            raise SystemExit(f'{name} geometry is empty or non-finite')
        require_domain_xyz(xyz, f'{name} geometry', strict=False)
        xyz_parts.append(xyz)
        cat_parts.append(np.full(len(xyz), cid, dtype=np.int64))
    emitter_xyz = np.concatenate(xyz_parts)
    emitter_category = np.concatenate(cat_parts)
    signature = geometry_signature(mesh, emitter_xyz)
    if signature != str(manifest.get('geometry_signature', '')):
        raise SystemExit(
            f'geometry signature mismatch: computed {signature}, '
            f"manifest {manifest.get('geometry_signature')}")

    expected_shadow_path = (
        manifest_path.parent / f'shadow_public_mapped_{signature}.npz')
    shadow_path = (Path(args.shadow).expanduser().resolve() if args.shadow else
                   expected_shadow_path)
    if shadow_path != expected_shadow_path.resolve():
        raise SystemExit(
            f'shadow artifact must be colocated with the manifest as '
            f'{expected_shadow_path.name}')
    shadow = np.load(shadow_path, allow_pickle=False)
    receiver_xyz = shadow['receiver_xyz'].astype(np.float64)
    shortwave_normal = shadow['receiver_normal_public'].astype(np.float64)
    if receiver_xyz.shape != (ray_eid.shape[0], 3):
        raise SystemExit('receiver shape does not match ray operator')
    if not np.isfinite(receiver_xyz).all():
        raise SystemExit('receiver geometry contains non-finite coordinates')
    require_domain_xyz(receiver_xyz, 'receiver geometry')
    raw_lit = np.asarray(shadow['lit'])
    if raw_lit.shape != (12, len(receiver_xyz)):
        raise SystemExit('shadow array must have shape (12, receivers)')
    if (not np.isfinite(raw_lit).all() or
            not np.isin(raw_lit, (0, 1, False, True)).all()):
        raise SystemExit('shadow visibility must contain only 0/1 values')
    if shortwave_normal.shape != receiver_xyz.shape:
        raise SystemExit('shortwave normal shape does not match receivers')
    shortwave_norm = np.linalg.norm(shortwave_normal, axis=1)
    if (not np.isfinite(shortwave_normal).all() or
            not np.allclose(shortwave_norm, 1.0, atol=2e-3)):
        raise SystemExit('shortwave normals are not finite unit vectors')

    from trimesh.proximity import closest_point
    distance = np.empty(len(receiver_xyz), dtype=np.float64)
    face_id = np.empty(len(receiver_xyz), dtype=np.int64)
    for start in range(0, len(receiver_xyz), 20_000):
        stop = min(start + 20_000, len(receiver_xyz))
        _, d, f = closest_point(mesh, receiver_xyz[start:stop])
        distance[start:stop] = d
        face_id[start:stop] = f
    valid_face = (np.isfinite(distance) & (distance <= args.face_map_max) &
                  (face_id >= 0) & (face_id < len(mesh.faces)))
    if not valid_face.any():
        raise SystemExit('no receiver maps to a valid STL face')
    face_count = np.bincount(face_id[valid_face], minlength=len(mesh.faces)).astype(np.float64)
    area_weight = np.zeros(len(receiver_xyz), dtype=np.float64)
    area_weight[valid_face] = (mesh.area_faces[face_id[valid_face]] /
                               np.maximum(face_count[face_id[valid_face]], 1.0))
    receiver_normal = np.zeros_like(receiver_xyz)
    receiver_normal[valid_face] = mesh.face_normals[face_id[valid_face]]
    normal_length = np.linalg.norm(receiver_normal, axis=1, keepdims=True)
    receiver_normal[valid_face] /= np.maximum(normal_length[valid_face], 1e-15)

    resolved_fraction = np.empty(len(receiver_xyz), dtype=np.float32)
    coarse_fraction = np.empty(len(receiver_xyz), dtype=np.float32)
    category_consistency = np.empty(len(receiver_xyz), dtype=np.float32)
    for start in range(0, len(receiver_xyz), 50_000):
        stop = min(start + 50_000, len(receiver_xyz))
        op = np.asarray(ray_eid[start:stop])
        hcat = np.asarray(hit_category[start:stop])
        invalid_negative = ((op < 0) & (op != sky_code) &
                            (op != unresolved_code))
        if invalid_negative.any():
            raise SystemExit('ray array contains an unknown negative sentinel')
        if not np.all(hcat[op == sky_code] == sky_code):
            raise SystemExit('sky ray/category sentinels disagree')
        positive = op >= 0
        if positive.any():
            emitter_id = op[positive]
            if int(emitter_id.max()) >= len(emitter_xyz):
                raise SystemExit('ray emitter ID exceeds category geometry')
            raw_category = hcat[positive].astype(np.int64)
            if ((raw_category < 0) | (raw_category >= 20)).any():
                raise SystemExit('resolved ray has an invalid hit category')
            if not np.array_equal(
                    raw_category % 10, emitter_category[emitter_id]):
                raise SystemExit(
                    'ray-hit category disagrees with emitter ID category')
        resolved_fraction[start:stop] = (op != unresolved_code).mean(axis=1)
        coarse_fraction[start:stop] = ((hcat >= 10) & positive).mean(axis=1)
        consistent = np.ones(op.shape, dtype=np.float32)
        consistent[positive] = (
            emitter_category[op[positive]] == (hcat[positive] % 10))
        category_consistency[start:stop] = consistent.mean(axis=1)

    normal_norm = np.linalg.norm(receiver_normal, axis=1)
    normal_quality = np.clip(1.0 - np.abs(normal_norm - 1.0) / 0.01,
                             0.0, 1.0)
    mapping_quality = np.clip(1.0 - distance / args.face_map_max, 0.0, 1.0)
    mapping_quality[~valid_face] = 0.0
    confidence = (resolved_fraction * (1.0 - coarse_fraction) *
                  category_consistency * normal_quality * mapping_quality)
    confidence = np.clip(confidence, 0.0, 1.0).astype(np.float32)
    if not (np.isfinite(confidence).all() and confidence.sum() > 0):
        raise SystemExit('computed confidence is invalid')

    building_tree, building_z = load_building_points(data_root / 'Building.csv')
    receiver_dbp = compute_dbp(receiver_xyz, building_tree, building_z)
    emitter_dbp = compute_dbp(emitter_xyz, building_tree, building_z)

    out = (Path(args.out).expanduser().resolve() if args.out else
           manifest_path.parent / f'physics_operator_meta_{tag}.npz')
    if out.exists() and not args.force:
        raise SystemExit(f'refusing to overwrite {out}; pass --force explicitly')
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out) + '.tmp.npz')
    artifact_hashes = {
        'manifest': file_sha256(manifest_path),
        'ray_eid': file_sha256(ray_eid_path),
        'ray_hit_category': file_sha256(hit_category_path),
        'shadow': file_sha256(shadow_path),
    }
    np.savez_compressed(
        tmp,
        geometry_signature=np.array(signature),
        emitter_geometry_digest=np.array(digest_array(emitter_xyz, 6)),
        receiver_geometry_digest=np.array(digest_array(receiver_xyz, 6)),
        n_ray=np.array(n_ray),
        receiver_area_weight=area_weight.astype(np.float32),
        receiver_normal=receiver_normal.astype(np.float32),
        receiver_shortwave_normal=shortwave_normal.astype(np.float32),
        receiver_confidence=confidence,
        resolved_fraction=resolved_fraction,
        coarse_fraction=coarse_fraction,
        category_consistency=category_consistency,
        mapping_quality=mapping_quality.astype(np.float32),
        normal_quality=normal_quality.astype(np.float32),
        receiver_dbp=receiver_dbp.astype(np.float32),
        emitter_dbp=emitter_dbp.astype(np.float32),
        artifact_sha256_json=np.array(json.dumps(
            artifact_hashes, sort_keys=True)),
        formula=np.array(
            'resolved_fraction*(1-coarse_fraction)*category_consistency*'
            'normal_quality*mapping_quality'))
    os.replace(tmp, out)

    represented_area = mesh.area_faces[face_count > 0].sum() / mesh.area_faces.sum()
    print(f'saved {out}')
    print(f'receivers={len(receiver_xyz):,}, rays={n_ray}, '
          f'area coverage={100*represented_area:.2f}%')
    print(f'confidence mean/p05/min={confidence.mean():.4f}/'
          f'{np.percentile(confidence, 5):.4f}/{confidence.min():.4f}')
    print(f'unresolved ray fraction={1-resolved_fraction.mean():.4%}, '
          f'coarse ray fraction={coarse_fraction.mean():.4%}')


if __name__ == '__main__':
    main()
