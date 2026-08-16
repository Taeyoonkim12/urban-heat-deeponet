"""Generate a fully synthetic dummy dataset for pipeline testing.

The real CFD dataset cannot be redistributed, so this script creates
random data with the same file layout and schema. The values are pure
noise - they carry no physical meaning and reflect no statistics of the
real data.

Usage:
    python scripts/make_dummy_data.py --out dummy_generated
"""

import os
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urbanheat import config
from urbanheat.geometry import compute_dbp, load_building_points
from urbanheat.solar import SUN_VECS

X_RANGE = (1000.0, 3000.0)
Y_RANGE = (-3000.0, -1000.0)
Z_RANGE = (-15.0, 260.0)

CONDITIONS = [(50, 33), (60, 35), (70, 37), (80, 38)]   # (RH %, T_amb C)
HOURS = [12, 15]
POINTS_PER_CASE = 5000

CATEGORY_POINTS = {'Building': 500, 'Road': 200, 'Green': 200,
                   'Water': 200, 'Topo_IN': 200}


def random_points(rng, n):
    x = rng.uniform(*X_RANGE, n)
    y = rng.uniform(*Y_RANGE, n)
    z = rng.uniform(*Z_RANGE, n)
    return x, y, z


def write_cases(out_dir, rng):
    os.makedirs(out_dir, exist_ok=True)
    for rh, ta in CONDITIONS:
        for hr in HOURS:
            x, y, z = random_points(rng, POINTS_PER_CASE)
            temp_k = rng.uniform(295.0, 340.0, POINTS_PER_CASE)  # 22-67 C
            df = pd.DataFrame({'X (m)': x, 'Y (m)': y, 'Z (m)': z,
                               'Temperature (K)': temp_k})
            name = f"Case_{rh}_{ta}_{hr}_XYZInternalTable.csv"
            df.to_csv(os.path.join(out_dir, name), index=False)
    print(f"wrote {len(CONDITIONS) * len(HOURS)} case files to {out_dir}")


def sample_mesh_surface(mesh, rng, n):
    probability = mesh.area_faces / mesh.area_faces.sum()
    face_id = rng.choice(len(mesh.faces), n, replace=True, p=probability)
    triangle = mesh.triangles[face_id]
    a = np.sqrt(rng.random(n))
    b = rng.random(n)
    points = ((1.0 - a)[:, None] * triangle[:, 0] +
              (a * (1.0 - b))[:, None] * triangle[:, 1] +
              (a * b)[:, None] * triangle[:, 2])
    return points, face_id


def write_category_csvs(root, rng, mesh):
    category_xyz = {}
    building_faces = None
    for name, n in CATEGORY_POINTS.items():
        if name == 'Building':
            xyz, building_faces = sample_mesh_surface(mesh, rng, n)
            x, y, z = xyz.T
        else:
            x, y, z = random_points(rng, n)
            xyz = np.column_stack([x, y, z])
        category_xyz[name] = xyz
        df = pd.DataFrame({'Point': np.arange(n),
                           'X (m)': x, 'Y (m)': y, 'Z (m)': z})
        df.to_csv(os.path.join(root, f"{name}.csv"), index=False)
    print(f"wrote category CSVs: {', '.join(CATEGORY_POINTS)}")
    return category_xyz, building_faces


def write_stl(root, rng, n_boxes=12):
    import trimesh
    boxes = []
    for _ in range(n_boxes):
        w, d, h = rng.uniform(20, 80), rng.uniform(20, 80), rng.uniform(20, 120)
        cx = rng.uniform(X_RANGE[0] + 100, X_RANGE[1] - 100)
        cy = rng.uniform(Y_RANGE[0] + 100, Y_RANGE[1] - 100)
        box = trimesh.creation.box(extents=[w, d, h])
        box.apply_translation([cx, cy, h / 2.0])
        boxes.append(box)
    mesh = trimesh.util.concatenate(boxes)
    mesh.export(os.path.join(root, 'building.stl'))
    print(f"wrote building.stl ({len(mesh.faces)} faces)")
    return mesh


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


def file_sha256(path, chunk_bytes=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            block = stream.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_physics_operator(root, rng, mesh, category_xyz, building_faces,
                           n_ray=32):
    """Write a schema-valid synthetic operator for end-to-end tests only."""
    import trimesh
    out = Path(root) / 'physics'
    out.mkdir(parents=True, exist_ok=True)
    # Signature and digests are computed from the artifacts as they exist ON
    # DISK (reloaded STL, re-read CSVs), because that is what the prepare
    # script and the training-time loader will see.
    mesh = trimesh.load(os.path.join(root, 'building.stl'), force='mesh')
    try:
        mesh.fix_normals()
    except Exception:
        pass
    emitter_parts = []
    for name in config.CATEGORY_NAMES:
        df = pd.read_csv(os.path.join(root, f'{name}.csv'))
        emitter_parts.append(
            df[['X (m)', 'Y (m)', 'Z (m)']].values.astype(np.float64))
    emitter_xyz = np.concatenate(emitter_parts)
    emitter_category = np.concatenate([
        np.full(len(category_xyz[name]), cid, dtype=np.int64)
        for cid, name in enumerate(config.CATEGORY_NAMES)])

    digest = hashlib.blake2b(digest_size=16)
    digest.update(digest_array(mesh.vertices, 6).encode())
    digest.update(digest_array(mesh.faces).encode())
    digest.update(digest_array(emitter_xyz, 6).encode())
    signature = digest.hexdigest()

    receiver_xyz = category_xyz['Building'].astype(np.float32)
    receiver_normal = mesh.face_normals[building_faces].astype(np.float32)
    receiver_normal /= np.maximum(
        np.linalg.norm(receiver_normal, axis=1, keepdims=True), 1e-12)
    n_receiver = len(receiver_xyz)

    ray_eid = rng.integers(0, len(emitter_xyz),
                           size=(n_receiver, n_ray), dtype=np.int32)
    draw = rng.random(ray_eid.shape)
    ray_eid[draw < 0.25] = config.RAY_SKY
    ray_eid[(draw >= 0.25) & (draw < 0.30)] = config.RAY_UNRESOLVED
    hit_category = np.full(ray_eid.shape, config.RAY_SKY, dtype=np.int8)
    hit_category[ray_eid == config.RAY_UNRESOLVED] = config.RAY_UNRESOLVED
    positive = ray_eid >= 0
    hit_category[positive] = emitter_category[ray_eid[positive]].astype(np.int8)
    coarse = positive & (hit_category != 0) & (rng.random(ray_eid.shape) < 0.04)
    hit_category[coarse] += 10

    tag = f'{signature}_N{n_ray}_tin2'
    ray_eid_path = out / f'ray_eid_{tag}.npy'
    hit_category_path = out / f'ray_hit_category_{tag}.npy'
    np.save(ray_eid_path, ray_eid)
    np.save(hit_category_path, hit_category)
    manifest = {
        'complete': True, 'geometry_signature': signature,
        'n_ray': n_ray, 'shape': list(ray_eid.shape),
        'sky_code': config.RAY_SKY,
        'unresolved_code': config.RAY_UNRESOLVED,
        'synthetic_schema_test_only': True,
    }
    manifest_path = out / f'ray_manifest_{tag}.json'
    with manifest_path.open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    lit = np.stack([(receiver_normal @ sun) > 0 for sun in SUN_VECS])
    shadow_path = out / f'shadow_public_mapped_{signature}.npz'
    np.savez_compressed(
        shadow_path,
        lit=lit, receiver_xyz=receiver_xyz,
        receiver_normal_public=receiver_normal)

    count = np.bincount(building_faces, minlength=len(mesh.faces)).astype(np.float64)
    area_weight = mesh.area_faces[building_faces] / np.maximum(
        count[building_faces], 1.0)
    resolved_fraction = (ray_eid != config.RAY_UNRESOLVED).mean(axis=1)
    coarse_fraction = coarse.mean(axis=1)
    category_consistency = np.ones(n_receiver, dtype=np.float32)
    confidence = (resolved_fraction * (1.0 - coarse_fraction)).astype(np.float32)
    building_tree, building_z = load_building_points(Path(root) / 'Building.csv')
    receiver_dbp = compute_dbp(receiver_xyz, building_tree, building_z)
    emitter_dbp = compute_dbp(emitter_xyz, building_tree, building_z)
    artifact_hashes = {
        'manifest': file_sha256(manifest_path),
        'ray_eid': file_sha256(ray_eid_path),
        'ray_hit_category': file_sha256(hit_category_path),
        'shadow': file_sha256(shadow_path),
    }
    np.savez_compressed(
        out / f'physics_operator_meta_{tag}.npz',
        geometry_signature=np.array(signature),
        emitter_geometry_digest=np.array(digest_array(emitter_xyz, 6)),
        receiver_geometry_digest=np.array(digest_array(receiver_xyz, 6)),
        n_ray=np.array(n_ray),
        receiver_area_weight=area_weight.astype(np.float32),
        receiver_normal=receiver_normal,
        receiver_shortwave_normal=receiver_normal,
        receiver_confidence=confidence,
        resolved_fraction=resolved_fraction.astype(np.float32),
        coarse_fraction=coarse_fraction.astype(np.float32),
        category_consistency=category_consistency,
        mapping_quality=np.ones(n_receiver, dtype=np.float32),
        normal_quality=np.ones(n_receiver, dtype=np.float32),
        receiver_dbp=receiver_dbp.astype(np.float32),
        emitter_dbp=emitter_dbp.astype(np.float32),
        artifact_sha256_json=np.array(json.dumps(
            artifact_hashes, sort_keys=True)),
        formula=np.array(
            'resolved_fraction*(1-coarse_fraction)*category_consistency*'
            'normal_quality*mapping_quality'))
    print(f'wrote synthetic physics operator to {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='dummy_generated')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_path = Path(args.out).expanduser().resolve()
    if out_path.exists() and not out_path.is_dir():
        raise SystemExit(f'dummy-data output is not a directory: {out_path}')
    if out_path.exists() and any(out_path.iterdir()):
        raise SystemExit(
            f'refusing to overwrite nonempty dummy-data directory: {out_path}; '
            'choose a new --out path')
    out_path.mkdir(parents=True, exist_ok=True)
    args.out = str(out_path)
    write_cases(os.path.join(args.out, 'cases'), rng)
    mesh = write_stl(args.out, rng)
    category_xyz, building_faces = write_category_csvs(args.out, rng, mesh)
    write_physics_operator(args.out, rng, mesh, category_xyz, building_faces)
    print("done - synthetic data only, not physically meaningful")


if __name__ == '__main__':
    main()
