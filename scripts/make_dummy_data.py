"""Generate a fully synthetic dummy dataset for pipeline testing.

The real CFD dataset cannot be redistributed, so this script creates
random data with the same file layout and schema. The values are pure
noise - they carry no physical meaning and reflect no statistics of the
real data.

Usage:
    python scripts/make_dummy_data.py --out data/dummy
"""

import os
import argparse

import numpy as np
import pandas as pd

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


def write_category_csvs(root, rng):
    for name, n in CATEGORY_POINTS.items():
        x, y, z = random_points(rng, n)
        df = pd.DataFrame({'Point': np.arange(n),
                           'X (m)': x, 'Y (m)': y, 'Z (m)': z})
        df.to_csv(os.path.join(root, f"{name}.csv"), index=False)
    print(f"wrote category CSVs: {', '.join(CATEGORY_POINTS)}")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/dummy')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    write_cases(os.path.join(args.out, 'cases'), rng)
    write_cases(os.path.join(args.out, 'cases_climate'), rng)
    write_category_csvs(args.out, rng)
    write_stl(args.out, rng)
    print("done - synthetic data only, not physically meaningful")


if __name__ == '__main__':
    main()
