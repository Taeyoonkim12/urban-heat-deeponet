"""Standalone reference implementation of the signed building-proximity
feature d_BP (paper Sec. 2.3.2).

Given a query point p = (x, y, z) and the set of building surface points B:

  1. find the horizontally nearest building point b (2-D KD-tree on x, y);
  2. if the horizontal distance is below 1 m, p is treated as on/inside
     the building envelope and d_BP = -|z - z_b| - 0.1 (strictly negative);
  3. otherwise d_BP is the positive distance to b: the horizontal distance
     if p lies at or below the building top (z <= z_b), else the full 3-D
     distance sqrt(d_2D^2 + (z - z_b)^2).

The same function is used by every model in the training pipeline (there
it is stored under the working name 'sdf').
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def build_reference(building_csv):
    df = pd.read_csv(building_csv)
    xy = df[['X (m)', 'Y (m)']].values.astype(np.float64)
    z = df['Z (m)'].values.astype(np.float64)
    return cKDTree(xy), z


def dbp_feature(query_xyz, building_tree, building_z):
    query_xy = query_xyz[:, :2]
    query_z = query_xyz[:, 2]

    dist_2d, idx = building_tree.query(query_xy, k=1)
    dist_2d = dist_2d.flatten()
    nearest_z = building_z[idx.flatten()]

    d = np.where(query_z <= nearest_z, dist_2d,
                 np.sqrt(dist_2d ** 2 + (query_z - nearest_z) ** 2))
    d = np.where(dist_2d < 1.0, -np.abs(query_z - nearest_z) - 0.1, d)
    return d.astype(np.float32)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--building-csv', required=True)
    ap.add_argument('--points-csv', required=True,
                    help='CSV with X (m), Y (m), Z (m) columns')
    args = ap.parse_args()

    tree, z = build_reference(args.building_csv)
    pts = pd.read_csv(args.points_csv)[['X (m)', 'Y (m)', 'Z (m)']].values
    d = dbp_feature(pts, tree, z)
    print(f"n={len(d)}  min={d.min():.2f}  max={d.max():.2f}  "
          f"negative fraction={float((d < 0).mean()):.3f}")
