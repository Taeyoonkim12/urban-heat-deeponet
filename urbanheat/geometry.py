"""Geometric preprocessing: d_BP feature, surface categories, normals."""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def load_building_points(building_csv):
    df = pd.read_csv(building_csv)
    xy = df[['X (m)', 'Y (m)']].values.astype(np.float64)
    z = df['Z (m)'].values.astype(np.float64)
    return cKDTree(xy), z


def compute_dbp(query_xyz, building_tree, building_z):
    """Signed building-proximity feature d_BP (paper Sec. 2.3.3).

    Stored under the case/scaler serialization key 'sdf'.

    With d2D the horizontal distance to the nearest building point and
    dz the height difference to that point:
      d2D < 1 m          ->  -|dz| - 0.1   (building-associated, negative)
      z <= building z    ->  d2D           (horizontal distance)
      z >  building z    ->  sqrt(d2D^2 + dz^2)  (3-D distance)
    """
    query_xy = query_xyz[:, :2]
    query_z = query_xyz[:, 2]
    dist_2d, idx = building_tree.query(query_xy, k=1)
    dist_2d = dist_2d.flatten()
    nearest_z = building_z[idx.flatten()]
    d = np.where(query_z <= nearest_z, dist_2d,
                 np.sqrt(dist_2d ** 2 + (query_z - nearest_z) ** 2))
    d = np.where(dist_2d < 1.0, -np.abs(query_z - nearest_z) - 0.1, d)
    return d.astype(np.float32)


class SurfaceCategoryClassifier:
    """Assigns each surface point to one of the five CFD boundary
    categories by nearest-neighbour matching against the per-category
    surface point CSVs exported from the CFD model."""

    def __init__(self, category_csvs):
        pts, labels, self.names = [], [], []
        for cid, (name, path) in enumerate(category_csvs.items()):
            df = pd.read_csv(path)
            p = df.iloc[:, [1, 2, 3]].values.astype(np.float32)
            pts.append(p)
            labels.append(np.full(len(p), cid, dtype=np.int8))
            self.names.append(name)
        self.points = np.vstack(pts)
        self.labels = np.concatenate(labels)
        self.tree = cKDTree(self.points)

    def classify(self, xyz):
        _, idx = self.tree.query(xyz, k=1, workers=-1)
        return self.labels[idx]


def load_mesh(stl_path):
    import trimesh
    mesh = trimesh.load(stl_path, force='mesh')
    try:
        mesh.fix_normals()
    except Exception:
        pass
    face_tree = cKDTree(mesh.triangles_center.astype(np.float64))
    return mesh, face_tree, mesh.face_normals.astype(np.float32)


def compute_normals(xyz, category, face_tree, face_normals):
    """Per-point outward normals. Building points (category 0) inherit
    the normal of the nearest STL face; everything else is treated as
    horizontal ground (0, 0, 1)."""
    normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(xyz), 1))
    mask = (category == 0)
    if face_tree is not None and mask.any():
        _, fi = face_tree.query(xyz[mask].astype(np.float64), k=1, workers=-1)
        normals[mask] = face_normals[fi]
    return normals
