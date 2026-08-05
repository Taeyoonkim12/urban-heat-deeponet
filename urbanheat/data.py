"""Case loading, train/test split, scalers and the point-cloud dataset."""

import os
import re
import glob
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from .config import (GLOBAL_COORD_RANGES, TEMP_LIMITS, SPLIT_SEED,
                     TEST_FRACTION, SCALER_SAMPLES, POINTS_PER_CASE)
from .geometry import compute_dbp, compute_normals
from .solar import solar_branch, EPS_CAT, RHO_CAT, TAU_CAT


def parse_case_filename(filename):
    """Case_{RH}_{T}_{Hr}_... -> condition dict, or None."""
    m = re.match(r'Case_(\d+)_(\d+)_(\d+)_', os.path.basename(filename))
    if m is None:
        return None
    return {'humidity': float(m.group(1)),
            'temperature': float(m.group(2)),
            'hour': float(m.group(3))}


def basic_branch(params):
    return np.array([params['humidity'], params['temperature'], params['hour']],
                    dtype=np.float32)


def load_cases(csv_dir, building_tree=None, building_z=None, classifier=None,
               face_tree=None, face_normals=None, with_material=False,
               scenario='normal', max_files=None, log=print):
    """Load all CFD export CSVs under csv_dir and attach the per-point
    features required by the requested model."""
    import pandas as pd

    files = sorted(glob.glob(os.path.join(csv_dir, "**", "Case_*_XYZInternalTable*"),
                             recursive=True))
    files = [f for f in files if os.path.isfile(f)]
    if max_files:
        files = files[:max_files]
    log(f"found {len(files)} case files in {csv_dir}")

    cases = []
    for path in files:
        params = parse_case_filename(path)
        if params is None:
            continue
        df = pd.read_csv(path)
        x = df['X (m)'].values.astype(np.float32)
        y = df['Y (m)'].values.astype(np.float32)
        z = df['Z (m)'].values.astype(np.float32)
        if 'Temperature (K)' in df.columns:
            temp = df['Temperature (K)'].values.astype(np.float32) - 273.15
        elif 'Temperature' in df.columns:
            temp = df['Temperature'].values.astype(np.float32)
            if temp.mean() > 200:
                temp -= 273.15
        else:
            continue
        del df

        valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z) | np.isnan(temp))
        x, y, z, temp = x[valid], y[valid], z[valid], temp[valid]

        g = GLOBAL_COORD_RANGES
        crop = ((x >= g['x_min']) & (x <= g['x_max']) &
                (y >= g['y_min']) & (y <= g['y_max']) &
                (z >= g['z_min']) & (z <= g['z_max']))
        x, y, z, temp = x[crop], y[crop], z[crop], temp[crop]

        tmask = (temp >= TEMP_LIMITS['t_min']) & (temp <= TEMP_LIMITS['t_max'])
        x, y, z, temp = x[tmask], y[tmask], z[tmask], temp[tmask]
        if len(x) < 1000:
            continue

        if building_tree is not None:
            dbp = compute_dbp(np.column_stack([x, y, z]), building_tree, building_z)
        else:
            dbp = np.zeros(len(x), dtype=np.float32)

        case = {'filename': os.path.basename(path), 'filepath': path,
                'params': params, 'scenario': scenario,
                'x_coords': x, 'y_coords': y, 'z_coords': z,
                'sdf': dbp,  # d_BP feature, see geometry.compute_dbp
                'temperature': temp, 'total_points': len(x)}

        if classifier is not None:
            xyz = np.column_stack([x, y, z])
            cat = classifier.classify(xyz)
            if scenario == 'climate':
                # Climate-material scenario replaces Road/Topo_IN surfaces
                # with Green, consistent with the CFD setup.
                cat = np.where((cat == 1) | (cat == 4), 2, cat).astype(cat.dtype)
            case['category'] = cat
            if face_tree is not None:
                case['normal'] = compute_normals(xyz, cat, face_tree, face_normals)
            if with_material:
                ci = cat.astype(np.int64)
                case['mat'] = np.column_stack([EPS_CAT[ci], RHO_CAT[ci],
                                               TAU_CAT[ci]]).astype(np.float32)
        cases.append(case)

    log(f"loaded {len(cases)} cases")
    return cases


def split_cases(cases, log=print):
    """Condition-based split: all files sharing the same (RH, T_amb)
    combination go to the same side, so no meteorological condition
    leaks between train and test."""
    groups = defaultdict(list)
    for i, c in enumerate(cases):
        groups[(c['params']['humidity'], c['params']['temperature'])].append(i)

    keys = sorted(groups.keys())
    np.random.seed(SPLIT_SEED)
    np.random.shuffle(keys)

    n_test = max(1, int(len(keys) * TEST_FRACTION))
    test_keys = set(keys[:n_test])
    train_keys = set(keys[n_test:])

    train = [cases[i] for k in train_keys for i in groups[k]]
    test = [cases[i] for k in test_keys for i in groups[k]]
    log(f"split: {len(keys)} conditions -> train {len(train_keys)} / test {len(test_keys)} "
        f"({len(train)} / {len(test)} files)")
    return train, test


def fit_scalers(train_cases, branch_fn=basic_branch, num_samples=SCALER_SAMPLES,
                log=print):
    branch_samples, dbp_samples, output_samples = [], [], []
    per_file = max(1, num_samples // len(train_cases))
    for c in train_cases:
        n = min(per_file, c['total_points'])
        idx = np.random.choice(c['total_points'], size=n, replace=False)
        branch_samples.append(np.tile(branch_fn(c['params']), (n, 1)))
        dbp_samples.append(c['sdf'][idx].reshape(-1, 1))
        output_samples.append(c['temperature'][idx].reshape(-1, 1))

    branch_scaler = StandardScaler().fit(np.concatenate(branch_samples))
    dbp_scaler = StandardScaler().fit(np.concatenate(dbp_samples))
    output_scaler = StandardScaler().fit(np.concatenate(output_samples))
    log(f"branch scaler mean={branch_scaler.mean_}")
    log(f"output scaler mean={output_scaler.mean_[0]:.2f} scale={output_scaler.scale_[0]:.2f}")
    return {'branch': branch_scaler, 'sdf': dbp_scaler, 'output': output_scaler}


def normalize_coords(case, indices):
    g = GLOBAL_COORD_RANGES
    x = (case['x_coords'][indices] - g['x_min']) / (g['x_max'] - g['x_min'] + 1e-8)
    y = (case['y_coords'][indices] - g['y_min']) / (g['y_max'] - g['y_min'] + 1e-8)
    z = (case['z_coords'][indices] - g['z_min']) / (g['z_max'] - g['z_min'] + 1e-8)
    return x, y, z


class UrbanHeatDataset(Dataset):
    """One item = one CFD case, subsampled to points_per_sample points.
    The trunk feature layout is controlled by the model spec."""

    def __init__(self, cases, scalers, spec, points_per_sample=POINTS_PER_CASE):
        self.cases = cases
        self.scalers = scalers
        self.spec = spec
        self.points_per_sample = points_per_sample
        self.branch_fn = solar_branch if spec['branch'] == 'solar' else basic_branch

    def __len__(self):
        return len(self.cases)

    def build_trunk(self, case, indices):
        x, y, z = normalize_coords(case, indices)
        cols = [x, y, z]
        if self.spec['use_dbp']:
            cols.append(self.scalers['sdf'].transform(
                case['sdf'][indices].reshape(-1, 1)).flatten())
        if self.spec['use_normals']:
            nrm = case['normal'][indices].astype(np.float32)
            cols.extend([nrm[:, 0], nrm[:, 1], nrm[:, 2]])
        if self.spec['use_material']:
            mat = case['mat'][indices]
            cols.extend([mat[:, 0], mat[:, 1], mat[:, 2]])
        return np.column_stack(cols).astype(np.float32)

    def __getitem__(self, idx):
        case = self.cases[idx]
        total = case['total_points']
        n = min(self.points_per_sample, total)
        indices = np.random.choice(total, n, replace=False) if total > n else np.arange(total)

        trunk = self.build_trunk(case, indices)
        branch = self.scalers['branch'].transform(
            np.tile(self.branch_fn(case['params']), (n, 1)))
        target = self.scalers['output'].transform(
            case['temperature'][indices].reshape(-1, 1)).astype(np.float32)
        weights = np.ones(n, dtype=np.float32)

        item = [torch.FloatTensor(branch), torch.FloatTensor(trunk)]
        if self.spec['use_category']:
            item.append(torch.LongTensor(case['category'][indices].astype(np.int64)))
        item += [torch.FloatTensor(target), torch.FloatTensor(weights)]
        if self.spec['use_seb']:
            item.append(torch.FloatTensor(case['qsw'][indices].astype(np.float32)))
            item.append(torch.tensor(np.float32(case['params']['temperature'])))
        return tuple(item)
