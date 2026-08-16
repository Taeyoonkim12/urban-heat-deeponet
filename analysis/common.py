"""Shared helpers for the analysis scripts: loading trained runs and
predicting whole cases."""

import os
import sys
import pickle

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from urbanheat import config
from urbanheat.data import (load_cases, split_cases, basic_branch,
                            normalize_coords)
from urbanheat.geometry import (SurfaceCategoryClassifier, load_building_points,
                                load_mesh)
from urbanheat.models import build_model
from urbanheat.solar import solar_branch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_run(name, run_dir, device=DEVICE):
    """Load a trained model + scalers produced by main.py."""
    model, spec = build_model(name)
    ckpt = torch.load(os.path.join(run_dir, 'best_model.pt'),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    with open(os.path.join(run_dir, 'scalers.pkl'), 'rb') as f:
        scalers = pickle.load(f)
    ref = None
    ref_path = os.path.join(run_dir, 'evaluation_results.pkl')
    if os.path.exists(ref_path):
        with open(ref_path, 'rb') as f:
            ref = pickle.load(f).get('mae')
    return model, spec, scalers, ckpt, ref


def load_geometry(data_root, spec):
    building_tree = building_z = classifier = face_tree = face_normals = None
    if spec['use_dbp']:
        building_tree, building_z = load_building_points(
            os.path.join(data_root, 'Building.csv'))
    if spec['use_category']:
        classifier = SurfaceCategoryClassifier(
            {n: os.path.join(data_root, f'{n}.csv') for n in config.CATEGORY_NAMES})
    if spec.get('use_seb'):
        _, face_tree, face_normals = load_mesh(os.path.join(data_root, 'building.stl'))
    return dict(building_tree=building_tree, building_z=building_z,
                classifier=classifier, face_tree=face_tree,
                face_normals=face_normals)


def load_test_cases(data_root, spec, max_files=None, log=print,
                    cases_dir=None):
    geo = load_geometry(data_root, spec)
    cases = load_cases(cases_dir or os.path.join(data_root, 'cases'),
                       max_files=max_files, log=log, **geo)
    _, test_cases = split_cases(cases, log=log)
    return test_cases, geo


def build_case_inputs(case, indices, spec, scalers):
    x, y, z = normalize_coords(case, indices)
    cols = [x, y, z]
    if spec['use_dbp']:
        cols.append(scalers['sdf'].transform(
            case['sdf'][indices].reshape(-1, 1)).flatten())
    trunk = np.column_stack(cols).astype(np.float32)

    branch_fn = solar_branch if spec['branch'] == 'solar' else basic_branch
    branch = scalers['branch'].transform(
        branch_fn(case['params']).reshape(1, -1)).astype(np.float32)
    branch = np.tile(branch, (len(indices), 1))
    cat = case['category'][indices].astype(np.int64) if spec['use_category'] else None
    return branch, trunk, cat


@torch.no_grad()
def predict_case(model, spec, case, scalers, indices=None, chunk=500_000,
                 device=DEVICE):
    """Predict surface temperature (deg C) for one case."""
    if indices is None:
        indices = np.arange(case['total_points'])
    branch, trunk, cat = build_case_inputs(case, indices, spec, scalers)
    outs = []
    for i in range(0, len(indices), chunk):
        bt = torch.from_numpy(branch[i:i + chunk]).to(device)
        tt = torch.from_numpy(trunk[i:i + chunk]).to(device)
        ct = None if cat is None else torch.from_numpy(cat[i:i + chunk]).to(device)
        outs.append(model(bt, tt, ct).float().cpu().numpy())
    pred = scalers['output'].inverse_transform(np.concatenate(outs)).flatten()
    return pred, case['temperature'][indices]
