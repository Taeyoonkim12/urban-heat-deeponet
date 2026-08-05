"""Entry point for training and evaluating the DeepONet variants.

Examples:
    python main.py --model m4 --data-root data/dummy --out runs/m4 --epochs 1
    python main.py --model m5 --data-root /path/to/cfd --out runs/m5
    python main.py --model m2 --data-root data/dummy --out runs/m2 --eval-only

Expected data layout under --data-root (see data/README_data.md):
    cases/            Case_{RH}_{T}_{Hr}_XYZInternalTable.csv
    cases_climate/    climate-material scenario cases (M5-mat only)
    Building.csv Road.csv Green.csv Water.csv Topo_IN.csv
    building.stl      (models with surface normals / shadows)
"""

import os
import argparse

import numpy as np
import torch
from scipy.spatial import cKDTree

from urbanheat import config
from urbanheat.data import UrbanHeatDataset, load_cases, split_cases, fit_scalers
from urbanheat.engine import train, evaluate, make_logger
from urbanheat.geometry import (SurfaceCategoryClassifier, load_building_points,
                                load_mesh)
from urbanheat.models import MODEL_SPECS, build_model
from urbanheat.seb import SEBHead, SEBLoss
from urbanheat.solar import compute_shadow_masks, compute_qsw

MODEL_TITLES = {
    'm1': 'M1 - pure DeepONet',
    'm2': 'M2 - +d_BP',
    'm3': 'M3 - +Fourier',
    'm4': 'M4 - +5-category embedding',
    'm5_seb': 'M5-seb - +normals +SEB loss',
    'm5': 'M5 - solar branch +normals +SEB loss',
    'm5_mat': 'M5-mat - +material properties',
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True, choices=sorted(MODEL_SPECS))
    p.add_argument('--data-root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--epochs', type=int, default=config.TOTAL_EPOCHS)
    p.add_argument('--points-per-case', type=int, default=config.POINTS_PER_CASE)
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--max-files', type=int, default=None,
                   help='limit the number of case files (debugging)')
    return p.parse_args()


def prepare_cases(args, spec, log):
    root = args.data_root
    building_tree = building_z = None
    if spec['use_dbp']:
        building_tree, building_z = load_building_points(os.path.join(root, 'Building.csv'))
        log("building points loaded for d_BP")

    classifier = None
    if spec['use_category']:
        category_csvs = {name: os.path.join(root, f'{name}.csv')
                         for name in config.CATEGORY_NAMES}
        classifier = SurfaceCategoryClassifier(category_csvs)
        log(f"category classifier: {len(classifier.points):,} reference points")

    mesh = face_tree = face_normals = None
    if spec['use_normals']:
        mesh, face_tree, face_normals = load_mesh(os.path.join(root, 'building.stl'))
        log(f"STL loaded: {len(face_normals):,} faces")

    kwargs = dict(building_tree=building_tree, building_z=building_z,
                  classifier=classifier, face_tree=face_tree,
                  face_normals=face_normals, with_material=spec['use_material'],
                  max_files=args.max_files, log=log)
    cases = load_cases(os.path.join(root, 'cases'), scenario='normal', **kwargs)
    if spec['use_material']:
        climate_dir = os.path.join(root, 'cases_climate')
        if os.path.isdir(climate_dir):
            cases += load_cases(climate_dir, scenario='climate', **kwargs)
        else:
            log("no cases_climate directory; training on the normal scenario only")

    if spec['use_seb']:
        # Shadow masks are computed once on the largest case and inherited
        # by nearest neighbour for every other case.
        ref = max(cases, key=lambda c: c['total_points'])
        ref_pts = np.column_stack([ref['x_coords'], ref['y_coords'],
                                   ref['z_coords']]).astype(np.float64)
        lit = compute_shadow_masks(mesh, ref_pts, ref['normal'].astype(np.float64),
                                   cache_path=os.path.join(args.out, 'shadow_masks_ref.npy'),
                                   log=log)
        ref_tree = cKDTree(ref_pts)
        for c in cases:
            c['qsw'] = compute_qsw(c, ref_tree, lit)
        log(f"qsw computed, peak {max(float(c['qsw'].max()) for c in cases):.0f} W/m2")
    return cases


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    log = make_logger(args.out)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"model {args.model} | device {device}")

    model, spec = build_model(args.model)
    model = model.to(device)

    cases = prepare_cases(args, spec, log)
    train_cases, test_cases = split_cases(cases, log=log)

    from urbanheat.data import basic_branch
    from urbanheat.solar import solar_branch
    branch_fn = solar_branch if spec['branch'] == 'solar' else basic_branch
    scalers = fit_scalers(train_cases, branch_fn=branch_fn, log=log)

    def dataset_cls(cases_, scalers_, spec_):
        return UrbanHeatDataset(cases_, scalers_, spec_,
                                points_per_sample=args.points_per_case)

    seb_head = seb_loss = None
    if spec['use_seb']:
        seb_head = SEBHead().to(device)
        seb_loss = SEBLoss(seb_head, scalers['output'], device)

    if not args.eval_only:
        train(model, spec, train_cases, test_cases, scalers, args.out,
              dataset_cls, epochs=args.epochs, seb_loss=seb_loss,
              seb_head=seb_head, device=device, log=log)
        if seb_head is not None:
            log(f"learned SEB parameters: h={seb_head.h.item():.2f} W/m2K, "
                f"dT_sky={seb_head.dt.item():.2f} K")

    evaluate(model, spec, test_cases, scalers, args.out, dataset_cls,
             MODEL_TITLES[args.model], device=device, log=log)


if __name__ == '__main__':
    main()
