"""Entry point for training and evaluating the DeepONet variants.

Examples:
    python main.py --model m4 --data-root data/dummy --out runs/m4 --epochs 1
    python main.py --model m5 --data-root /path/to/cfd --out runs/m5 \
        --physics-operator /path/to/rays/ray_manifest_TAG.json
    python main.py --model m2 --data-root data/dummy --out runs/m2 --eval-only

Expected data layout under --data-root (see data/README_data.md):
    cases/            Case_{RH}_{T}_{Hr}_XYZInternalTable.csv
    Building.csv Road.csv Green.csv Water.csv Topo_IN.csv
    building.stl      building geometry used by physics preprocessing
"""

import os
import argparse
import json
import pickle

import numpy as np
import torch

from urbanheat import config
from urbanheat.data import (UrbanHeatDataset, fit_scalers, load_cases,
                            split_cases, validate_physics_time_coverage)
from urbanheat.engine import train, evaluate, make_logger
from urbanheat.geometry import SurfaceCategoryClassifier, load_building_points
from urbanheat.models import MODEL_SPECS, build_model
from urbanheat.physics_operator import (RayPhysicsOperator,
                                        RayPhysicsRegularizer,
                                        resolve_operator_manifest)

MODEL_TITLES = {
    'm0': 'M0 - canonical DeepONet (inner-product coupling)',
    'm1': 'M1 - adaptive dual-path branch-trunk combiner',
    'm2': 'M2 - + signed building-proximity feature d_BP',
    'm3': 'M3 - + multiscale Fourier features',
    'm4': 'M4 - + surface-category embedding (primary model)',
    'm5': 'M5 - solar re-encoding + transient SEB regularization',
    'd1': 'D1 - supplementary control (inner coupling + d_BP)',
    'solar_control': 'Solar control - M4 with solar branch, no SEB',
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True, choices=sorted(MODEL_SPECS))
    p.add_argument('--data-root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--epochs', type=int, default=config.TOTAL_EPOCHS)
    p.add_argument('--schedule-epochs', type=int, default=None,
                   help='cosine-schedule horizon (default: --epochs); keep '
                        'fixed across an interrupted/resumed run')
    p.add_argument('--patience', type=int, default=config.PATIENCE,
                   help='early-stopping patience (supplementary short '
                        'protocol: --epochs 200 --patience 40)')
    p.add_argument('--points-per-case', type=int, default=config.POINTS_PER_CASE)
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--max-files', type=int, default=None,
                   help='limit the number of case files (debugging)')
    p.add_argument('--seed', type=int, default=42,
                   help='global torch/numpy seed for weight init and sampling')
    # Optional path overrides for data layouts that differ from
    # <data-root>/cases etc. (see data/README_data.md).
    p.add_argument('--cases-dir', default=None,
                   help='case CSV directory (default: <data-root>/cases)')
    p.add_argument('--physics-operator', default=None,
                   help='ray_manifest_*.json or directory containing one '
                        '(required when training M5)')
    p.add_argument('--physics-meta', default=None,
                   help='optional explicit physics_operator_meta_*.npz')
    p.add_argument('--physics-weighting', choices=('uniform', 'confidence'),
                   default=config.PHYSICS_WEIGHTING)
    p.add_argument('--physics-points', type=int, default=config.PHYSICS_POINTS)
    p.add_argument('--beta-phys', type=float, default=config.BETA_PHYS)
    p.add_argument('--h-roof', type=float, default=config.H_CONV_ROOF)
    p.add_argument('--h-wall', type=float, default=config.H_CONV_WALL)
    p.add_argument('--c-areal', type=float, default=config.C_AREAL)
    p.add_argument('--residual-floor', type=float, default=config.RESIDUAL_FLOOR,
                   help='SEB no-penalty threshold R0 [W/m2]; penalize only '
                        '|R| above this value (0 = plain squared residual)')
    p.add_argument('--g0-roof', type=float, default=config.G0_ROOF,
                   help='prescribed constant baseline flux for roofs [W/m2]')
    p.add_argument('--g0-wall', type=float, default=config.G0_WALL,
                   help='prescribed constant baseline flux for walls [W/m2]')
    p.add_argument('--allow-overwrite', action='store_true',
                   help='allow training into a nonempty --out directory '
                        '(DANGEROUS: overwrites run artifacts)')
    p.add_argument('--resume', action='store_true',
                   help='resume an interrupted training run from '
                        '<out>/last_model.pt (model, optimizer, scheduler, '
                        'AMP and RNG state)')
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

    kwargs = dict(building_tree=building_tree, building_z=building_z,
                  classifier=classifier, face_tree=None,
                  face_normals=None, max_files=args.max_files, log=log)
    cases_dir = args.cases_dir or os.path.join(root, 'cases')
    return load_cases(cases_dir, **kwargs)


def main():
    args = parse_args()

    schedule_epochs = (args.epochs if args.schedule_epochs is None
                       else args.schedule_epochs)
    if args.epochs <= 0 or schedule_epochs < args.epochs:
        raise SystemExit('--epochs must be positive and --schedule-epochs '
                         'must be at least --epochs')

    if args.eval_only and args.resume:
        raise SystemExit('--eval-only and --resume are mutually exclusive')
    positive = {
        '--patience': args.patience,
        '--points-per-case': args.points_per_case,
        '--physics-points': args.physics_points,
        '--beta-phys': args.beta_phys,
        '--h-roof': args.h_roof,
        '--h-wall': args.h_wall,
        '--c-areal': args.c_areal,
    }
    bad = [name for name, value in positive.items()
           if not np.isfinite(value) or value <= 0]
    if bad:
        raise SystemExit(f"values must be positive: {', '.join(bad)}")
    if args.max_files is not None and args.max_files <= 0:
        raise SystemExit('--max-files must be positive when provided')
    if not np.isfinite(args.residual_floor) or args.residual_floor < 0:
        raise SystemExit('--residual-floor must be finite and nonnegative')
    for name, value in (('--g0-roof', args.g0_roof), ('--g0-wall', args.g0_wall)):
        if not np.isfinite(value) or abs(value) > 1000.0:
            raise SystemExit(f'{name} must be finite and within +/-1000 W/m2')

    # Checkpoint-overwrite guard: training must always go to a fresh
    # directory. An existing best_model.pt would be silently destroyed
    # by a new run, so refuse unless the user explicitly opts in.
    resume_ckpt = os.path.join(args.out, 'last_model.pt')
    if args.resume and not os.path.exists(resume_ckpt):
        raise SystemExit(f"--resume given but {resume_ckpt} does not exist.")
    output_nonempty = (os.path.isdir(args.out) and bool(os.listdir(args.out)))
    if (not args.eval_only) and (not args.resume) and output_nonempty \
            and not args.allow_overwrite:
        raise SystemExit(
            f"REFUSING to train into nonempty directory: {args.out}.\n"
            f"Use a NEW --out directory (recommended), pass --resume to "
            f"continue an interrupted run, pass --allow-overwrite if you "
            f"really mean to destroy it, or --eval-only to evaluate the "
            f"existing checkpoint.")

    os.makedirs(args.out, exist_ok=True)
    log = make_logger(args.out)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"model {args.model} | device {device} | seed {args.seed}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, spec = build_model(args.model)
    model = model.to(device)

    cases = prepare_cases(args, spec, log)
    train_cases, test_cases = split_cases(cases, log=log)

    from urbanheat.data import basic_branch
    from urbanheat.solar import solar_branch
    branch_fn = solar_branch if spec['branch'] == 'solar' else basic_branch
    scaler_path = os.path.join(args.out, 'scalers.pkl')
    if args.eval_only or args.resume:
        if not os.path.exists(scaler_path):
            raise SystemExit(f'saved scalers are missing: {scaler_path}')
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
        log(f'loaded saved scalers from {scaler_path}')
    else:
        scalers = fit_scalers(train_cases, branch_fn=branch_fn, log=log)

    if args.eval_only:
        config_path = os.path.join(args.out, 'run_config.json')
        if os.path.exists(config_path):
            with open(config_path, encoding='utf-8') as f:
                saved_model = json.load(f).get('model')
            if saved_model != args.model:
                raise SystemExit(
                    f'checkpoint run_config model is {saved_model}, not {args.model}')

    def dataset_cls(cases_, scalers_, spec_):
        return UrbanHeatDataset(cases_, scalers_, spec_,
                                points_per_sample=args.points_per_case)

    physics_regularizer = None
    physics_config = None
    if spec['use_seb'] and not args.eval_only:
        if args.physics_operator is None:
            raise SystemExit(
                '--physics-operator is required when training M5')
        manifest = resolve_operator_manifest(args.physics_operator)
        operator = RayPhysicsOperator(
            manifest, args.data_root, meta_path=args.physics_meta)
        if operator.manifest.get('synthetic_schema_test_only') is True:
            log('synthetic operator: complete 07-18 trajectory gate skipped')
        else:
            validate_physics_time_coverage(cases)
            log('verified complete 07-18 trajectories for physics training')
        physics_regularizer = RayPhysicsRegularizer(
            operator, scalers, spec, device,
            weighting=args.physics_weighting,
            n_points=args.physics_points,
            h_roof=args.h_roof, h_wall=args.h_wall,
            c_areal=args.c_areal, resid_scale=config.RESID_SCALE,
            residual_floor=args.residual_floor,
            g0_roof=args.g0_roof, g0_wall=args.g0_wall)
        physics_config = {
            'manifest': str(manifest),
            'geometry_signature': operator.geometry_signature,
            'n_ray': operator.n_ray,
            'weighting': args.physics_weighting,
            'points': args.physics_points,
            'beta': args.beta_phys,
            'h_roof_w_m2_k': args.h_roof,
            'h_wall_w_m2_k': args.h_wall,
            'c_areal_j_m2_k': args.c_areal,
            'residual_floor_w_m2': args.residual_floor,
            'g0_roof_w_m2': args.g0_roof,
            'g0_wall_w_m2': args.g0_wall,
            'sky_boundary': 'transparent_zero_incoming_longwave',
            'residual_scale_w_m2': config.RESID_SCALE,
            'physics_rng_seed': config.PHYSICS_SEED,
            'artifact_sha256': operator.artifact_sha256,
            'metadata_path': str(operator.meta_path),
            'metadata_sha256': operator.meta_sha256,
        }
        log(f"physics operator: {operator.describe()}")
        log(f"fixed SEB: h_roof={args.h_roof:g}, h_wall={args.h_wall:g} "
            f"W/m2/K | C_A={args.c_areal:g} J/m2/K | "
            f"weighting={args.physics_weighting} | beta={args.beta_phys:g} | "
            f"residual_floor={args.residual_floor:g} W/m2 | "
            f"G0 roof/wall={args.g0_roof:g}/{args.g0_wall:g} W/m2")

    if not args.eval_only:
        current_config = {'model': args.model, 'seed': args.seed,
                          'test_set': 'condition_grouped_test',
                          'test_use': 'early_stopping_and_evaluation',
                          'ood_use': 'evaluation_only_after_freeze',
                          'data': {
                              'data_root': os.path.abspath(args.data_root),
                              'cases_dir': os.path.abspath(
                                  args.cases_dir or os.path.join(
                                      args.data_root, 'cases')),
                              'max_files': args.max_files,
                              'points_per_case': args.points_per_case,
                          },
                          'training': {
                              'patience': args.patience,
                              'schedule_epochs': schedule_epochs,
                              'lr_init': config.LR_INIT,
                              'lr_min': config.LR_MIN,
                              'warmup_epochs': config.WARMUP_EPOCHS,
                              'weight_decay': config.WEIGHT_DECAY,
                          },
                          'physics': physics_config}
        config_path = os.path.join(args.out, 'run_config.json')
        if args.resume:
            if not os.path.exists(config_path):
                raise SystemExit(f'cannot resume without {config_path}')
            with open(config_path, encoding='utf-8') as f:
                saved_config = json.load(f)
            if saved_config != current_config:
                raise SystemExit(
                    'resume configuration differs from the saved run_config.json')
        else:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(current_config, f, indent=2)

    if not args.eval_only:
        train(model, spec, train_cases, test_cases, scalers, args.out,
              dataset_cls, epochs=args.epochs, patience=args.patience,
              schedule_epochs=schedule_epochs,
              physics_regularizer=physics_regularizer,
              physics_beta=args.beta_phys, device=device, log=log,
              resume_from=resume_ckpt if args.resume else None)

    evaluate(model, spec, test_cases, scalers, args.out, dataset_cls,
             MODEL_TITLES[args.model], device=device, log=log)


if __name__ == '__main__':
    main()
