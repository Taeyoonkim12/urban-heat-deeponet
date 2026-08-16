"""Ray-resolved Building SEB comparison: solar control versus M5.

Both models are evaluated with the same area-sampled receivers, ray geometry,
fixed coefficients and QC weights. Receiver, emitter and adjacent-hour
temperatures all come from the evaluated neural network; CFD targets are not
read by the physics operator.
"""

import argparse
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.common import load_run, load_test_cases, DEVICE  # noqa: E402
from urbanheat import config  # noqa: E402
from urbanheat.physics_operator import (RayPhysicsOperator,
                                        RayPhysicsRegularizer,
                                        resolve_operator_manifest)  # noqa: E402


def load_physics_config(run_dir):
    path = os.path.join(run_dir, 'run_config.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f'M5 run configuration is missing: {path}')
    with open(path, encoding='utf-8') as f:
        value = json.load(f).get('physics')
    if not value:
        raise ValueError(f'run has no physics configuration: {path}')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--run-a', required=True, help='solar-control run directory')
    parser.add_argument('--model-a', default='solar_control')
    parser.add_argument('--run-b', required=True, help='M5 run directory')
    parser.add_argument('--model-b', default='m5')
    parser.add_argument('--physics-operator', required=True)
    parser.add_argument('--physics-meta', default=None)
    parser.add_argument('--out', required=True)
    parser.add_argument('--cases-dir', default=None,
                        help='case CSV directory (default: <data-root>/cases)')
    parser.add_argument('--max-files', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model_a, spec_a, scalers_a, _, _ = load_run(args.model_a, args.run_a)
    model_b, spec_b, scalers_b, _, _ = load_run(args.model_b, args.run_b)
    matched_a, matched_b = dict(spec_a), dict(spec_b)
    for key in ('use_seb', 'seb_mode'):
        matched_a.pop(key, None)
        matched_b.pop(key, None)
    if matched_a != matched_b:
        raise ValueError('SEB comparison requires architecture-matched models')
    # Each model is evaluated with its own training scalers below; runs fit
    # sdf/output scalers from random point samples, so tiny differences are
    # expected. Warn (do not fail) beyond a small relative tolerance.
    for name in ('branch', 'sdf', 'output'):
        for attr in ('mean_', 'scale_'):
            va = np.asarray(getattr(scalers_a[name], attr), dtype=np.float64)
            vb = np.asarray(getattr(scalers_b[name], attr), dtype=np.float64)
            if not np.allclose(va, vb, rtol=2e-2, atol=1e-6):
                raise ValueError(
                    f'SEB comparison scaler mismatch: {name}.{attr}')
            if not np.allclose(va, vb, rtol=1e-6, atol=1e-9):
                print(f'[note] small scaler sampling difference in '
                      f'{name}.{attr} (max rel '
                      f'{np.max(np.abs(va-vb)/np.maximum(np.abs(vb),1e-9)):.2e})')
    run_physics = load_physics_config(args.run_b)

    manifest = resolve_operator_manifest(args.physics_operator)
    operator = RayPhysicsOperator(
        manifest, args.data_root, meta_path=args.physics_meta)
    if operator.geometry_signature != run_physics['geometry_signature']:
        raise ValueError('analysis operator does not match the trained M5 run')
    if operator.n_ray != int(run_physics['n_ray']):
        raise ValueError('analysis ray count does not match the trained M5 run')

    common = dict(
        operator=operator, device=DEVICE,
        weighting=run_physics['weighting'],
        n_points=int(run_physics['points']),
        h_roof=float(run_physics['h_roof_w_m2_k']),
        h_wall=float(run_physics['h_wall_w_m2_k']),
        c_areal=float(run_physics['c_areal_j_m2_k']),
        resid_scale=config.RESID_SCALE,
        g0_roof=float(run_physics.get('g0_roof_w_m2', 0.0)),
        g0_wall=float(run_physics.get('g0_wall_w_m2', 0.0)))
    reg_a = RayPhysicsRegularizer(scalers=scalers_a, spec=spec_a, **common)
    reg_b = RayPhysicsRegularizer(scalers=scalers_b, spec=spec_b, **common)

    test_cases, _ = load_test_cases(
        args.data_root, spec_b, max_files=args.max_files,
        cases_dir=args.cases_dir)
    rows = []
    for ci, case in enumerate(tqdm(test_cases, desc='SEB residual')):
        rng_a = np.random.default_rng(args.seed + ci)
        rng_b = np.random.default_rng(args.seed + ci)
        with torch.no_grad():
            _, info_a = reg_a(
                model_a, case['params'], rng_a)
            _, info_b = reg_b(
                model_b, case['params'], rng_b)
        rows.append({
            'case': case['filename'],
            'humidity': case['params']['humidity'],
            'ambient_temperature': case['params']['temperature'],
            'hour': case['params']['hour'],
            'rms_a': info_a['rms_wm2'],
            'rms_b': info_b['rms_wm2'],
            'mae_a': info_a['mae_wm2'],
            'mae_b': info_b['mae_wm2'],
            'p90_a': info_a['p90_abs_wm2'],
            'p90_b': info_b['p90_abs_wm2'],
            'p95_a': info_a['p95_abs_wm2'],
            'p95_b': info_b['p95_abs_wm2'],
            'bias_a': info_a['bias_wm2'],
            'bias_b': info_b['bias_wm2'],
            'groups_a': info_a['groups'],
            'groups_b': info_b['groups'],
            'confidence_mean': info_b['confidence_mean'],
            'resolved_mean': info_b['resolved_mean'],
            'coarse_mean': info_b['coarse_mean'],
        })

    if not rows:
        raise RuntimeError('no test cases were available for SEB analysis')
    rms_a = np.mean([r['rms_a'] for r in rows])
    rms_b = np.mean([r['rms_b'] for r in rows])
    print(f'case-mean Building SEB residual RMS: '
          f'{args.model_a} {rms_a:.2f} W/m2 | '
          f'{args.model_b} {rms_b:.2f} W/m2')
    for metric in ('mae', 'p90', 'p95', 'bias'):
        value_a = np.mean([r[f'{metric}_a'] for r in rows])
        value_b = np.mean([r[f'{metric}_b'] for r in rows])
        print(f'case-mean {metric}: {args.model_a} {value_a:.2f} W/m2 | '
              f'{args.model_b} {value_b:.2f} W/m2')

    for group in ('roof', 'wall', 'sunlit', 'shaded'):
        for model_key, model_name in (('a', args.model_a), ('b', args.model_b)):
            values = [r[f'groups_{model_key}'][group]['rms_wm2']
                      for r in rows
                      if r[f'groups_{model_key}'][group] is not None]
            if values:
                print(f'case-mean {group} RMS {model_name}: '
                      f'{np.mean(values):.2f} W/m2')

    trajectory_delta = {}
    for row in rows:
        key = (row['humidity'], row['ambient_temperature'])
        trajectory_delta.setdefault(key, []).append(row['rms_b'] - row['rms_a'])
    paired = np.array(
        [np.mean(value) for value in trajectory_delta.values()], dtype=np.float64)
    bootstrap_rng = np.random.default_rng(args.seed + 100_000)
    if len(paired):
        draws = bootstrap_rng.choice(
            paired, size=(5000, len(paired)), replace=True).mean(axis=1)
        paired_ci = np.percentile(draws, [2.5, 97.5]).tolist()
        print(f'trajectory-paired delta RMS ({args.model_b}-{args.model_a}): '
              f'{paired.mean():+.2f} W/m2 '
              f'[{paired_ci[0]:+.2f}, {paired_ci[1]:+.2f}]')
    else:
        paired_ci = [None, None]

    period_bias = {}
    for period, predicate in (
            ('AM', lambda hour: hour <= 12),
            ('PM', lambda hour: hour >= 13)):
        period_bias[period] = {}
        for model_key, model_name in (('a', args.model_a), ('b', args.model_b)):
            values = [row[f'bias_{model_key}'] for row in rows
                      if predicate(row['hour'])]
            period_bias[period][model_key] = (
                float(np.mean(values)) if values else None)
            if values:
                print(f'{period} case-mean signed residual {model_name}: '
                      f'{np.mean(values):+.2f} W/m2')

    hours = sorted({r['hour'] for r in rows})
    ha = [np.mean([r['rms_a'] for r in rows if r['hour'] == h]) for h in hours]
    hb = [np.mean([r['rms_b'] for r in rows if r['hour'] == h]) for h in hours]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(hours, ha, '-o', color='#8855aa', label='Solar control')
    ax.plot(hours, hb, '-s', color='#cc4444', label='M5')
    ax.set_xlabel('Hour')
    ax.set_ylabel(r'Building SEB residual RMS (W m$^{-2}$)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'seb_residual.png'), dpi=300)

    payload = {'rows': rows, 'physics': run_physics,
               'case_mean_rms_a': float(rms_a),
               'case_mean_rms_b': float(rms_b),
               'trajectory_paired_delta_rms_b_minus_a': float(paired.mean()),
               'trajectory_bootstrap_ci95': paired_ci,
               'period_bias': period_bias}
    with open(os.path.join(args.out, 'seb_residual.pkl'), 'wb') as f:
        pickle.dump(payload, f)
    with open(os.path.join(args.out, 'seb_residual.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f'saved results to {args.out}')


if __name__ == '__main__':
    main()
