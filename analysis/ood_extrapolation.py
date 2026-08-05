"""Out-of-distribution extrapolation evaluation (paper Sec. 3.6.2).

Evaluates the ablation ladder on the in-distribution test set and on
OOD cases with ambient temperatures beyond the training envelope.
Before any OOD number is reported, each model must reproduce its own
in-distribution MAE (gate check, tolerance 0.09 C) - this guards
against scaler/checkpoint mismatches.

Protocol: 100,000 points per case, seed 0, and all models share the
same point indices per case (paired comparison).

Usage:
    python analysis/ood_extrapolation.py --data-root data/dummy \
        --ood-dir data/dummy/cases --runs runs --models m1 m2 m3 m4 m5 \
        --out analysis_out/ood
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import load_run, load_geometry, predict_case
from urbanheat.data import load_cases, split_cases
from urbanheat.models import MODEL_SPECS

PTS_PER_CASE = 100_000
SEED = 0
GATE_TOL = 0.09

COLORS = {'m1': '#77aa55', 'm2': '#4477bb', 'm3': '#cc8833',
          'm4': '#333333', 'm5': '#cc4444'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--ood-dir', required=True)
    ap.add_argument('--runs', required=True, help='directory containing per-model run folders')
    ap.add_argument('--models', nargs='+', default=['m1', 'm2', 'm3', 'm4', 'm5'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-files', type=int, default=None)
    ap.add_argument('--gate-tol', type=float, default=GATE_TOL)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs = {}
    union_spec = {k: False for k in
                  ('use_dbp', 'use_category', 'use_normals', 'use_material')}
    for name in args.models:
        model, spec, scalers, ckpt, ref = load_run(name, os.path.join(args.runs, name))
        runs[name] = (model, spec, scalers, ref)
        for k in union_spec:
            union_spec[k] = union_spec[k] or spec[k]
        print(f"{name}: checkpoint epoch {ckpt.get('epoch', '?')}, in-dist ref MAE {ref}")

    # Load data once with the union of all required features.
    geo_spec = dict(MODEL_SPECS[args.models[-1]])
    geo_spec.update(union_spec)
    geo = load_geometry(args.data_root, geo_spec)
    cases = load_cases(os.path.join(args.data_root, 'cases'),
                       max_files=args.max_files, **geo)
    _, test_cases = split_cases(cases)
    ood_cases = load_cases(args.ood_dir, max_files=args.max_files, **geo)

    rows = []
    rng = np.random.default_rng(SEED)
    for setname, case_list in (('in', test_cases), ('ood', ood_cases)):
        for case in case_list:
            n = min(PTS_PER_CASE, case['total_points'])
            idx = rng.choice(case['total_points'], n, replace=False)  # shared across models
            for name in args.models:
                model, spec, scalers, _ = runs[name]
                pred, true = predict_case(model, spec, case, scalers, indices=idx)
                err = pred - true
                rows.append({'model': name, 'set': setname,
                             'T': float(case['params']['temperature']),
                             'hour': float(case['params']['hour']),
                             'mae': float(np.abs(err).mean()),
                             'rmse': float(np.sqrt((err ** 2).mean())),
                             'bias': float(err.mean()),
                             'p95': float(np.percentile(np.abs(err), 95)),
                             'n_pts': int(n)})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, 'ood_results.csv'), index=False)

    # Gate: each model must reproduce its recorded in-distribution MAE.
    print("\ngate check (in-dist MAE vs training record):")
    fails = []
    for name in args.models:
        got = df[(df.model == name) & (df.set == 'in')]['mae'].mean()
        ref = runs[name][3]
        if ref is None:
            print(f"  {name}: {got:.3f} (no reference recorded - verify manually)")
            continue
        ok = abs(got - ref) < args.gate_tol
        print(f"  {name}: {got:.3f} vs ref {ref:.3f} ({got - ref:+.3f}) "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            fails.append(name)
    assert not fails, f"gate failed for {fails}; OOD numbers are not interpretable"

    by_t = df.groupby(['model', 'T'])['mae'].mean()
    tab = by_t.unstack(level='model')[args.models]
    tab.round(4).to_csv(os.path.join(args.out, 'ood_mae_by_T.csv'))
    print("\nMAE by ambient temperature:")
    print(tab.round(3).to_string())

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for name in args.models:
        s = by_t.loc[name].sort_index()
        ax.plot(s.index, s.values, '-o', color=COLORS.get(name), markersize=4,
                linewidth=1.6, label=name.upper())
    ax.set_xlabel('Ambient temperature (°C)')
    ax.set_ylabel('MAE (°C)')
    ax.legend(frameon=True)
    ax.yaxis.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'ood_error_curve.png'), dpi=300)
    print(f"\nsaved results to {args.out}")


if __name__ == '__main__':
    main()
