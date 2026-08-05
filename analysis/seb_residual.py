"""Surface-energy-balance residual comparison between two trained models
(e.g. the data-only M4 vs the SEB-constrained M5). Both prediction fields
are evaluated with the same learned SEB parameters, so the comparison is
about physical consistency of the predictions, not about the loss.

Usage:
    python analysis/seb_residual.py --data-root data/dummy \
        --run-a runs/m4 --model-a m4 --run-b runs/m5 --model-b m5 \
        --out analysis_out/seb_residual
"""

import os
import pickle
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from common import load_run, load_geometry, predict_case
from urbanheat.config import CATEGORY_NAMES, N_CATEGORIES, W_PHYS_CAT
from urbanheat.data import load_cases, split_cases
from urbanheat.geometry import load_mesh
from urbanheat.seb import SEBHead, SIGMA_SB
from urbanheat.solar import EPS_CAT, compute_shadow_masks, compute_qsw


def seb_residual(temp_c, case, qsw, h, dt_sky, g_cat):
    cat = case['category'].astype(np.int64)
    ta = float(case['params']['temperature'])
    tsky = ta + 273.15 - dt_sky
    ldn = SIGMA_SB * tsky ** 4
    tk = temp_c + 273.15
    return qsw + EPS_CAT[cat] * (ldn - SIGMA_SB * tk ** 4) - h * (temp_c - ta) - g_cat[cat]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--run-a', required=True)
    ap.add_argument('--model-a', default='m4')
    ap.add_argument('--run-b', required=True)
    ap.add_argument('--model-b', default='m5')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-files', type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model_a, spec_a, scalers_a, _, _ = load_run(args.model_a, args.run_a)
    model_b, spec_b, scalers_b, ckpt_b, _ = load_run(args.model_b, args.run_b)

    # Learned SEB parameters come from the physics-constrained run.
    seb_head = SEBHead()
    seb_head.load_state_dict(ckpt_b['seb_head_state_dict'])
    h = float(seb_head.h.item())
    dt_sky = float(seb_head.dt.item())
    g_cat = seb_head.g_cat.detach().numpy()
    print(f"learned SEB parameters: h={h:.2f} W/m2K, dT_sky={dt_sky:.2f} K, "
          f"G_cat={np.round(g_cat, 1)}")

    # Load test cases with the union of both models' features.
    geo_spec = dict(spec_b)
    for k in ('use_dbp', 'use_category', 'use_normals', 'use_material'):
        geo_spec[k] = spec_a[k] or spec_b[k]
    geo = load_geometry(args.data_root, geo_spec)
    cases = load_cases(os.path.join(args.data_root, 'cases'),
                       max_files=args.max_files, **geo)
    _, test_cases = split_cases(cases)

    mesh, _, _ = load_mesh(os.path.join(args.data_root, 'building.stl'))
    ref = max(test_cases, key=lambda c: c['total_points'])
    ref_pts = np.column_stack([ref['x_coords'], ref['y_coords'],
                               ref['z_coords']]).astype(np.float64)
    lit = compute_shadow_masks(mesh, ref_pts, ref['normal'].astype(np.float64),
                               cache_path=os.path.join(args.out, 'shadow_masks.npy'))
    ref_tree = cKDTree(ref_pts)

    tags = {args.model_a: (model_a, spec_a, scalers_a),
            args.model_b: (model_b, spec_b, scalers_b)}
    acc = {t: {c: [] for c in range(N_CATEGORIES)} for t in tags}
    for case in test_cases:
        qsw = compute_qsw(case, ref_tree, lit)
        cat = case['category'].astype(np.int64)
        for tag, (model, spec, scalers) in tags.items():
            pred, _ = predict_case(model, spec, case, scalers)
            resid = seb_residual(pred, case, qsw, h, dt_sky, g_cat)
            for c in range(N_CATEGORIES):
                m = cat == c
                if m.any():
                    acc[tag][c].append(np.abs(resid[m]))

    rows = []
    print("\nmean |SEB residual| (W/m2):")
    print(f"{'category':<10s} {args.model_a:>10s} {args.model_b:>10s}")
    for c in range(N_CATEGORIES):
        vals = {}
        for tag in tags:
            vals[tag] = float(np.mean(np.concatenate(acc[tag][c]))) if acc[tag][c] else np.nan
        note = '' if W_PHYS_CAT[c] > 0 else ' (excluded from loss)'
        print(f"{CATEGORY_NAMES[c]:<10s} {vals[args.model_a]:10.1f} "
              f"{vals[args.model_b]:10.1f}{note}")
        rows.append({'cat': CATEGORY_NAMES[c], **vals, 'w_phys': W_PHYS_CAT[c]})

    with open(os.path.join(args.out, 'seb_residual_compare.pkl'), 'wb') as f:
        pickle.dump({'rows': rows, 'h': h, 'dT_sky': dt_sky,
                     'G': g_cat.tolist()}, f)

    sel = [c for c in range(N_CATEGORIES) if W_PHYS_CAT[c] > 0]
    xw = np.arange(len(sel))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.bar(xw - w / 2, [rows[c][args.model_a] for c in sel], w,
           label=args.model_a.upper(), color='#888888')
    ax.bar(xw + w / 2, [rows[c][args.model_b] for c in sel], w,
           label=args.model_b.upper(), color='#cc4444')
    ax.set_xticks(xw)
    ax.set_xticklabels([CATEGORY_NAMES[c] for c in sel])
    ax.set_ylabel('Mean |SEB residual| (W m$^{-2}$)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'seb_residual_compare.png'), dpi=300)
    print(f"saved results to {args.out}")


if __name__ == '__main__':
    main()
