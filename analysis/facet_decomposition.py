"""Facet-wise error decomposition on building surfaces: rooftops vs
walls by orientation (E/N/W/S).

Usage:
    python analysis/facet_decomposition.py --data-root data/dummy \
        --run runs/m4 --model m4 --out analysis_out/facet
"""

import os
import pickle
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import load_run, load_geometry, predict_case
from urbanheat.data import load_cases, split_cases
from urbanheat.geometry import load_mesh

FACET_NAMES = ['Rooftop', 'Wall-E', 'Wall-N', 'Wall-W', 'Wall-S']


def facet_labels(case, face_tree, face_normals):
    """-1 = non-building, 0 = rooftop, 1..4 = wall E/N/W/S."""
    lab = np.full(case['total_points'], -1, dtype=np.int8)
    b = (case['category'] == 0)
    if not b.any():
        return lab
    pts = np.column_stack([case['x_coords'][b], case['y_coords'][b],
                           case['z_coords'][b]]).astype(np.float64)
    _, fi = face_tree.query(pts, k=1, workers=-1)
    n = face_normals[fi]
    roof = (n[:, 2] > 0.7)
    az = np.degrees(np.arctan2(n[:, 0], n[:, 1])) % 360.0
    w = np.full(roof.shape, -1, dtype=np.int8)
    w[(az >= 45) & (az < 135)] = 1    # east
    w[(az < 45) | (az >= 315)] = 2    # north
    w[(az >= 225) & (az < 315)] = 3   # west
    w[(az >= 135) & (az < 225)] = 4   # south
    lab[np.where(b)[0]] = np.where(roof, 0, w).astype(np.int8)
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--model', default='m4')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-files', type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, spec, scalers, _, _ = load_run(args.model, args.run)
    geo = load_geometry(args.data_root, spec)
    cases = load_cases(os.path.join(args.data_root, 'cases'),
                       max_files=args.max_files, **geo)
    _, test_cases = split_cases(cases)

    mesh, face_tree, face_normals = load_mesh(
        os.path.join(args.data_root, 'building.stl'))

    acc = {i: [] for i in range(5)}
    for case in test_cases:
        pred, true = predict_case(model, spec, case, scalers)
        err = np.abs(pred - true)
        lab = facet_labels(case, face_tree, face_normals)
        for i in range(5):
            m = (lab == i)
            if m.any():
                acc[i].append(err[m])

    rows = []
    print("\nfacet MAE (building surfaces):")
    for i in range(5):
        e = np.concatenate(acc[i]) if acc[i] else np.array([np.nan])
        rows.append((FACET_NAMES[i], float(np.nanmean(e)), int(len(e))))
        print(f"  {FACET_NAMES[i]:<9s} MAE {np.nanmean(e):.3f} C ({len(e):,} pts)")
    with open(os.path.join(args.out, 'facet_mae.pkl'), 'wb') as f:
        pickle.dump(rows, f)

    d = {r[0]: r[1] for r in rows}
    order = ['Wall-E', 'Wall-S', 'Wall-W', 'Wall-N']  # solar path order
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    bars = ax.bar(order, [d[k] for k in order], color='#B85450',
                  edgecolor='black', linewidth=0.6, width=0.58)
    for b, k in zip(bars, order):
        ax.text(b.get_x() + b.get_width() / 2, d[k], f"{d[k]:.2f}",
                ha='center', va='bottom', fontsize=10)
    if np.isfinite(d['Rooftop']):
        ax.axhline(d['Rooftop'], color='#555555', linestyle='--', linewidth=1.2)
        ax.text(0.98, d['Rooftop'], f"Rooftop {d['Rooftop']:.2f}", fontsize=9.5,
                color='#555555', ha='right', va='bottom', transform=ax.get_yaxis_transform())
    ax.set_ylabel('MAE (°C)')
    ax.yaxis.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'facet_mae.png'), dpi=300)
    print(f"saved results to {args.out}")


if __name__ == '__main__':
    main()
