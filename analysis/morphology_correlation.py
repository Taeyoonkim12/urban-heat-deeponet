"""Correlation between urban-morphology indicators and prediction error:
sky view factor (SVF), building height and building density on a 25 m
grid vs cell-averaged MAE.

Usage:
    python analysis/morphology_correlation.py --data-root data/dummy \
        --run runs/m4 --model m4 --out analysis_out/morphology
"""

import os
import pickle
import argparse

import numpy as np
import scipy.stats as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import load_run, load_geometry, predict_case
from urbanheat.config import GLOBAL_COORD_RANGES as G
from urbanheat.data import load_cases, split_cases
from urbanheat.geometry import load_mesh

GRID = 25.0
SVF_CELLS = 4000
SVF_SEED = 0


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
    mesh, _, _ = load_mesh(os.path.join(args.data_root, 'building.stl'))

    nx = int(np.ceil((G['x_max'] - G['x_min']) / GRID))
    ny = int(np.ceil((G['y_max'] - G['y_min']) / GRID))

    # Cell-averaged absolute error over all test cases.
    cell_err = np.zeros(nx * ny)
    cell_cnt = np.zeros(nx * ny)
    for case in test_cases:
        pred, true = predict_case(model, spec, case, scalers)
        err = np.abs(pred - true)
        ix = np.clip(((case['x_coords'] - G['x_min']) / GRID).astype(int), 0, nx - 1)
        iy = np.clip(((case['y_coords'] - G['y_min']) / GRID).astype(int), 0, ny - 1)
        np.add.at(cell_err, iy * nx + ix, err)
        np.add.at(cell_cnt, iy * nx + ix, 1)
    cell_mae = np.where(cell_cnt > 0, cell_err / np.maximum(cell_cnt, 1), np.nan)

    # Morphology from a reference case geometry.
    ref = test_cases[0]
    ix = np.clip(((ref['x_coords'] - G['x_min']) / GRID).astype(int), 0, nx - 1)
    iy = np.clip(((ref['y_coords'] - G['y_min']) / GRID).astype(int), 0, ny - 1)
    cell = iy * nx + ix
    bmask = ref['category'] == 0
    gmask = np.isin(ref['category'], [1, 4])  # Road, Topo_IN = ground

    # Ground elevation per cell (min z of ground points, fallback all points).
    gz = np.full(nx * ny, np.inf)
    np.minimum.at(gz, cell[gmask], ref['z_coords'][gmask])
    gz_all = np.full(nx * ny, np.inf)
    np.minimum.at(gz_all, cell, ref['z_coords'])
    gz = np.where(np.isinf(gz), gz_all, gz)
    gz = np.where(np.isinf(gz), np.nan, gz)

    # Building height above local ground and building point density.
    bz = np.full(nx * ny, -np.inf)
    np.maximum.at(bz, cell[bmask], ref['z_coords'][bmask])
    bh = np.clip(np.where(np.isinf(bz), 0.0, bz - np.nan_to_num(gz, nan=0.0)), 0, None)
    bd = np.zeros(nx * ny)
    np.add.at(bd, cell[bmask], 1)

    # SVF by hemispheric ray casting from 2 m above local ground.
    cx = G['x_min'] + (np.arange(nx) + 0.5) * GRID
    cy = G['y_min'] + (np.arange(ny) + 0.5) * GRID
    CXX, CYY = np.meshgrid(cx, cy)
    valid = (~np.isnan(cell_mae)) & (~np.isnan(gz))
    vi = np.where(valid)[0]
    if len(vi) > SVF_CELLS:
        vi = np.random.default_rng(SVF_SEED).choice(vi, SVF_CELLS, replace=False)

    n_az, n_el = 16, 4
    azs = np.linspace(0, 2 * np.pi, n_az, endpoint=False)
    els = np.linspace(0.15, 1.35, n_el)
    dirs = np.array([[np.sin(a) * np.cos(e), np.cos(a) * np.cos(e), np.sin(e)]
                     for e in els for a in azs])
    w = np.cos(np.repeat(els, n_az))
    w = w / w.sum()

    svf = np.full(nx * ny, np.nan)
    svf[vi] = 0.0
    origins = np.column_stack([CXX.flatten()[vi], CYY.flatten()[vi], gz[vi] + 2.0])
    print(f"SVF rays: {len(vi):,} cells x {len(dirs)} directions")
    for k, d in enumerate(dirs):
        hit = mesh.ray.intersects_any(ray_origins=origins,
                                      ray_directions=np.tile(d, (len(vi), 1)))
        svf[vi] += w[k] * (~hit)

    def corr(name, vals):
        m = (~np.isnan(cell_mae)) & (~np.isnan(vals))
        r, p = st.pearsonr(vals[m], cell_mae[m])
        print(f"  {name:<18s} r={r:+.3f} (p={p:.1e}, n={m.sum():,})")
        return name, float(r), float(p), int(m.sum())

    print("\nmorphology vs cell MAE:")
    res = [corr('SVF', svf), corr('Building height', bh),
           corr('Building density', bd)]
    with open(os.path.join(args.out, 'morphology_corr.pkl'), 'wb') as f:
        pickle.dump({'rows': res, 'svf': svf, 'cell_mae': cell_mae,
                     'bh': bh, 'bd': bd, 'gz': gz}, f)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0))
    panels = [('SVF', svf), ('Building height (m)', bh),
              ('Building density (points per cell)', bd)]
    for j, (ax, (xl, vals)) in enumerate(zip(axes, panels)):
        m = (~np.isnan(cell_mae)) & (~np.isnan(vals))
        x, y = vals[m], cell_mae[m]
        ax.scatter(x, y, s=5, alpha=0.35, color='#4C72A0', edgecolors='none',
                   rasterized=True)
        r, _ = st.pearsonr(x, y)
        ax.set_title(f"({chr(97 + j)})", loc='left', fontsize=12)
        ax.text(0.96, 0.94, f"r = {r:+.3f}", transform=ax.transAxes,
                ha='right', va='top', fontsize=10.5,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#888888', lw=0.6))
        ax.set_xlabel(xl)
    axes[0].set_ylabel('Cell MAE (°C)')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'morphology_error_corr.png'), dpi=300)
    print(f"saved results to {args.out}")


if __name__ == '__main__':
    main()
