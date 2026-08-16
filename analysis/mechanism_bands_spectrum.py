"""Residual mechanism analysis (paper Sec. 3.2 / Supplementary S2).

Compares two models (default M1 vs M4, which share the dual-path
coupling so the difference isolates the spatial representation) along
two views:

  (a) case-averaged MAE within building-proximity bands of d_BP,
      with "building-associated" points (d_BP < 0) as a separate band;
  (b) case-wise radially averaged spectra of the Hann-windowed signed
      residual on a 25-m plan-view grid.

Usage:
    python analysis/mechanism_bands_spectrum.py --data-root data/dummy \
        --run-a runs/m1 --model-a m1 --run-b runs/m4 --model-b m4 \
        --out analysis_out/mechanism
"""

import os
import sys
import csv
import pickle
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import load_run, load_test_cases, predict_case  # noqa: E402
from urbanheat.config import GLOBAL_COORD_RANGES as G                # noqa: E402

SEED = 0
MAX_POINTS = 100_000
GRID_M = 25.0
BAND_EDGES = [0.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
BAND_LABELS = ['Building\nassociated', '0-5', '5-10', '10-20',
               '20-50', '50-100', '100-200', '>200']


def band_masks(dbp):
    masks = [dbp < 0.0]
    edges = BAND_EDGES + [np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        masks.append((dbp >= lo) & (dbp < hi))
    return masks


def grid_field(values, xs, ys, nx, ny):
    ix = np.clip(((xs - G['x_min']) / GRID_M).astype(np.int64), 0, nx - 1)
    iy = np.clip(((ys - G['y_min']) / GRID_M).astype(np.int64), 0, ny - 1)
    ssum = np.zeros((ny, nx)); cnt = np.zeros((ny, nx))
    np.add.at(ssum, (iy, ix), values); np.add.at(cnt, (iy, ix), 1)
    valid = cnt > 0
    field = np.empty((ny, nx))
    field[valid] = ssum[valid] / cnt[valid]
    field[~valid] = field[valid].mean()
    return field


def radial_spectrum(field, hann, wenergy, bin_idx, nbins):
    p = np.abs(np.fft.fftshift(np.fft.fft2((field - field.mean()) * hann))) ** 2
    p = p.ravel() / wenergy
    s = np.bincount(bin_idx, weights=p, minlength=nbins)
    n = np.bincount(bin_idx, minlength=nbins)
    out = np.full(nbins, np.nan)
    out[n > 0] = s[n > 0] / n[n > 0]
    return out


def mean_ci(a):
    a = np.asarray(a, dtype=np.float64)
    n = np.sum(np.isfinite(a), axis=0)
    m = np.nanmean(a, axis=0)
    with np.errstate(invalid='ignore'):
        ci = 1.96 * np.nanstd(a, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    return m, np.where(n >= 2, ci, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--run-a', required=True)
    ap.add_argument('--model-a', default='m1')
    ap.add_argument('--run-b', required=True)
    ap.add_argument('--model-b', default='m4')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-files', type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model_a, spec_a, sc_a, _, ref_a = load_run(args.model_a, args.run_a)
    model_b, spec_b, sc_b, _, ref_b = load_run(args.model_b, args.run_b)
    cases, _ = load_test_cases(args.data_root, spec_b, max_files=args.max_files)

    nx = int(np.ceil((G['x_max'] - G['x_min']) / GRID_M))
    ny = int(np.ceil((G['y_max'] - G['y_min']) / GRID_M))
    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=GRID_M))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=GRID_M))
    FY, FX = np.meshgrid(fy, fx, indexing='ij')
    fr = np.sqrt(FX ** 2 + FY ** 2)
    df = min(1.0 / (nx * GRID_M), 1.0 / (ny * GRID_M))
    edges = np.arange(0.0, 1.0 / (2 * GRID_M) + df * 1.01, df)
    edges[-1] += 1e-12
    centres = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.digitize(fr.ravel(), edges) - 1, 0, len(centres) - 1)
    hann = np.outer(np.hanning(ny), np.hanning(nx))
    wenergy = np.sum(hann ** 2)

    rng = np.random.default_rng(SEED)
    nb = len(BAND_LABELS)
    mae_a = np.full((len(cases), nb), np.nan)
    mae_b = np.full((len(cases), nb), np.nan)
    spec_ac, spec_bc = [], []
    abs_a = abs_b = npts = 0.0

    for k, case in enumerate(tqdm(cases, desc='mechanism')):
        n = min(MAX_POINTS, case['total_points'])
        idx = rng.choice(case['total_points'], n, replace=False)
        pa, ref = predict_case(model_a, spec_a, case, sc_a, indices=idx)
        pb, _ = predict_case(model_b, spec_b, case, sc_b, indices=idx)
        ra, rb = pa - ref, pb - ref
        abs_a += np.abs(ra).sum(); abs_b += np.abs(rb).sum(); npts += n

        dbp = case['sdf'][idx]
        for bi, m in enumerate(band_masks(dbp)):
            if m.any():
                mae_a[k, bi] = np.abs(ra[m]).mean()
                mae_b[k, bi] = np.abs(rb[m]).mean()

        xs, ys = case['x_coords'][idx], case['y_coords'][idx]
        spec_ac.append(radial_spectrum(grid_field(ra, xs, ys, nx, ny),
                                       hann, wenergy, bin_idx, len(centres)))
        spec_bc.append(radial_spectrum(grid_field(rb, xs, ys, nx, ny),
                                       hann, wenergy, bin_idx, len(centres)))

    print(f"\n[reproduction gate] sampled MAE: "
          f"{args.model_a} {abs_a/npts:.3f} (ref {ref_a}) | "
          f"{args.model_b} {abs_b/npts:.3f} (ref {ref_b})")

    ma, ca = mean_ci(mae_a)
    mb, cb = mean_ci(mae_b)
    imp = 100.0 * (ma - mb) / ma
    for i, lb in enumerate(BAND_LABELS):
        print(f"{lb.replace(chr(10), ' '):>20s}: {args.model_a} {ma[i]:.3f} | "
              f"{args.model_b} {mb[i]:.3f} | improvement {imp[i]:.1f}%")

    sa, sca = mean_ci(np.asarray(spec_ac))
    sb, scb = mean_ci(np.asarray(spec_bc))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    x = np.arange(nb); w = 0.38
    axes[0].bar(x - w / 2, ma, w, yerr=ca, capsize=2.5,
                label=args.model_a.upper(), color='#77aa55',
                edgecolor='black', linewidth=0.5)
    axes[0].bar(x + w / 2, mb, w, yerr=cb, capsize=2.5,
                label=args.model_b.upper(), color='#333333',
                edgecolor='black', linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(BAND_LABELS, rotation=28, ha='right')
    axes[0].set_xlabel(r'Building-proximity feature band, $d_{\mathrm{BP}}$ (m)')
    axes[0].set_ylabel(r'MAE ($^\circ$C)')
    axes[0].set_title('(a)', loc='left'); axes[0].legend()

    ok = np.isfinite(sa) & np.isfinite(sb) & (centres > 0) & (sa > 0) & (sb > 0)
    axes[1].loglog(centres[ok], sa[ok], color='#77aa55', lw=1.7,
                   label=args.model_a.upper())
    axes[1].loglog(centres[ok], sb[ok], color='#333333', lw=1.7,
                   label=args.model_b.upper())
    for m, c, col in ((sa, sca, '#77aa55'), (sb, scb, '#333333')):
        axes[1].fill_between(centres[ok],
                             np.maximum(m[ok] - np.nan_to_num(c[ok]), 1e-30),
                             m[ok] + np.nan_to_num(c[ok]),
                             color=col, alpha=0.15, lw=0)
    axes[1].set_xlabel(r'Spatial frequency (m$^{-1}$)')
    axes[1].set_ylabel('Residual spectral power (a.u.)')
    axes[1].set_title('(b)', loc='left'); axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'mechanism_bands_spectrum.png'), dpi=300)

    with open(os.path.join(args.out, 'mechanism_bands.csv'), 'w',
              newline='') as f:
        wcsv = csv.writer(f)
        wcsv.writerow(['band', f'{args.model_a}_MAE', f'{args.model_a}_CI95',
                       f'{args.model_b}_MAE', f'{args.model_b}_CI95',
                       'improvement_pct'])
        for i, lb in enumerate(BAND_LABELS):
            wcsv.writerow([lb.replace('\n', ' '), ma[i], ca[i], mb[i], cb[i],
                           imp[i]])
    with open(os.path.join(args.out, 'mechanism_spectrum.pkl'), 'wb') as f:
        pickle.dump({'frequency': centres, 'a_mean': sa, 'a_ci': sca,
                     'b_mean': sb, 'b_ci': scb}, f)
    print(f"saved: {args.out}/mechanism_bands_spectrum.png, "
          f"mechanism_bands.csv, mechanism_spectrum.pkl")


if __name__ == '__main__':
    main()
