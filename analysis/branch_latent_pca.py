"""PCA of the 256-D Branch latent representations (paper Fig. 8c).

One Branch latent vector is extracted per test case (the branch input
is constant within a case), projected onto the top-2 principal
components and colored by hour. Reports the cumulative explained
variance of PC1/PC2.

Usage:
    python analysis/branch_latent_pca.py --data-root data/dummy \
        --run runs/m4 --model m4 --out analysis_out/branch_pca
"""

import os
import sys
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import load_run, DEVICE          # noqa: E402
from urbanheat.data import (load_cases, split_cases,  # noqa: E402
                            basic_branch)
from urbanheat.solar import solar_branch              # noqa: E402


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
    branch_fn = solar_branch if spec['branch'] == 'solar' else basic_branch

    # branch latents need only the case conditions - no geometry features
    cases = load_cases(os.path.join(args.data_root, 'cases'),
                       max_files=args.max_files)
    _, test_cases = split_cases(cases)

    latents, hours = [], []
    with torch.no_grad():
        for c in test_cases:
            b = scalers['branch'].transform(
                branch_fn(c['params']).reshape(1, -1)).astype(np.float32)
            z = model.branch_net(torch.from_numpy(b).to(DEVICE))
            latents.append(z.cpu().numpy()[0])
            hours.append(c['params']['hour'])
    latents = np.asarray(latents)
    hours = np.asarray(hours)
    print(f"branch latents: {latents.shape} from {len(test_cases)} test cases")

    centred = latents - latents.mean(axis=0)
    cov = np.cov(centred, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    z2 = centred @ evecs[:, :2]
    evr = evals / evals.sum()
    print(f"explained variance: PC1 {100*evr[0]:.1f}% | PC2 {100*evr[1]:.1f}% "
          f"| cumulative {100*(evr[0]+evr[1]):.1f}%")

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    sc = ax.scatter(z2[:, 0], z2[:, 1], c=hours, cmap='viridis', s=28)
    fig.colorbar(sc, ax=ax, label='Hour')
    ax.set_xlabel(f'PC1 ({100*evr[0]:.1f}%)')
    ax.set_ylabel(f'PC2 ({100*evr[1]:.1f}%)')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'branch_latent_pca.png'), dpi=300)

    np.savez(os.path.join(args.out, 'branch_latent_pca.npz'),
             latents=latents, pc=z2, hours=hours, explained=evr)
    print(f"saved: {args.out}/branch_latent_pca.png, branch_latent_pca.npz")


if __name__ == '__main__':
    main()
