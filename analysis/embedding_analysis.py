"""Learned surface-category embedding analysis (paper Sec. 3.5):
PCA projection and pairwise cosine similarity of the 5x8 embedding
matrix. Multiple runs (e.g. different seeds) can be passed to check
the stability of the embedding structure.

Usage:
    python analysis/embedding_analysis.py --model m4 \
        --runs runs/m4 [runs/m4_seed123 ...] --out analysis_out/embedding
"""

import os
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from common import load_run
from urbanheat.config import CATEGORY_NAMES

COLORS = ['#B85450', '#7f7f7f', '#55a868', '#4c72b0', '#937860']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='m4')
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    all_emb = []
    for run_dir in args.runs:
        model, spec, _, ckpt, _ = load_run(args.model, run_dir)
        emb = model.trunk_net.cat_embed.weight.detach().cpu().numpy()
        all_emb.append(emb)
        print(f"{run_dir}: embeddings {emb.shape} (epoch {ckpt.get('epoch', '?')})")
    np.save(os.path.join(args.out, 'category_embeddings.npy'), np.stack(all_emb))

    # Cosine similarity per run, then mean/std across runs.
    sims = np.stack([cosine_similarity(e) for e in all_emb])
    sim_mean, sim_std = sims.mean(axis=0), sims.std(axis=0)
    print("\npairwise cosine similarity (mean over runs):")
    header = ' ' * 10 + ' '.join(f"{n:>9s}" for n in CATEGORY_NAMES)
    print(header)
    for i, n in enumerate(CATEGORY_NAMES):
        row = ' '.join(f"{sim_mean[i, j]:+9.3f}" for j in range(len(CATEGORY_NAMES)))
        print(f"{n:>9s} {row}")
    if len(all_emb) > 1:
        print(f"max std across runs: {sim_std.max():.3f}")

    emb = all_emb[0]
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(emb)
    ev = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    for i, name in enumerate(CATEGORY_NAMES):
        ax.scatter(emb_2d[i, 0], emb_2d[i, 1], s=300, c=COLORS[i],
                   edgecolors='black', linewidth=1.2, zorder=3)
        ax.annotate(name, (emb_2d[i, 0], emb_2d[i, 1]), fontsize=11,
                    xytext=(9, 9), textcoords='offset points')
    ax.set_xlabel(f'PC1 ({ev[0] * 100:.1f}% var)')
    ax.set_ylabel(f'PC2 ({ev[1] * 100:.1f}% var)')
    ax.set_title('(a) PCA projection of category embeddings')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.grid(alpha=0.3)

    im = axes[1].imshow(sim_mean, vmin=-1, vmax=1, cmap='RdBu_r')
    axes[1].set_xticks(range(len(CATEGORY_NAMES)))
    axes[1].set_xticklabels(CATEGORY_NAMES, rotation=45, ha='right')
    axes[1].set_yticks(range(len(CATEGORY_NAMES)))
    axes[1].set_yticklabels(CATEGORY_NAMES)
    for i in range(len(CATEGORY_NAMES)):
        for j in range(len(CATEGORY_NAMES)):
            axes[1].text(j, i, f"{sim_mean[i, j]:.2f}", ha='center', va='center',
                         fontsize=9)
    axes[1].set_title('(b) Pairwise cosine similarity')
    plt.colorbar(im, ax=axes[1], shrink=0.85)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'embedding_analysis.png'), dpi=300)
    print(f"saved results to {args.out}")


if __name__ == '__main__':
    main()
