"""Baseline: enriched MLP (Supplementary Table S1, short protocol).

A single-path MLP that receives all of M4's input information as one
flattened 268-D vector: branch 3 + coords/d_BP 4 + multiscale Fourier
features 256 (identical projection matrix, seed 42) + one-hot category
5. Depth and hidden width are auto-matched to M4's parameter count
(within 1%). The supplementary comparison uses the unified short
protocol: 200 epochs, early-stopping patience 40, identical data
pipeline; M4 is retrained under the same protocol for fairness
(``python main.py --model m4 --epochs 200 --patience 40 ...``).

Usage:
    python baselines/train_mlp_enriched.py --data-root data/dummy --out runs/mlp_enriched
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from urbanheat import config
from urbanheat.data import UrbanHeatDataset, load_cases, split_cases, fit_scalers
from urbanheat.engine import train, evaluate, make_logger
from urbanheat.geometry import SurfaceCategoryClassifier, load_building_points
from urbanheat.models import MODEL_SPECS, MultiScaleFourierFeatures

BRANCH_DIM, TRUNK_DIM = 3, 4


class EnrichedMLP(nn.Module):
    """Single-path MLP over the flattened 268-D M4 input. Keeps the
    (branch, trunk, cat) interface of the engine."""

    def __init__(self, hidden_dim=512, depth=8, dropout=0.1):
        super().__init__()
        self.fourier = MultiScaleFourierFeatures(TRUNK_DIM)   # seed 42, as M4
        in_dim = (BRANCH_DIM + TRUNK_DIM + self.fourier.output_dim
                  + config.N_CATEGORIES)
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                          nn.LayerNorm(hidden_dim), nn.GELU(),
                          nn.Dropout(dropout))
            for _ in range(depth)])
        self.output_layer = nn.Linear(hidden_dim, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, branch_in, trunk_in, cat_idx):
        ff = self.fourier(trunk_in)
        oh = torch.nn.functional.one_hot(cat_idx, config.N_CATEGORIES).float()
        x = torch.cat([branch_in, trunk_in, ff, oh], dim=-1)
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x) + x
        return self.output_layer(x)


def m4_param_count():
    from urbanheat.models import build_model
    m, _ = build_model('m4')
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def match_capacity(target):
    """Scan depth x hidden width for the closest parameter match."""
    best = None
    for d in range(6, 13):
        for h in range(384, 705, 16):
            n = sum(p.numel() for p in EnrichedMLP(hidden_dim=h, depth=d).parameters())
            if best is None or abs(n - target) < abs(best[2] - target):
                best = (d, h, n)
    assert abs(best[2] - target) / target < 0.01, f"no match: {best}"
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--patience', type=int, default=40)
    ap.add_argument('--points-per-case', type=int, default=config.POINTS_PER_CASE)
    ap.add_argument('--max-files', type=int, default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cases-dir', default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log = make_logger(args.out)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    target = m4_param_count()
    depth, hidden, n_params = match_capacity(target)
    log(f"EnrichedMLP: depth={depth}, hidden={hidden}, params={n_params:,} "
        f"(M4 target {target:,}, {100*n_params/target-100:+.2f}%)")
    model = EnrichedMLP(hidden_dim=hidden, depth=depth).to(device)

    # M4's data spec supplies exactly the same batch layout.
    spec = MODEL_SPECS['m4']
    root = args.data_root
    building_tree, building_z = load_building_points(os.path.join(root, 'Building.csv'))
    classifier = SurfaceCategoryClassifier(
        {n: os.path.join(root, f'{n}.csv') for n in config.CATEGORY_NAMES})
    cases = load_cases(args.cases_dir or os.path.join(root, 'cases'),
                       building_tree=building_tree, building_z=building_z,
                       classifier=classifier, max_files=args.max_files, log=log)
    train_cases, test_cases = split_cases(cases, log=log)
    scalers = fit_scalers(train_cases, log=log)

    def dataset_cls(cases_, scalers_, spec_):
        return UrbanHeatDataset(cases_, scalers_, spec_,
                                points_per_sample=args.points_per_case)

    train(model, spec, train_cases, test_cases, scalers, args.out,
          dataset_cls, epochs=args.epochs, patience=args.patience,
          device=device, log=log)
    evaluate(model, spec, test_cases, scalers, args.out, dataset_cls,
             f'Enriched MLP 268D (hidden {hidden}, depth {depth})',
             device=device, log=log)


if __name__ == '__main__':
    main()
