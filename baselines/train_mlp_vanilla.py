"""Baseline: single-path vanilla MLP (paper name: MLP-vanilla).

Same 6-D input information as M0 ([RH, T_amb, hour, x, y, z]) and the
same training protocol, but a plain residual MLP with hidden width 768
(parameter-matched to M0) instead of the branch/trunk operator split.

Usage:
    python baselines/train_mlp_vanilla.py --data-root data/dummy --out runs/mlp --epochs 1
"""

import os
import sys
import argparse

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from urbanheat import config
from urbanheat.data import UrbanHeatDataset, load_cases, split_cases, fit_scalers
from urbanheat.engine import train, evaluate, make_logger
from urbanheat.models import MODEL_SPECS


class MLPVanilla6D(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=768, depth=6, dropout=0.1):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                          nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
            for _ in range(depth - 2)])
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.output_bias = nn.Parameter(torch.zeros(1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, branch_in, trunk_in, cat_idx=None):
        x = torch.cat([branch_in, trunk_in], dim=-1)
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x) + x
        return self.output_layer(x) + self.output_bias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=config.TOTAL_EPOCHS)
    ap.add_argument('--points-per-case', type=int, default=config.POINTS_PER_CASE)
    ap.add_argument('--max-files', type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log = make_logger(args.out)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Same input pipeline as M0: coordinates only, no d_BP/categories.
    spec = MODEL_SPECS['m0']
    cases = load_cases(os.path.join(args.data_root, 'cases'),
                       max_files=args.max_files, log=log)
    train_cases, test_cases = split_cases(cases, log=log)
    scalers = fit_scalers(train_cases, log=log)

    def dataset_cls(cases_, scalers_, spec_):
        return UrbanHeatDataset(cases_, scalers_, spec_,
                                points_per_sample=args.points_per_case)

    model = MLPVanilla6D().to(device)
    train(model, spec, train_cases, test_cases, scalers, args.out,
          dataset_cls, epochs=args.epochs, device=device, log=log)
    evaluate(model, spec, test_cases, scalers, args.out, dataset_cls,
             'MLP-vanilla (hidden width 768)', device=device, log=log)


if __name__ == '__main__':
    main()
