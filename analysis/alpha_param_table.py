"""Report trainable parameter counts and learned alpha for every model.

Parameter counts come from the architecture alone (no checkpoint needed).
Learned sigmoid(alpha) is extracted from each run's checkpoint when a
runs directory is given; inner-product models (m0, d1) have no alpha by
construction.

Usage:
    python -m analysis.alpha_param_table --runs runs
(expects runs/<model>/best_model.pt for each trained model)
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from urbanheat.models import MODEL_SPECS, build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default=None,
                    help='directory containing <model>/best_model.pt runs')
    args = ap.parse_args()

    rows = []
    for name in MODEL_SPECS:
        model, spec = build_model(name)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        alpha = 'n/a (inner)'
        epoch = '-'
        path = (os.path.join(args.runs, name, 'best_model.pt')
                if args.runs else None)
        if spec['coupling'] == 'dual':
            alpha = '(no ckpt)'
        if path and os.path.exists(path):
            ck = torch.load(path, map_location='cpu', weights_only=False)
            sd = ck.get('model_state_dict', ck)
            if 'alpha' in sd:
                alpha = f"{torch.sigmoid(sd['alpha']).item():.4f}"
            epoch = str(ck.get('epoch', '?'))
        rows.append((name, n_params, alpha, epoch))

    print(f"{'model':>14s} {'trainable params':>17s} {'sigmoid(alpha)':>15s} "
          f"{'best epoch':>10s}")
    for name, n, a, e in rows:
        print(f"{name:>14s} {n:>17,d} {a:>15s} {e:>10s}")


if __name__ == '__main__':
    main()
