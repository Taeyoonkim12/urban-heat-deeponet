"""Integrated-gradients sensitivity of the branch inputs (paper Sec. 2.5.2,
Fig. 5). Riemann approximation with 30 steps (31 evaluation points),
5,000 points per case, up to 40 test cases, seed 0.

Usage:
    python analysis/integrated_gradients.py --data-root data/dummy \
        --run runs/m4 --model m4 --out analysis_out/ig
"""

import os
import pickle
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from common import DEVICE, load_run, load_test_cases, build_case_inputs

N_STEPS = 30
N_POINTS = 5000
N_CASES = 40
SEED = 0

BRANCH_NAMES = ['Humidity', 'Air Temp', 'Hour']


def integrated_gradients(model, branch_t, trunk_t, cat_t, n_steps=N_STEPS):
    baseline = torch.zeros_like(branch_t)
    alphas = torch.linspace(0, 1, n_steps + 1, device=branch_t.device).view(-1, 1, 1)
    accum = torch.zeros_like(branch_t)
    for i in range(n_steps + 1):
        b = (baseline + alphas[i] * (branch_t - baseline)).detach().requires_grad_(True)
        pred = model(b, trunk_t, cat_t).sum()
        accum = accum + torch.autograd.grad(pred, b)[0]
    return (branch_t - baseline) * accum / (n_steps + 1)


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
    test_cases, _ = load_test_cases(args.data_root, spec, max_files=args.max_files)

    np.random.seed(SEED)
    selected = np.random.choice(len(test_cases), min(N_CASES, len(test_cases)),
                                replace=False)
    results = []
    for ci in tqdm(selected, desc='IG'):
        case = test_cases[ci]
        n = min(N_POINTS, case['total_points'])
        idx = np.random.choice(case['total_points'], n, replace=False)
        branch, trunk, cat = build_case_inputs(case, idx, spec, scalers)
        bt = torch.from_numpy(branch).to(DEVICE)
        tt = torch.from_numpy(trunk).to(DEVICE)
        ct = None if cat is None else torch.from_numpy(cat).to(DEVICE)
        with torch.enable_grad():
            ig = integrated_gradients(model, bt, tt, ct)
        results.append({'hour': case['params']['hour'],
                        'ig_mean': ig.abs().detach().cpu().numpy().mean(axis=0)})

    with open(os.path.join(args.out, 'ig_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    df = pd.DataFrame([{'hour': r['hour'],
                        **{n: v for n, v in zip(BRANCH_NAMES, r['ig_mean'])}}
                       for r in results])
    df.to_csv(os.path.join(args.out, 'ig_by_case.csv'), index=False)

    mean_attr = df[BRANCH_NAMES].mean()
    print("\nmean |IG| attribution:")
    print(mean_attr.round(4).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].bar(BRANCH_NAMES, mean_attr.values, color=['#3498DB', '#E67E22', '#27AE60'])
    axes[0].set_ylabel('mean |IG|')
    axes[0].set_title('(a) Branch input attribution')
    hourly = df.groupby('hour')[BRANCH_NAMES].mean()
    for n in BRANCH_NAMES:
        axes[1].plot(hourly.index, hourly[n], '-o', markersize=4, label=n)
    axes[1].set_xlabel('Hour')
    axes[1].set_ylabel('mean |IG|')
    axes[1].set_title('(b) Attribution by hour')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'ig_attribution.png'), dpi=300)
    print(f"saved results to {args.out}")


if __name__ == '__main__':
    main()
