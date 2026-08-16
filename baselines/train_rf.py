"""Baseline: Random Forest on the plain 6-D inputs (paper Tables 4 and 7).

Same six-dimensional input as MLP-vanilla and M0
([RH, T_amb, hour, x, y, z], scaled/normalized identically) on a
2M-point train / 1M-point test subsample. Hyperparameters are selected
by grid search per the paper's Table 4 (n_estimators x max_depth,
5 x 3 = 15 combinations) on a case-grouped validation split holding out 20% of the training
(RH, T_amb) conditions; the best configuration is retrained on the full
training set and evaluated once on the test set.

Usage:
    python baselines/train_rf.py --data-root data/dummy --out runs/rf
"""

import os
import sys
import time
import pickle
import argparse
import itertools

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from urbanheat.data import load_cases, split_cases, fit_scalers
from urbanheat.engine import make_logger
from urbanheat.geometry import load_building_points
from train_lightgbm import make_tabular, condition_holdout, FEATURES  # noqa: E402

GRID = list(itertools.product([50, 100, 200, 500, 1000],  # n_estimators
                              [15, 20, 30]))              # max_depth
FIXED = dict(min_samples_split=20, min_samples_leaf=10, max_features=0.7,
             n_jobs=-1, random_state=42)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--train-samples', type=int, default=2_000_000)
    ap.add_argument('--test-samples', type=int, default=1_000_000)
    ap.add_argument('--grid-samples', type=int, default=500_000,
                    help='fit-subsample size per grid combination '
                         '(full grid on 2M points is impractical for RF)')
    ap.add_argument('--max-files', type=int, default=None)
    ap.add_argument('--cases-dir', default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log = make_logger(args.out)
    root = args.data_root

    building_tree, building_z = load_building_points(os.path.join(root, 'Building.csv'))
    cases = load_cases(args.cases_dir or os.path.join(root, 'cases'),
                       building_tree=building_tree, building_z=building_z,
                       max_files=args.max_files, log=log)
    train_cases, test_cases = split_cases(cases, log=log)
    scalers = fit_scalers(train_cases, log=log)

    np.random.seed(42)
    fit_cases, val_cases = condition_holdout(train_cases)
    X_fit, y_fit = make_tabular(fit_cases, scalers,
                                args.grid_samples, 'grid-fit tabular')
    X_val, y_val = make_tabular(val_cases, scalers,
                                max(1, args.grid_samples // 4), 'grid-val tabular')

    log(f"grid search: {len(GRID)} combinations (n_estimators x max_depth)")
    best = None
    for ne, md in GRID:
        t0 = time.time()
        m = RandomForestRegressor(n_estimators=ne, max_depth=md, **FIXED)
        m.fit(X_fit, y_fit)
        val_mae = float(mean_absolute_error(y_val, m.predict(X_val)))
        log(f"  n_estimators={ne:>4d} max_depth={str(md):>4s} "
            f"-> val MAE {val_mae:.5f} ({(time.time()-t0)/60:.1f} min)")
        if best is None or val_mae < best[0]:
            best = (val_mae, ne, md)
        del m

    _, best_ne, best_md = best
    log(f"best config: n_estimators={best_ne}, max_depth={best_md}")

    np.random.seed(42)
    X_train, y_train = make_tabular(train_cases, scalers,
                                    args.train_samples, 'train tabular')
    X_test, y_test = make_tabular(test_cases, scalers,
                                  args.test_samples, 'test tabular')
    y_test_c = scalers['output'].inverse_transform(y_test.reshape(-1, 1)).flatten()

    t0 = time.time()
    model = RandomForestRegressor(n_estimators=best_ne, max_depth=best_md,
                                  **FIXED)
    model.fit(X_train, y_train)
    log(f"retrained in {(time.time()-t0)/60:.1f} min")

    pred_c = scalers['output'].inverse_transform(
        model.predict(X_test).reshape(-1, 1)).flatten()
    mae = mean_absolute_error(y_test_c, pred_c)
    rmse = float(np.sqrt(mean_squared_error(y_test_c, pred_c)))
    r2 = r2_score(y_test_c, pred_c)
    log(f"[Random Forest 6D] MAE {mae:.4f} | RMSE {rmse:.4f} | R2 {r2:.4f}")

    with open(os.path.join(args.out, 'evaluation_results.pkl'), 'wb') as f:
        pickle.dump({'model_name': 'Random Forest 6D', 'mae': mae,
                     'rmse': rmse, 'r2': r2,
                     'best_params': {'n_estimators': best_ne,
                                     'max_depth': best_md, **FIXED},
                     'features': FEATURES}, f)


if __name__ == '__main__':
    main()
