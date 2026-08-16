"""Baseline: LightGBM on the plain 6-D inputs (paper Tables 4 and 7).

Same six-dimensional input as MLP-vanilla and M0
([RH, T_amb, hour, x, y, z], scaled/normalized identically) on a
2M-point train / 1M-point test subsample. Hyperparameters are selected
by grid search per the paper's Table 4 (num_leaves x learning_rate x
min_data_in_leaf, 4 x 3 x 1 = 12 combinations, 2000 boosting rounds) on
a case-grouped validation split holding out 20% of the training
(RH, T_amb) conditions; the best configuration is retrained on
the full training set and evaluated once on the test set.

Usage:
    python baselines/train_lightgbm.py --data-root data/dummy --out runs/lgbm
"""

import os
import sys
import time
import pickle
import argparse
import itertools
from collections import defaultdict

import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from urbanheat.data import load_cases, split_cases, fit_scalers, normalize_coords
from urbanheat.engine import make_logger
from urbanheat.geometry import load_building_points

FEATURES = ['H', 'T', 'Hr', 'x', 'y', 'z']

GRID = list(itertools.product([31, 63, 127, 255],   # num_leaves
                              [0.01, 0.05, 0.1],     # learning_rate
                              [100]))                # min_data_in_leaf
BASE_PARAMS = {
    'objective': 'regression', 'metric': 'mae',
    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'verbose': -1, 'n_jobs': -1, 'seed': 42,
}
N_ROUNDS = 2000


def make_tabular(cases, scalers, total_samples, desc):
    per_case = max(1, total_samples // len(cases))
    Xs, ys = [], []
    for c in tqdm(cases, desc=desc):
        n = min(per_case, c['total_points'])
        idx = np.random.choice(c['total_points'], n, replace=False) \
            if c['total_points'] > n else np.arange(c['total_points'])
        x, y, z = normalize_coords(c, idx)
        p = c['params']
        br = scalers['branch'].transform(np.array(
            [p['humidity'], p['temperature'], p['hour']],
            dtype=np.float32).reshape(1, -1))[0]
        X = np.zeros((n, 6), dtype=np.float32)
        X[:, 0], X[:, 1], X[:, 2] = br[0], br[1], br[2]
        X[:, 3], X[:, 4], X[:, 5] = x, y, z
        Xs.append(X)
        ys.append(scalers['output'].transform(
            c['temperature'][idx].reshape(-1, 1)).flatten().astype(np.float32))
    return np.concatenate(Xs), np.concatenate(ys)


def condition_holdout(train_cases, frac=0.2, seed=42):
    """Case-grouped validation split on the (RH, T_amb) conditions."""
    groups = defaultdict(list)
    for c in train_cases:
        groups[(c['params']['humidity'], c['params']['temperature'])].append(c)
    keys = sorted(groups.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_val = max(1, int(len(keys) * frac))
    val = [c for k in keys[:n_val] for c in groups[k]]
    fit = [c for k in keys[n_val:] for c in groups[k]]
    return fit, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--train-samples', type=int, default=2_000_000)
    ap.add_argument('--test-samples', type=int, default=1_000_000)
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
                                int(args.train_samples * 0.8), 'grid-fit tabular')
    X_val, y_val = make_tabular(val_cases, scalers,
                                int(args.train_samples * 0.2), 'grid-val tabular')

    log(f"grid search: {len(GRID)} combinations "
        f"(num_leaves x learning_rate x min_data_in_leaf)")
    best = None
    for nl, lr, mdl in GRID:
        params = dict(BASE_PARAMS, num_leaves=nl, learning_rate=lr,
                      min_data_in_leaf=mdl)
        dfit = lgb.Dataset(X_fit, label=y_fit, feature_name=FEATURES)
        dval = lgb.Dataset(X_val, label=y_val, reference=dfit)
        m = lgb.train(params, dfit, num_boost_round=N_ROUNDS,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(stopping_rounds=50,
                                                    verbose=False)])
        val_mae = float(mean_absolute_error(
            y_val, m.predict(X_val, num_iteration=m.best_iteration)))
        log(f"  leaves={nl:>3d} lr={lr:.2f} min_data={mdl:>3d} "
            f"-> val MAE {val_mae:.5f} (iter {m.best_iteration})")
        if best is None or val_mae < best[0]:
            best = (val_mae, params, m.best_iteration)

    _, best_params, best_iter = best
    log(f"best config: {best_params} | rounds {best_iter}")

    # retrain the selected configuration on the full training subsample
    np.random.seed(42)
    X_train, y_train = make_tabular(train_cases, scalers,
                                    args.train_samples, 'train tabular')
    X_test, y_test = make_tabular(test_cases, scalers,
                                  args.test_samples, 'test tabular')
    y_test_c = scalers['output'].inverse_transform(y_test.reshape(-1, 1)).flatten()

    t0 = time.time()
    dtr = lgb.Dataset(X_train, label=y_train, feature_name=FEATURES)
    model = lgb.train(best_params, dtr, num_boost_round=best_iter)
    log(f"retrained in {(time.time()-t0)/60:.1f} min")

    pred_c = scalers['output'].inverse_transform(
        model.predict(X_test).reshape(-1, 1)).flatten()
    mae = mean_absolute_error(y_test_c, pred_c)
    rmse = float(np.sqrt(mean_squared_error(y_test_c, pred_c)))
    r2 = r2_score(y_test_c, pred_c)
    log(f"[LightGBM 6D] MAE {mae:.4f} | RMSE {rmse:.4f} | R2 {r2:.4f}")

    model.save_model(os.path.join(args.out, 'lgbm_model.txt'))
    with open(os.path.join(args.out, 'evaluation_results.pkl'), 'wb') as f:
        pickle.dump({'model_name': 'LightGBM 6D', 'mae': mae, 'rmse': rmse,
                     'r2': r2, 'best_params': best_params,
                     'num_boost_round': best_iter, 'features': FEATURES}, f)


if __name__ == '__main__':
    main()
