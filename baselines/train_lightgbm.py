"""Baseline: enhanced LightGBM on 11-D tabular features
[RH, T_amb, hour, x, y, z, d_BP, nx, ny, nz, category].

Usage:
    python baselines/train_lightgbm.py --data-root data/dummy --out runs/lgbm
"""

import os
import sys
import time
import pickle
import argparse

import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from urbanheat import config
from urbanheat.data import load_cases, split_cases, fit_scalers, normalize_coords
from urbanheat.engine import make_logger
from urbanheat.geometry import (SurfaceCategoryClassifier, load_building_points,
                                load_mesh)

FEATURES = ['H', 'T', 'Hr', 'x', 'y', 'z', 'sdf', 'nx', 'ny', 'nz', 'cat']

LGB_PARAMS = {
    'objective': 'regression', 'metric': 'mae', 'num_leaves': 127,
    'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'bagging_freq': 5, 'min_data_in_leaf': 100, 'verbose': -1,
    'n_jobs': -1, 'seed': 42,
}
N_ESTIMATORS = 1000


def make_tabular(cases, scalers, total_samples, desc):
    per_case = max(1, total_samples // len(cases))
    Xs, ys = [], []
    for c in tqdm(cases, desc=desc):
        n = min(per_case, c['total_points'])
        idx = np.random.choice(c['total_points'], n, replace=False) \
            if c['total_points'] > n else np.arange(c['total_points'])
        x, y, z = normalize_coords(c, idx)
        dbp = scalers['sdf'].transform(c['sdf'][idx].reshape(-1, 1)).flatten()
        p = c['params']
        br = scalers['branch'].transform(np.array(
            [p['humidity'], p['temperature'], p['hour']], dtype=np.float32).reshape(1, -1))[0]
        nrm = c['normal'][idx].astype(np.float32)
        X = np.zeros((n, 11), dtype=np.float32)
        X[:, 0], X[:, 1], X[:, 2] = br[0], br[1], br[2]
        X[:, 3], X[:, 4], X[:, 5], X[:, 6] = x, y, z, dbp
        X[:, 7:10] = nrm
        X[:, 10] = c['category'][idx].astype(np.float32)
        Xs.append(X)
        ys.append(scalers['output'].transform(
            c['temperature'][idx].reshape(-1, 1)).flatten().astype(np.float32))
    return np.concatenate(Xs), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--train-samples', type=int, default=2_000_000)
    ap.add_argument('--test-samples', type=int, default=1_000_000)
    ap.add_argument('--max-files', type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log = make_logger(args.out)
    root = args.data_root

    building_tree, building_z = load_building_points(os.path.join(root, 'Building.csv'))
    classifier = SurfaceCategoryClassifier(
        {name: os.path.join(root, f'{name}.csv') for name in config.CATEGORY_NAMES})
    _, face_tree, face_normals = load_mesh(os.path.join(root, 'building.stl'))

    cases = load_cases(os.path.join(root, 'cases'), building_tree=building_tree,
                       building_z=building_z, classifier=classifier,
                       face_tree=face_tree, face_normals=face_normals,
                       max_files=args.max_files, log=log)
    train_cases, test_cases = split_cases(cases, log=log)
    scalers = fit_scalers(train_cases, log=log)

    np.random.seed(42)
    X_train, y_train = make_tabular(train_cases, scalers, args.train_samples, 'train tabular')
    X_test, y_test = make_tabular(test_cases, scalers, args.test_samples, 'test tabular')
    y_test_c = scalers['output'].inverse_transform(y_test.reshape(-1, 1)).flatten()
    log(f"train {X_train.shape} | test {X_test.shape} | features: {FEATURES}")

    dtr = lgb.Dataset(X_train, label=y_train, feature_name=FEATURES,
                      categorical_feature=['cat'])
    dva = lgb.Dataset(X_test, label=y_test, reference=dtr)
    t0 = time.time()
    model = lgb.train(LGB_PARAMS, dtr, num_boost_round=N_ESTIMATORS,
                      valid_sets=[dtr, dva], valid_names=['train', 'valid'],
                      callbacks=[lgb.early_stopping(stopping_rounds=50),
                                 lgb.log_evaluation(period=100)])
    log(f"trained in {(time.time()-t0)/60:.1f} min (best iter {model.best_iteration})")

    pred = model.predict(X_test, num_iteration=model.best_iteration)
    pred_c = scalers['output'].inverse_transform(pred.reshape(-1, 1)).flatten()
    mae = mean_absolute_error(y_test_c, pred_c)
    rmse = float(np.sqrt(mean_squared_error(y_test_c, pred_c)))
    r2 = r2_score(y_test_c, pred_c)
    log(f"[LightGBM 11D] MAE {mae:.4f} | RMSE {rmse:.4f} | R2 {r2:.4f}")

    cat_test = X_test[:, 10].astype(int)
    for c in range(config.N_CATEGORIES):
        m = cat_test == c
        if m.any():
            log(f"  {config.CATEGORY_NAMES[c]:<9s} MAE "
                f"{mean_absolute_error(y_test_c[m], pred_c[m]):.3f} ({m.sum():,} pts)")

    model.save_model(os.path.join(args.out, 'lgbm_model.txt'))
    with open(os.path.join(args.out, 'evaluation_results.pkl'), 'wb') as f:
        pickle.dump({'model_name': 'LightGBM 11D', 'mae': mae, 'rmse': rmse, 'r2': r2}, f)


if __name__ == '__main__':
    main()
