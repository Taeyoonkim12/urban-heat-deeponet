"""Sensitivity baseline: Fourier-enriched LightGBM (264-D).

Answers the reviewer-facing question left open by the 8-D enriched
LightGBM: does providing the tree model with the SAME multiscale Fourier
representation used by M3/M4 close the gap to the operator model?

Inputs (264-D):
    [RH, T_amb, hour]                       3   (branch, standardized)
    [x, y, z, d_BP]                         4   (normalized coords + scaled d_BP)
    sin/cos multiscale Fourier projections  256 (identical B matrix to M3/M4:
                                                 MultiScaleFourierFeatures,
                                                 input_dim=4, FOURIER_SEED)
    surface category                        1   (single categorical feature)

Normals are excluded (consistent with the manuscript). Tuning follows the
same Table 4 grid search as the plain/8-D LightGBM baselines.

Usage:
    python baselines/train_lightgbm_fourier.py --data-root data/dummy --out runs/lgbm_fourier
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
from urbanheat.geometry import SurfaceCategoryClassifier, load_building_points
from train_lightgbm import GRID, BASE_PARAMS, N_ROUNDS, condition_holdout  # noqa: E402


def fourier_matrix():
    """The exact multiscale Fourier B matrix used by M3/M4 (seed-identical)."""
    from urbanheat.models import MultiScaleFourierFeatures
    module = MultiScaleFourierFeatures(input_dim=4)
    return module.B.detach().cpu().numpy().astype(np.float32)   # (4, 128)


FEATURES = (['H', 'T', 'Hr', 'x', 'y', 'z', 'd_bp'] +
            [f'sin{i}' for i in range(128)] + [f'cos{i}' for i in range(128)] +
            ['cat'])


def make_tabular(cases, scalers, B, total_samples, desc):
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
            [p['humidity'], p['temperature'], p['hour']],
            dtype=np.float32).reshape(1, -1))[0]
        coords4 = np.column_stack([x, y, z, dbp]).astype(np.float32)
        proj = 2.0 * np.pi * coords4 @ B                      # (n, 128)
        X = np.empty((n, 264), dtype=np.float32)
        X[:, 0], X[:, 1], X[:, 2] = br[0], br[1], br[2]
        X[:, 3:7] = coords4
        X[:, 7:135] = np.sin(proj)
        X[:, 135:263] = np.cos(proj)
        X[:, 263] = c['category'][idx].astype(np.float32)
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
    ap.add_argument('--cases-dir', default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log = make_logger(args.out)
    root = args.data_root
    B = fourier_matrix()
    log(f"Fourier B matrix: {B.shape}, seed-identical to M3/M4 "
        f"(first values {B[0, :3]})")

    building_tree, building_z = load_building_points(os.path.join(root, 'Building.csv'))
    classifier = SurfaceCategoryClassifier(
        {n: os.path.join(root, f'{n}.csv') for n in config.CATEGORY_NAMES})
    cases = load_cases(args.cases_dir or os.path.join(root, 'cases'),
                       building_tree=building_tree, building_z=building_z,
                       classifier=classifier, max_files=args.max_files, log=log)
    train_cases, test_cases = split_cases(cases, log=log)
    scalers = fit_scalers(train_cases, log=log)

    np.random.seed(42)
    fit_cases, val_cases = condition_holdout(train_cases)
    X_fit, y_fit = make_tabular(fit_cases, scalers, B,
                                int(args.train_samples * 0.8), 'grid-fit tabular')
    X_val, y_val = make_tabular(val_cases, scalers, B,
                                int(args.train_samples * 0.2), 'grid-val tabular')

    log(f"grid search: {len(GRID)} combinations (same procedure as the "
        f"plain/8-D LightGBM)")
    best = None
    for nl, lr, mdl in GRID:
        params = dict(BASE_PARAMS, num_leaves=nl, learning_rate=lr,
                      min_data_in_leaf=mdl)
        dfit = lgb.Dataset(X_fit, label=y_fit, feature_name=FEATURES,
                           categorical_feature=['cat'])
        dval = lgb.Dataset(X_val, label=y_val, reference=dfit)
        t0 = time.time()
        m = lgb.train(params, dfit, num_boost_round=N_ROUNDS,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(stopping_rounds=50,
                                                    verbose=False)])
        val_mae = float(mean_absolute_error(
            y_val, m.predict(X_val, num_iteration=m.best_iteration)))
        log(f"  leaves={nl:>3d} lr={lr:.2f} min_data={mdl:>3d} "
            f"-> val MAE {val_mae:.5f} (iter {m.best_iteration}, "
            f"{(time.time()-t0)/60:.1f} min)")
        if best is None or val_mae < best[0]:
            best = (val_mae, params, m.best_iteration)

    _, best_params, best_iter = best
    log(f"best config: {best_params} | rounds {best_iter}")

    np.random.seed(42)
    X_train, y_train = make_tabular(train_cases, scalers, B,
                                    args.train_samples, 'train tabular')
    X_test, y_test = make_tabular(test_cases, scalers, B,
                                  args.test_samples, 'test tabular')
    y_test_c = scalers['output'].inverse_transform(y_test.reshape(-1, 1)).flatten()

    t0 = time.time()
    dtr = lgb.Dataset(X_train, label=y_train, feature_name=FEATURES,
                      categorical_feature=['cat'])
    model = lgb.train(best_params, dtr, num_boost_round=best_iter)
    log(f"retrained in {(time.time()-t0)/60:.1f} min")

    pred_c = scalers['output'].inverse_transform(
        model.predict(X_test).reshape(-1, 1)).flatten()
    mae = mean_absolute_error(y_test_c, pred_c)
    rmse = float(np.sqrt(mean_squared_error(y_test_c, pred_c)))
    r2 = r2_score(y_test_c, pred_c)
    log(f"[Fourier LightGBM 264D] MAE {mae:.4f} | RMSE {rmse:.4f} | R2 {r2:.4f}")

    # Confirm the Fourier features are actually used: top gain importances.
    imp = model.feature_importance(importance_type='gain')
    order = np.argsort(imp)[::-1][:10]
    log("top-10 features by gain: " +
        ", ".join(f"{FEATURES[i]}({imp[i]:.0f})" for i in order))
    fourier_gain = imp[7:263].sum() / max(imp.sum(), 1)
    log(f"Fourier features share of total gain: {100*fourier_gain:.1f}%")

    cat_test = X_test[:, 263].astype(int)
    for ci, name in enumerate(config.CATEGORY_NAMES):
        m = cat_test == ci
        if m.any():
            log(f"  {name:<9s} MAE {mean_absolute_error(y_test_c[m], pred_c[m]):.3f} "
                f"({m.sum():,} pts)")

    model.save_model(os.path.join(args.out, 'lgbm_fourier.txt'))
    with open(os.path.join(args.out, 'evaluation_results.pkl'), 'wb') as f:
        pickle.dump({'model_name': 'Fourier LightGBM 264D', 'mae': mae,
                     'rmse': rmse, 'r2': r2, 'best_params': best_params,
                     'num_boost_round': best_iter,
                     'fourier_gain_share': float(fourier_gain)}, f)


if __name__ == '__main__':
    main()
