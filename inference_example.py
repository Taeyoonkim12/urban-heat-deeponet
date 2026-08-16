"""Minimal inference example.

Loads a trained run (checkpoint + scalers), predicts the surface
temperature field for a single CFD case CSV, prints the error metrics
and writes the predictions to an .npz file.

Usage:
    python inference_example.py --run runs/m4 --model m4 \
        --case data/dummy/cases/Case_60_35_12_XYZInternalTable.csv \
        --data-root data/dummy --out predictions_m4.npz
"""

import os
import argparse

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from analysis.common import load_run, load_geometry, predict_case
from urbanheat.data import load_cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='run directory (best_model.pt)')
    ap.add_argument('--model', required=True, help='model flag, e.g. m4')
    ap.add_argument('--case', required=True, help='one Case_*.csv file')
    ap.add_argument('--data-root', required=True,
                    help='root with Building.csv etc. for the geometry features')
    ap.add_argument('--out', default='predictions.npz')
    args = ap.parse_args()

    model, spec, scalers, _, _ = load_run(args.model, args.run)
    geo = load_geometry(args.data_root, spec)
    cases = load_cases(os.path.dirname(args.case),
                       only=os.path.basename(args.case), **geo)
    case = cases[0]

    pred, ref = predict_case(model, spec, case, scalers)
    mae = mean_absolute_error(ref, pred)
    rmse = float(np.sqrt(mean_squared_error(ref, pred)))
    r2 = r2_score(ref, pred)
    print(f"{case['filename']}: {len(pred):,} points | "
          f"MAE {mae:.4f} C | RMSE {rmse:.4f} C | R2 {r2:.4f}")

    np.savez(args.out, x=case['x_coords'], y=case['y_coords'],
             z=case['z_coords'], t_pred=pred, t_cfd=ref)
    print(f"predictions written to {args.out}")


if __name__ == '__main__':
    main()
