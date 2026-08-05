# urban-heat-deeponet

Code for the paper 
Taeyoon Kim (Pukyong National University), Yongjin Choi (KAIST),
Jaekyoung Kim (Konkuk University; corresponding author, jkkim4769@konkuk.ac.kr).

DeepONet-based surrogate models for CFD-simulated urban surface
temperature fields, including the full ablation ladder (M1-M5, M5-mat),
the baseline comparison models and the analysis code used in the paper.

## Model overview

| Flag (`--model`) | Paper name | Description |
|---|---|---|
| `m1` | M1 | Pure DeepONet (inner-product coupling) |
| `m2` | M2 | + signed building-proximity feature d_BP, dual-path coupling |
| `m3` | M3 | + multiscale Fourier embedding |
| `m4` | M4 | + 5-category surface embedding (proposed model) |
| `m5_seb` | SEB variant | + surface normals + SEB physics loss |
| `m5` | M5 | + solar branch (RH, T, altitude, azimuth, DNI) |
| `m5_mat` | M5-mat | + material properties (emissivity, reflectivity, transmissivity) |

Baselines (paper Table 4): `baselines/train_mlp_h768.py` (parameter-matched
single-path MLP) and `baselines/train_lightgbm.py` (11-D LightGBM).

## Data

The CFD dataset used in the paper is not included in this repository
(see the data availability statement of the paper). The files in
`data/dummy/` are synthetic samples - random numbers with the correct
file layout and schema - that allow end-to-end execution of the full
pipeline. They are not physically meaningful.

See `data/README_data.md` for the expected data layout and schema.

## Quick start

```bash
pip install -r requirements.txt

# generate the synthetic dummy dataset
python scripts/make_dummy_data.py --out data/dummy

# train + evaluate one model (1 epoch, dummy data)
python main.py --model m4 --data-root data/dummy --out runs/m4 --epochs 1

# full training on the real dataset
python main.py --model m4 --data-root /path/to/data --out runs/m4
```

Baselines:

```bash
python baselines/train_mlp_h768.py --data-root data/dummy --out runs/mlp --epochs 1
python baselines/train_lightgbm.py --data-root data/dummy --out runs/lgbm
```

Analysis (each script loads trained runs from `runs/`):

```bash
python analysis/ood_extrapolation.py --data-root data/dummy \
    --ood-dir /path/to/ood_cases --runs runs --models m1 m2 m3 m4 m5 \
    --out analysis_out/ood
python analysis/integrated_gradients.py --data-root data/dummy \
    --run runs/m4 --model m4 --out analysis_out/ig
python analysis/embedding_analysis.py --model m4 --runs runs/m4 \
    --out analysis_out/embedding
python analysis/facet_decomposition.py --data-root data/dummy \
    --run runs/m4 --model m4 --out analysis_out/facet
python analysis/morphology_correlation.py --data-root data/dummy \
    --run runs/m4 --model m4 --out analysis_out/morphology
python analysis/seb_residual.py --data-root data/dummy \
    --run-a runs/m4 --model-a m4 --run-b runs/m5 --model-b m5 \
    --out analysis_out/seb_residual
```

## Repository layout

```
main.py                     training / evaluation entry point
urbanheat/                  shared library
  config.py                 fixed experiment constants
  geometry.py               d_BP feature, categories, normals
  solar.py                  solar table, radiative properties, shadows
  seb.py                    SEB physics loss
  data.py                   loading, split, scalers, dataset
  models.py                 all DeepONet variants
  engine.py                 training / evaluation loops
baselines/                  MLP and LightGBM comparison models
analysis/                   OOD, IG, embeddings, facets, morphology, SEB residual
utils/dbp_feature.py        standalone d_BP reference implementation
scripts/make_dummy_data.py  synthetic data generator
```

## Notes on reproducibility

The train/test split is condition-based (seed 42): all cases sharing an
(RH, T_amb) combination fall on the same side, so no meteorological
condition leaks between the two sets. The integrated-gradients protocol
uses 30 steps, 5,000 points per case, 40 cases and seed 0, matching the
paper.

The experiments in the paper were run with Python 3.8, PyTorch 2.4.1
(CUDA 11.8) on a single NVIDIA L40S (46 GB); `requirements.txt` pins
these versions. Full training takes roughly 20 h per model on this GPU;
the dummy dataset runs in minutes on CPU.

## License

MIT - see `LICENSE`.
