# urban-heat-deeponet

Code for the paper (full citation will be added upon publication).

Taeyoon Kim (Pukyong National University), Yongjin Choi (KAIST),
Jaekyoung Kim (Konkuk University; corresponding author, jkkim4769@konkuk.ac.kr).

DeepONet-based surrogate models for CFD-simulated urban surface
temperature fields: the model ladder M0-M5, the baseline comparison
models, two supplementary controls and the analysis code used in the
paper.

## Paper-to-code correspondence

| Paper label | Description | Training command | Distinguishing component |
|---|---|---|---|
| MLP-vanilla | Parameter-matched single-path neural baseline (hidden width 768) | `python baselines/train_mlp_vanilla.py --data-root D --out runs/mlp` | no branch/trunk operator split |
| Random Forest | Tree-ensemble baseline, plain 6-D inputs, grid-searched | `python baselines/train_rf.py --data-root D --out runs/rf` | non-neural reference |
| LightGBM | Gradient-boosted baseline, plain 6-D inputs, grid-searched | `python baselines/train_lightgbm.py --data-root D --out runs/lgbm` | non-neural reference |
| M0 | Canonical DeepONet | `python main.py --model m0 --data-root D --out runs/m0` | inner-product coupling, raw coordinates |
| M1 | M0 + adaptive dual-path combiner | `python main.py --model m1 ...` | learnable inner-product/MLP blend |
| M2 | M1 + signed building-proximity feature | `python main.py --model m2 ...` | d_BP trunk feature |
| M3 | M2 + multiscale Fourier features | `python main.py --model m3 ...` | Fourier embedding of coordinates |
| M4 | M3 + surface-category embedding (primary data-driven model) | `python main.py --model m4 ...` | learnable 5-category embedding |
| M5 | Physics-consistent extension of M4 | `python main.py --model m5 ... --physics-operator R` | solar branch re-encoding + fixed-coefficient, ray-resolved transient SEB regularization |

Supplementary controls (not part of the M0-M5 progression):

| Label | Description | Training command |
|---|---|---|
| D1 | inner-product coupling + d_BP | `python main.py --model d1 ...` |
| Solar control | M4 architecture with the solar branch, no SEB loss (verifies that the hour attribution is not an encoding artifact; identical architecture to M5 - only the loss differs) | `python main.py --model solar_control ...` |

Supplementary feature-enriched baselines: `baselines/train_mlp_enriched.py` (268-D flattened M4 inputs, parameter-matched), `baselines/train_lightgbm_enriched.py` (8-D: plain inputs + d_BP + category), and `baselines/train_lightgbm_fourier.py` (264-D: plain inputs + d_BP + the M3/M4 multiscale Fourier features + category). The matched M4 short run uses the same 200-epoch protocol (`python main.py --model m4 --epochs 200 ...`). 

Notes:
- Solar-branch inputs are `[RH, T_amb, solar altitude, solar azimuth,
  BHI]`, where BHI is the beam horizontal irradiance. DNI is recovered
  as BHI / sin(altitude) and used once, with the incidence-angle
  projection, inside the SEB shortwave-flux computation only
  (`urbanheat/solar.py`; self-test: `python -m urbanheat.solar`).
- The supervised temperature loss uses all surface categories. The SEB
  penalty acts on Building receivers only because that category has verified
  STL facet normals and does not require the latent-heat terms used for Green
  and Water.
- M5 area-samples Building receivers independently of ray confidence. The
  primary confidence setting continuously weights the SEB residual using
  target-independent ray-resolution, coarse-hit, category-consistency,
  normal and coordinate-mapping QC. `--physics-weighting uniform` evaluates
  the matched sensitivity control on the same receiver population.
- Surrounding longwave radiance is assembled from ray-hit emitter
  temperatures predicted by the same network. CFD target temperatures never
  enter the physics loss. Sky rays have zero incoming longwave, consistent
  with the completely transparent upper radiative boundary in the CFD.
- Direct shortwave reproduces the diagnostic convention: incidence and the
  inherited shadow mask use the verified public shortwave normal stored with
  the shadow artifact. The closest-STL surface normal is stored separately
  and is used for the ray hemisphere and roof/wall classification.
- The SEB closure settings (h_roof = h_wall = 2 W/m2/K,
  C_A = 42,336 J/m2/K, G0_roof = 132.1 W/m2, G0_wall = 157.9 W/m2,
  R0 = 76 W/m2) and the loss weight beta = 0.075 are fixed constants,
  not fitted neural-network parameters. They are the defaults in
  `urbanheat/config.py`, and every run records the values it used in
  `run_config.json`.
- The internal storage key `'sdf'` denotes the signed building-proximity
  feature d_BP.

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

# generate the synthetic dummy dataset in a new directory
python scripts/make_dummy_data.py --out dummy_generated

# train + evaluate one model (1 epoch, dummy data)
python main.py --model m4 --data-root dummy_generated \
    --out runs/m4 --epochs 1

# M5 end-to-end schema/gradient smoke test on synthetic data
python main.py --model m5 --data-root dummy_generated \
    --out runs/m5_smoke \
    --physics-operator dummy_generated/physics --epochs 1 \
    --points-per-case 1000 --physics-points 16

# full training on the real dataset
python main.py --model m4 --data-root /path/to/data --out runs/m4

# prepare target-independent ray metadata, then train M5
python scripts/prepare_physics_operator.py \
    --manifest /path/to/rays/ray_manifest_TAG.json \
    --data-root /path/to/data
python main.py --model m5 --data-root /path/to/data --out runs/m5 \
    --physics-operator /path/to/rays/ray_manifest_TAG.json
```

Each DeepONet run writes `best_model.pt`, `last_model.pt`, `scalers.pkl`,
`run_config.json`, `training_log.txt`, `training_history.pkl` and
`evaluation_results.pkl` into its own `--out` directory. An interrupted run
resumes from the last completed epoch with `--resume`; model, optimizer,
scheduler, AMP, sampling and RNG state are restored together.
For an intentionally interruptible run, predeclare a fixed cosine horizon
with `--schedule-epochs`; a resumed `--epochs` value may increase only up to
that unchanged horizon.

Analysis (each script loads trained runs from `runs/`):

```bash
python analysis/ood_extrapolation.py --data-root data/dummy \
    --ood-dir /path/to/ood_cases --runs runs --models m0 m1 m2 m3 m4 m5 \
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
    --run-a runs/solar_control --model-a solar_control \
    --run-b runs/m5 --model-b m5 \
    --physics-operator /path/to/rays/ray_manifest_TAG.json \
    --out analysis_out/seb_residual
python analysis/mechanism_bands_spectrum.py --data-root data/dummy \
    --run-a runs/m1 --model-a m1 --run-b runs/m4 --model-b m4 \
    --out analysis_out/mechanism
python analysis/branch_latent_pca.py --data-root data/dummy \
    --run runs/m4 --model m4 --out analysis_out/branch_pca
python -m analysis.alpha_param_table --runs runs
```

Minimal inference example (single case, no training):

```bash
python inference_example.py --run runs/m4 --model m4 \
    --case data/dummy/cases/Case_60_35_12_XYZInternalTable.csv \
    --data-root data/dummy --out predictions_m4.npz
```

Pretrained weights for the paper's models will be provided as release
assets of this repository (best_model.pt + scalers.pkl per model);
place them under runs/<model>/ to use the analysis and inference
scripts without retraining.

## Repository layout

```
main.py                     training / evaluation entry point
inference_example.py        minimal single-case inference
urbanheat/                  shared library
  config.py                 fixed experiment constants
  geometry.py               d_BP feature, categories, SEB geometry
  solar.py                  solar table, radiative properties, shadows
  physics_operator.py       ray geometry, QC sampling and physics minibatches
  seb.py                    fixed-coefficient ray-resolved SEB loss
  data.py                   loading, split, scalers, dataset
  models.py                 all DeepONet variants
  engine.py                 training / evaluation loops
baselines/                  MLP-vanilla, Random Forest and LightGBM baselines
analysis/                   OOD, IG, embeddings, facets, morphology, SEB
utils/dbp_feature.py        standalone d_BP reference implementation
scripts/make_dummy_data.py  synthetic data generator
scripts/prepare_physics_operator.py  target-independent ray QC metadata
```

## Notes on reproducibility

The train/test split is condition-based (seed 42): all cases sharing an
(RH, T_amb) combination fall on the same side, so no meteorological
condition leaks between the two sets. Global weight-init/sampling seed
is `--seed` (default 42) for the DeepONet variants. Exception:
MLP-vanilla follows its original run, which fixed the numpy seed (42,
data split/sampling) but not the torch weight-initialization seed. The integrated-gradients protocol uses 30
steps, 5,000 points per case, 40 cases and seed 0, matching the paper.
Each physics metadata file records and verifies SHA-256 hashes for its
manifest, ray-ID, hit-category and mapped-shadow artifacts. Optimizer,
learning-rate schedule, sampling and early-stopping settings
are defined in `urbanheat/config.py` and `urbanheat/engine.py`.

The experiments in the paper were run with Python 3.8, PyTorch 2.4.1
(CUDA 11.8) on a single NVIDIA L40S (48 GB); `requirements.txt` records
the core package versions used. Full training of the data-driven models
takes roughly 20 h each on this GPU (M5 takes longer due to the
adjacent-hour SEB evaluations); the dummy dataset runs in minutes on CPU.

## License

MIT - see `LICENSE`.
