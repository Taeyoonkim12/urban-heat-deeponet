"""Experiment constants shared by all models.

These values are fixed to match the paper. Changing them will break
comparability with the reported results.
"""

# Analysis domain (m). Points outside are discarded.
GLOBAL_COORD_RANGES = {
    'x_min': 1000.0, 'x_max': 3000.0,
    'y_min': -3000.0, 'y_max': -1000.0,
    'z_min': -15.0, 'z_max': 260.0,
}

# Surface temperature filter (deg C).
TEMP_LIMITS = {'t_min': 20.0, 't_max': 70.0}

# Train/test split: 20% of the (RH, T_amb) combinations are held out.
SPLIT_SEED = 42
TEST_FRACTION = 0.2

# Network dimensions.
LATENT_DIM = 256
HIDDEN_DIM = 512
DEPTH = 6
DROPOUT = 0.1
NUM_FREQUENCIES = 128
FOURIER_SEED = 42

# Surface categories (order defines the embedding index).
CATEGORY_NAMES = ['Building', 'Road', 'Green', 'Water', 'Topo_IN']
N_CATEGORIES = 5
CAT_EMBED_DIM = 8

# Training protocol.
POINTS_PER_CASE = 100_000
CHUNK_SIZE = 50_000
TOTAL_EPOCHS = 800
LR_INIT = 5e-4
LR_MIN = 1e-6
WARMUP_EPOCHS = 20
PATIENCE = 120
WEIGHT_DECAY = 1e-4
SCALER_SAMPLES = 200_000

# SEB physics loss (M5 family). Green and Water are excluded from the
# residual: latent-heat / heat-capacity effects make the simple SEB invalid.
BETA_PHYS = 0.1
RESID_SCALE = 100.0        # residual normalisation scale (W/m^2)
W_PHYS_CAT = [1.0, 1.0, 0.0, 0.0, 1.0]
