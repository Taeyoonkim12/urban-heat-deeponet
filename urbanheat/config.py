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

# SEB physics loss: applied to BUILDING surfaces only.
# Building is the only category with a verified facet orientation (STL
# face normals), required for the incidence-angle term of the direct
# shortwave flux. All other categories are excluded:
#   Road    - facet orientation not verified (may follow sloped terrain)
#   Green   - latent-heat effects invalidate the simple SEB
#   Water   - different thermal formulation (heat capacity, evaporation)
#   Topo_IN - sloped terrain approximated as horizontal (0,0,1)
BETA_PHYS = 0.075
RESID_SCALE = 100.0        # SEB loss normalisation Q0 (W/m^2)
PHYSICS_POINTS = 256       # area-sampled Building receivers per CFD case
PHYSICS_WEIGHTING = 'confidence'
PHYSICS_SEED = 2718

# Fixed SEB closure settings. The effective convective coefficients, the
# effective areal heat capacity C_A = rho * c_p * delta_eff (concrete:
# 2400 kg/m3, 882 J/kg/K, effective thermal depth 0.02 m), the roof/wall
# closure fluxes G0 and the no-penalty threshold R0 were determined from
# the training-condition CFD temperature trajectories before network
# training and held fixed thereafter.
H_CONV_ROOF = 2.0          # W/m^2/K
H_CONV_WALL = 2.0          # W/m^2/K
ROOF_NORMAL_Z = 0.7
C_AREAL = 2400.0 * 882.0 * 0.02   # J/m2/K
G0_ROOF = 132.1            # W/m^2
G0_WALL = 157.9            # W/m^2
RESIDUAL_FLOOR = 76.0      # R0 (W/m^2)

# Ray-operator sentinel values. Sky directions have zero incoming longwave,
# matching the completely transparent upper radiative boundary in the CFD.
RAY_SKY = -1
RAY_UNRESOLVED = -2
