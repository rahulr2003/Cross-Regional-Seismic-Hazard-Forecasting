# src/config.py
# Central configuration for all hyperparameters, constants, and patch definitions.
# All other modules import from here — never hardcode values elsewhere.

# ── Patch definitions ─────────────────────────────────────────────────────────

PATCHES = [
    "Kanto_Japan",
    "Tohoku_Japan",
    "Central_Chile",
    "Central_Turkey",
    "Central_Nepal",
    "North_Island_NZ",
    "Sumatra",
    "Kutch_India",
    "Sichuan_China",
    "W_Australia",
    "S_Norway",
    "Ordos_China",
]

# Excluded from training gradient — too sparse for useful signal
# Still used in validation and MCPC evaluation
EXCLUDE_FROM_TRAIN = {"S_Norway", "W_Australia"}

# Patch bounding boxes (WGS84, for reference)
PATCH_BOUNDS = {
    "Kanto_Japan":     dict(minlat=34.5,  maxlat=37.2,  minlon=138.5, maxlon=141.5),
    "Tohoku_Japan":    dict(minlat=37.5,  maxlat=40.5,  minlon=140.5, maxlon=143.5),
    "Central_Chile":   dict(minlat=-36.5, maxlat=-33.5, minlon=-72.5, maxlon=-69.5),
    "Central_Turkey":  dict(minlat=36.5,  maxlat=39.0,  minlon=35.5,  maxlon=39.0),
    "Central_Nepal":   dict(minlat=27.0,  maxlat=29.7,  minlon=83.5,  maxlon=86.5),
    "North_Island_NZ": dict(minlat=-40.5, maxlat=-37.5, minlon=174.5, maxlon=178.0),
    "Sumatra":         dict(minlat=-5.5,  maxlat=-2.0,  minlon=100.5, maxlon=104.5),
    "Kutch_India":     dict(minlat=21.5,  maxlat=24.5,  minlon=68.5,  maxlon=72.0),
    "Sichuan_China":   dict(minlat=29.5,  maxlat=32.5,  minlon=102.0, maxlon=105.5),
    "W_Australia":     dict(minlat=-32.0, maxlat=-29.0, minlon=117.0, maxlon=120.5),
    "S_Norway":        dict(minlat=58.5,  maxlat=61.5,  minlon=5.0,   maxlon=9.0),
    "Ordos_China":     dict(minlat=37.0,  maxlat=40.0,  minlon=107.5, maxlon=111.0),
}

# Tectonic regime label per patch (for evaluation and reporting)
PATCH_REGIMES = {
    "Kanto_Japan":     "Subduction",
    "Tohoku_Japan":    "Subduction",
    "Central_Chile":   "Megathrust",
    "Central_Turkey":  "Strike-slip",
    "Central_Nepal":   "Collision",
    "North_Island_NZ": "Subduction",
    "Sumatra":         "Subduction",
    "Kutch_India":     "Intraplate",
    "Sichuan_China":   "Collision",
    "W_Australia":     "Craton",
    "S_Norway":        "Post-glacial",
    "Ordos_China":     "Craton",
}

# ── Temporal split ────────────────────────────────────────────────────────────
# Monthly time steps, Jan 2000 = step 0

TRAIN_END = 228   # Jan 2000 – Dec 2018  (228 steps)
VAL_END   = 264   # Jan 2019 – Dec 2021  (36 steps)
TEST_END  = 300   # Jan 2022 – Dec 2024  (36 steps)

# ── Feature dimensions ────────────────────────────────────────────────────────

STATIC_DIM   = 12   # static geological features per cell
PRIOR_DIM    = 6    # GMM cluster probabilities (k=6)
TEMPORAL_DIM = 17   # temporal features per cell per time step

STATIC_FEATURE_NAMES = [
    'vs30', 'elevation', 'slope', 'roughness',
    'sediment_km', 'crustal_km', 'dist_fault_km',
    'fault_density', 'fault_slip', 'heat_flow',
    'stress_azi', 'stress_regime',
]

TEMPORAL_FEATURE_NAMES = [
    'count_30d', 'count_90d', 'count_180d', 'count_365d',
    'count_m4_180d', 'mean_mag_90d', 'max_mag_90d', 'max_mag_365d',
    'mean_iet_90d', 'cv_iet_90d',
    'time_since_m3', 'time_since_m4',
    'bvalue_180d', 'omori_K', 'omori_p',
    'neighbour_count_30d', 'neighbour_max_mag_90d',
]

# ── Target variable ───────────────────────────────────────────────────────────

# Primary target used for training and main evaluation
PRIMARY_RADIUS_KM = 50
PRIMARY_MW_THRESH = 3.0

# Full grid for sensitivity analysis
TARGET_RADII_KM   = [10, 30, 50, 70, 100]
TARGET_MW_THRESHS = [3.0, 3.5, 4.0, 4.5]

TARGET = dict(
    radius_km  = 50,
    mw_thresh  = 3.0,
)

# ── Architecture ──────────────────────────────────────────────────────────────

ARCH = dict(
    static_dim     = STATIC_DIM,
    prior_dim      = PRIOR_DIM,
    temporal_dim   = TEMPORAL_DIM,
    geo_embed_dim  = 32,    # geological encoder output dimension
    hidden_dim     = 128,   # GNN hidden dimension
    node_embed_dim = 64,    # backbone output dimension
    n_heads        = 4,     # Graph Transformer attention heads
    dropout        = 0.1,
)

# ── Training ──────────────────────────────────────────────────────────────────

TRAIN = dict( # new training dictionary
    n_epochs      = 200,
    lr            = 3e-4,
    weight_decay  = 1e-4,
    patience      = 30,
    batch_t_size  = 48,
    lambda_adv    = 0.1,
    lambda_con    = 2.0,    # much higher — cluster alignment loss
                            # is smaller in magnitude than focal
    alpha_max     = 1.0,
    clip_grad     = 1.0,
    max_weight    = 20.0,
    n_seeds       = 3,
    phase1_epochs = 60,     # epochs of con+adv only before focal
)

# ── GMM frozen prior ──────────────────────────────────────────────────────────

GMM = dict(
    k                = 6,       # number of clusters
    n_pca_components = 4,       # PCA components before GMM
    covariance_type  = 'full',  # GMM covariance type
    n_init           = 5,       # GMM random initialisations
)

# ── Graph construction ────────────────────────────────────────────────────────

GRAPH = dict(
    k_spatial          = 8,     # geographic nearest neighbours
    corr_threshold     = 0.3,   # Pearson r for temporal edges
    corr_feature_idx   = 0,     # feature index for correlation (count_30d)
    grid_resolution    = 0.1,   # degrees
)

# ── MCPC evaluation ───────────────────────────────────────────────────────────

MCPC = dict(
    n_cycles           = 500,   # Monte Carlo cycles
    n_source_patches   = 9,     # R patches per cycle
    n_finetune_patches = 1,     # F patches per cycle
    n_predict_patches  = 2,     # P patches per cycle
    min_cross_regime   = 0.4,   # fraction of cross-regime (F,P) pairs
    min_low_seismicity = 0.2,   # fraction of cycles with low-seis P
    finetune_epochs    = 50,
    finetune_patience  = 10,
    finetune_lr        = 1e-4,
)