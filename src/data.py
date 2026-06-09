# src/data.py
# Data loading, batch utilities, class weight computation.
# All functions take explicit paths — no hardcoded directories.

import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

from .config import (
    PATCHES, EXCLUDE_FROM_TRAIN,
    TRAIN_END, VAL_END, TEST_END,
    PRIMARY_RADIUS_KM, PRIMARY_MW_THRESH,
)


# Graph and target loading

def load_graphs(graph_dir):
    """
    Load all patch PyG graph objects.
    Returns dict: patch_name → Data object.
    """
    graph_dir = Path(graph_dir)
    graphs    = {}

    for patch in PATCHES:
        path = graph_dir / f"{patch}_graph.pt"
        if path.exists():
            graphs[patch] = torch.load(path, weights_only=False)
        else:
            print(f"  WARNING: graph not found — {path}")

    print(f"Loaded {len(graphs)}/{len(PATCHES)} graphs")
    return graphs


def load_targets(target_dir, radius_km=None, mw_thresh=None):
    """
    Load binary target tensors for one target definition.
    Defaults to primary target (50km, Mw≥3.0).

    Returns dict: patch_name → tensor (n_lat, n_lon, 300)
    """
    target_dir = Path(target_dir)
    r          = radius_km or PRIMARY_RADIUS_KM
    mw         = mw_thresh or PRIMARY_MW_THRESH
    targets    = {}

    for patch in PATCHES:
        path = target_dir / f"{patch}_target_r{r}_mw{mw}.npy"
        if path.exists():
            targets[patch] = torch.tensor(
                np.load(path), dtype=torch.float32
            )
        else:
            print(f"  WARNING: target not found — {path}")

    print(f"Loaded {len(targets)}/{len(PATCHES)} targets "
          f"(r={r}km, Mw≥{mw})")
    return targets


def load_all_targets(target_dir, radii=None, mw_threshs=None):
    """
    Load all 20 target definitions.
    Returns dict: (radius_km, mw_thresh) → {patch: tensor}
    """
    from .config import TARGET_RADII_KM, TARGET_MW_THRESHS
    radii      = radii      or TARGET_RADII_KM
    mw_threshs = mw_threshs or TARGET_MW_THRESHS
    all_targets = {}

    for r in radii:
        for mw in mw_threshs:
            all_targets[(r, mw)] = load_targets(target_dir, r, mw)

    print(f"Loaded {len(all_targets)} target definitions")
    return all_targets


def load_all_data(data_dir, radius_km=None, mw_thresh=None):
    """
    Convenience wrapper — load graphs and primary targets together.
    Returns (graphs, targets).
    """
    data_dir = Path(data_dir)
    graphs   = load_graphs(data_dir / "graphs")
    targets  = load_targets(
        data_dir / "targets",
        radius_km=radius_km,
        mw_thresh=mw_thresh,
    )
    return graphs, targets


# Class weight computation 

def compute_class_weights(graphs, targets, max_weight=20.0):
    """
    Per-patch inverse frequency class weights capped at max_weight.
    Computed from training period only.

    Returns dict: patch_name → (w_neg, w_pos)
    """
    weights = {}

    for patch in PATCHES:
        if patch not in targets or patch not in graphs:
            continue

        data    = graphs[patch]
        n_lat   = data.grid_shape[0].item()
        n_lon   = data.grid_shape[1].item()

        # Flatten to (n_cells, 300), select valid cells, training period
        t_flat  = targets[patch].reshape(n_lat * n_lon, 300)
        t_valid = t_flat[data.valid_idx]
        t_train = t_valid[:, :TRAIN_END].flatten()

        pos = t_train.mean().item()
        if 0 < pos < 1:
            w_pos = min(1.0 / (2 * pos),       max_weight)
            w_neg = min(1.0 / (2 * (1 - pos)), max_weight)
        else:
            w_pos, w_neg = 1.0, 1.0

        weights[patch] = (w_neg, w_pos)

    return weights


def print_class_weights(class_weights):
    """Pretty-print class weight table."""
    print(f"\n{'Patch':<25} {'pos_rate':>9} {'w_neg':>7} "
          f"{'w_pos':>7} {'status':>10}")
    print("-" * 62)
    for patch, (w_neg, w_pos) in class_weights.items():
        # Recover approximate pos_rate from weights
        pos_rate = 1.0 / (2 * w_pos) if w_pos < 20 else "< 0.025"
        excl     = "EXCLUDED" if patch in EXCLUDE_FROM_TRAIN else ""
        if isinstance(pos_rate, float):
            print(f"  {patch:<23} {pos_rate:>9.3f} {w_neg:>7.2f} "
                  f"{w_pos:>7.2f} {excl:>10}")
        else:
            print(f"  {patch:<23} {pos_rate:>9} {w_neg:>7.2f} "
                  f"{w_pos:>7.2f} {excl:>10}")


# Batch utilities

def get_batch_targets(patch, graphs, targets, t_start, t_end, device):
    """
    Extract valid-cell targets for a specific time window.
    Targets stay on CPU until the final slice — avoids device mismatch
    with valid_idx which is stored on CPU in the graph object.
    """
    data    = graphs[patch] # graph stays on CPU for indexing
    n_lat   = data.grid_shape[0].item()
    n_lon   = data.grid_shape[1].item()

    # Keep on CPU for indexing with valid_idx
    t_flat  = targets[patch].reshape(n_lat * n_lon, 300)
    t_valid = t_flat[data.valid_idx.cpu()]   # explicit .cpu()

    # Only move the final window slice to device
    return t_valid[:, t_start:t_end].to(device)


def get_cluster_labels(patch, graphs, device):
    """
    Get dominant GMM cluster label per valid cell.
    """
    return graphs[patch].x_prior.cpu().argmax(dim=-1).to(device)


def get_sample_weights(targets_batch, w_neg, w_pos):
    """
    Per-sample focal loss weights from class weights.

    targets_batch: (N, T) binary targets
    Returns:       (N*T,) flattened weights
    """
    return (
        targets_batch * w_pos + (1 - targets_batch) * w_neg
    ).flatten()


# Metrics

def compute_auc_roc(probs, targets):
    """
    AUC-ROC — returns NaN if only one class present.
    Inputs can be tensors or numpy arrays.
    """
    p = _to_numpy(probs)
    t = _to_numpy(targets).astype(int)
    if len(np.unique(t)) < 2:
        return float('nan')
    try:
        return float(roc_auc_score(t, p))
    except Exception:
        return float('nan')


def compute_auc_pr(probs, targets):
    """
    AUC-PR (average precision) — more informative than AUC-ROC
    for imbalanced targets. Returns NaN if only one class.
    """
    p = _to_numpy(probs)
    t = _to_numpy(targets).astype(int)
    if len(np.unique(t)) < 2 or t.sum() == 0:
        return float('nan')
    try:
        return float(average_precision_score(t, p))
    except Exception:
        return float('nan')


def compute_brier_score(probs, targets):
    """Brier score — proper scoring rule measuring calibration."""
    p = _to_numpy(probs)
    t = _to_numpy(targets).astype(float)
    return float(np.mean((p - t) ** 2))


def compute_brier_skill_score(probs, targets):
    """
    Brier Skill Score = 1 - BS / BS_climatology.
    BSS > 0: beats climatology. BSS = 1: perfect.
    """
    t        = _to_numpy(targets).astype(float)
    bs       = compute_brier_score(probs, targets)
    bs_clim  = float(np.mean((t.mean() - t) ** 2))
    if bs_clim == 0:
        return float('nan')
    return float(1 - bs / bs_clim)


def compute_all_metrics(probs, targets):
    """
    Compute full metric suite for one patch/period.
    Returns dict of all metrics.
    """
    p_flat = _to_numpy(probs).ravel()
    t_flat = _to_numpy(targets).ravel().astype(int)

    return {
        'auc_roc': compute_auc_roc(p_flat, t_flat),
        'auc_pr':  compute_auc_pr(p_flat, t_flat),
        'brier':   compute_brier_score(p_flat, t_flat),
        'bss':     compute_brier_skill_score(p_flat, t_flat),
        'pos_rate': float(t_flat.mean()),
        'mean_prob': float(p_flat.mean()),
        'n_cells_steps': int(len(t_flat)),
    }


def _to_numpy(x):
    """Convert tensor or array to numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# Prior builder (leakage-free per MCPC cycle)

def build_prior_for_cycle(R_patches, feature_dir, cont_idx, k=6, n_pca=4):
    """
    Build leakage-free frozen prior for one MCPC cycle.
    Fits normalisation, PCA, and GMM exclusively on R patches.

    Parameters
    ----------
    R_patches   : list of patch name strings (source patches only)
    feature_dir : Path to feature tensor directory
    cont_idx    : indices of continuous features in feature tensor
    k           : number of GMM clusters (fixed at 6)
    n_pca       : number of PCA components (fixed at 4)

    Returns
    -------
    get_prior : callable(patch_name) → (n_lat, n_lon, k) prob array
    gmm       : fitted GaussianMixture
    pca       : fitted PCA
    R_mean    : normalisation mean (from R only)
    R_std     : normalisation std  (from R only)
    """
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture

    feature_dir = Path(feature_dir)

    # Step 1 — collect R patch feature vectors
    X_R = []
    for patch in R_patches:
        t = np.load(feature_dir / f"{patch}_features.npy")
        n_lat, n_lon, _ = t.shape
        for i in range(n_lat):
            for j in range(n_lon):
                vec = t[i, j, cont_idx]
                if not np.any(np.isnan(vec)):
                    X_R.append(vec)
    X_R = np.array(X_R)

    # Step 2 — normalise on R only
    R_mean = X_R.mean(axis=0)
    R_std  = X_R.std(axis=0) + 1e-8
    X_R_norm = (X_R - R_mean) / R_std

    # Step 3 — PCA on R only
    pca      = PCA(n_components=n_pca, random_state=42)
    X_R_pca  = pca.fit_transform(X_R_norm)

    # Step 4 — GMM on R only
    gmm = GaussianMixture(
        n_components=k, covariance_type='full',
        random_state=42, n_init=5, max_iter=200
    )
    gmm.fit(X_R_pca)

    # Step 5 — return transform function for any patch
    def get_prior(patch_name):
        t    = np.load(feature_dir / f"{patch_name}_features.npy")
        lons = np.load(feature_dir / f"{patch_name}_lons.npy")
        lats = np.load(feature_dir / f"{patch_name}_lats.npy")
        n_lat, n_lon, _ = t.shape

        prob_map = np.full((n_lat, n_lon, k), np.nan)
        vecs, idx = [], []

        for i in range(n_lat):
            for j in range(n_lon):
                vec = t[i, j, cont_idx]
                if not np.any(np.isnan(vec)):
                    vecs.append(vec)
                    idx.append((i, j))

        if len(vecs) == 0:
            return prob_map

        Xp      = np.array(vecs)
        Xp_norm = (Xp - R_mean) / R_std
        Xp_pca  = pca.transform(Xp_norm)
        probs   = gmm.predict_proba(Xp_pca)

        for k_idx, (i, j) in enumerate(idx):
            prob_map[i, j, :] = probs[k_idx]

        return prob_map

    return get_prior, gmm, pca, R_mean, R_std