"""
Adaptive KD-Tree ST-KNN (Ad-KNN-ST) implementation.
This version integrates the "Multiple KD-Tree Superset Filter" approach to
reduce the distance calculation complexity from O(N) to O(log N).
"""

import numpy as np
from scipy.spatial import cKDTree
from dataclasses import dataclass
import time

# Data container
# ---------------------------------------------------------------------------
@dataclass
class STData:
    xy: np.ndarray      # (N,2) normalized spatial coords
    t: np.ndarray       # (N,)  normalized time coords
    v: np.ndarray       # (N,)  values


def spatiotemporal_dist(xy, t, xy_q, t_q, alpha):
    """Exact d_ST(x,p) = alpha*dS + (1-alpha)*dT"""
    if xy.ndim == 1:
        xy = xy.reshape(1, 2)
        t = np.array([t])
    ds = np.sqrt(((xy - xy_q) ** 2).sum(axis=1))
    dt = np.abs(t - t_q)
    return alpha * ds + (1 - alpha) * dt


def build_alpha_trees(obs: STData, alphas):
    """
    Builds a dictionary of 3D cKDTrees, one for each unique alpha value.
    The points are pre-scaled so that the KDTree's L1 norm acts as a bounding metric
    for the custom spatiotemporal distance.
    """
    trees = {}
    for a in alphas:
        scaled_pts = np.zeros((len(obs.v), 3))
        scaled_pts[:, 0] = obs.xy[:, 0] * a
        scaled_pts[:, 1] = obs.xy[:, 1] * a
        scaled_pts[:, 2] = obs.t * (1 - a)
        trees[a] = cKDTree(scaled_pts)
    return trees


# ---------------------------------------------------------------------------
# Fast Prediction using KD-Tree Superset Filter
# ---------------------------------------------------------------------------
def _predict_one_fast(cand, obs: STData, x_xy, x_t, trees_dict, eps=1e-3, k_multiplier=5):
    """
    O(log N) prediction using the KDTree for superset filtering.
    """
    alpha, k = cand["alpha"], cand["k"]
    tree = trees_dict[alpha]
    
    # Scale query point to match tree
    q_3d = np.array([x_xy[0] * alpha, x_xy[1] * alpha, x_t * (1 - alpha)])
    
    # Fetch a superset of candidates using L1 norm (p=1)
    k_fetch = min(k * k_multiplier, len(obs.v))
    # query returns (distances, indices)
    _, idx_fast = tree.query(q_3d, k=k_fetch, p=1)
    
    # Ensure idx_fast is an array (tree.query returns a scalar if k=1 and N=1)
    if np.isscalar(idx_fast):
        idx_fast = np.array([idx_fast])
        
    # Calculate EXACT custom distance ONLY for the superset points
    exact_d = spatiotemporal_dist(obs.xy[idx_fast], obs.t[idx_fast], x_xy, x_t, alpha)
    
    # Sort and pick the true top K
    k_final = min(k, len(exact_d))
    idx_sort = np.argsort(exact_d)[:k_final]
    best_idx = idx_fast[idx_sort]
    best_d = exact_d[idx_sort]
    
    # Apply weighting
    if cand["kind"] == "gauss":
        w = np.exp(-(best_d ** 2) / (2 * cand["sigma"] ** 2)) + 1e-9
    else:
        w = 1.0 / (best_d + eps)
        
    return float(np.sum(w * obs.v[best_idx]) / np.sum(w))


def _ne(actual, pred):
    actual, pred = np.asarray(actual), np.asarray(pred)
    denom = np.sum(np.abs(actual))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(actual - pred)) / denom)


def _validate_candidates_fast(cands, obs: STData, trees_dict, n_splits=3, val_frac=0.15, seed=0):
    """Validation engine optimized with the fast prediction function."""
    rng = np.random.default_rng(seed)
    n = len(obs.v)
    scores = np.zeros(len(cands))
    n_val = max(3, int(n * val_frac))
    for s in range(n_splits):
        val_idx = rng.choice(n, size=n_val, replace=False)
        mask = np.ones(n, dtype=bool)
        mask[val_idx] = False
        train = STData(obs.xy[mask], obs.t[mask], obs.v[mask])
        if len(train.v) < 2:
            continue
            
        # We must rebuild the trees for the validation training set!
        # This takes O(Alphas * N log N), but is done very few times.
        train_alphas = set(c["alpha"] for c in cands)
        train_trees = build_alpha_trees(train, train_alphas)
        
        for ci, cand in enumerate(cands):
            preds = [_predict_one_fast(cand, train, obs.xy[i], obs.t[i], train_trees) for i in val_idx]
            scores[ci] += _ne(obs.v[val_idx], preds)
    return scores / n_splits


# ---------------------------------------------------------------------------
# Adaptive KD-Tree ST-KNN (Ad-KNN-ST)
# ---------------------------------------------------------------------------
ALPHAS = [0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85]
KS = [2, 3, 4, 5, 6, 8]
SIGMAS = [0.2, 0.3, 0.5]

def build_candidates():
    cands = []
    cands.append(dict(kind="orig", alpha=0.5, k=4))
    for a in ALPHAS:
        for k in KS:
            cands.append(dict(kind="tuned", alpha=a, k=k))
    for a in [0.35, 0.5, 0.65]:
        for k in [4, 5, 6]:
            for s in SIGMAS:
                cands.append(dict(kind="gauss", alpha=a, k=k, sigma=s))
    return cands

CANDIDATES = build_candidates()
SHORTLIST = [
    dict(kind="orig", alpha=0.5, k=4),
    dict(kind="tuned", alpha=0.35, k=4),
    dict(kind="tuned", alpha=0.5, k=6),
    dict(kind="gauss", alpha=0.5, k=5, sigma=0.3),
    dict(kind="gauss", alpha=0.35, k=6, sigma=0.3),
]


def local_density(tree: cKDTree, query_pts, radius):
    counts = tree.query_ball_point(query_pts, r=radius, return_length=True)
    return np.asarray(counts)

def adaptive_sigma(rho, sigma0=0.3, rho_ref=6.0, gamma=0.5, smin=0.15, smax=0.9):
    return np.clip(sigma0 * (rho_ref / (rho + 1)) ** gamma, smin, smax)

def adaptive_lambda(rho, lam0=0.7, delta=0.35, rho_max=10.0, lmin=0.35, lmax=0.85):
    return np.clip(lam0 - delta * (1 - np.minimum(rho, rho_max) / rho_max), lmin, lmax)


def ad_knn_st_predict(obs: STData, miss_xy, miss_t, missing_rate,
                       density_radius=0.15, density_threshold=6,
                       n_top=5, seed=0):
    """
    Optimized Ad-KNN-ST using KDTree Superset Filtering.
    """
    # 1. Build standard 2D spatial tree for density checking
    spatial_tree = cKDTree(obs.xy)
    n_eval_total = 0

    rho = local_density(spatial_tree, miss_xy, density_radius)
    sigma_x = adaptive_sigma(rho)
    lambda_x = adaptive_lambda(rho)

    high_density_mask = rho >= density_threshold
    final = np.zeros(len(miss_xy))

    # Build the specialized 3D alpha-trees for the main dataset ONLY once
    unique_alphas = set(c["alpha"] for c in CANDIDATES)
    main_trees = build_alpha_trees(obs, unique_alphas)

    # ---- high-density points: cheap shortlist, single pass ----
    if high_density_mask.any():
        idxs = np.where(high_density_mask)[0]
        scores = _validate_candidates_fast(SHORTLIST, obs, main_trees, n_splits=1, seed=seed)
        n_eval_total += len(SHORTLIST) * 1
        order = np.argsort(scores)[:min(n_top, len(SHORTLIST))]
        cands = [SHORTLIST[i] for i in order]
        inv = 1.0 / (scores[order] + 1e-6)
        w = inv / inv.sum()
        for i in idxs:
            preds = []
            for c in cands:
                if c["kind"] == "gauss":
                    c = dict(c, sigma=float(sigma_x[i]))
                preds.append(_predict_one_fast(c, obs, miss_xy[i], miss_t[i], main_trees))
            final[i] = float(np.dot(w, preds))

    # ---- low-density points: pruned full search + per-point two-pass ----
    if (~high_density_mask).any():
        idxs = np.where(~high_density_mask)[0]
        
        # Stage A: cheap 1-split pass over full pool
        stage_a_scores = _validate_candidates_fast(CANDIDATES, obs, main_trees, n_splits=1, seed=seed)
        n_eval_total += len(CANDIDATES) * 1
        shortlist_idx = np.argsort(stage_a_scores)[:20]
        
        # Stage B: full 3-split CV only on survivors
        survivors = [CANDIDATES[i] for i in shortlist_idx]
        stage_b_scores = _validate_candidates_fast(survivors, obs, main_trees, n_splits=3, seed=seed)
        n_eval_total += len(survivors) * 3
        order = np.argsort(stage_b_scores)[:n_top]
        top_cands = [survivors[i] for i in order]
        inv = 1.0 / (stage_b_scores[order] + 1e-6)
        w = inv / inv.sum()

        def ens_predict(o, xy, t, trees, sigma_override=None):
            preds = []
            for c in top_cands:
                if c["kind"] == "gauss" and sigma_override is not None:
                    c = dict(c, sigma=sigma_override)
                preds.append(_predict_one_fast(c, o, xy, t, trees))
            return float(np.dot(w, preds))

        pass1_vals = {}
        for i in idxs:
            pass1_vals[i] = ens_predict(obs, miss_xy[i], miss_t[i], main_trees, float(sigma_x[i]))
            final[i] = pass1_vals[i]

        # per-point two-pass
        k_min = 3
        need_two_pass = [i for i in idxs if rho[i] < k_min]
        if need_two_pass:
            aug_xy = np.vstack([obs.xy, miss_xy[list(pass1_vals.keys())]])
            aug_t = np.concatenate([obs.t, miss_t[list(pass1_vals.keys())]])
            aug_v = np.concatenate([obs.v, np.array(list(pass1_vals.values()))])
            aug = STData(aug_xy, aug_t, aug_v)
            
            # Rebuild trees for the augmented data
            aug_trees = build_alpha_trees(aug, unique_alphas)
            
            for i in need_two_pass:
                p2 = ens_predict(aug, miss_xy[i], miss_t[i], aug_trees, float(sigma_x[i]))
                final[i] = lambda_x[i] * pass1_vals[i] + (1 - lambda_x[i]) * p2

    return final, n_eval_total
