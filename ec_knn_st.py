"""
EC-KNN-ST implementation family:
  1. knn_st            - original baseline (Marchang & Tripathi), fixed alpha=0.5, k=4
  2. ec_knn_st_v2       - as described in the BTech report (Gaussian kernel,
                          weighted ensemble, iterative two-pass, all 109 candidates,
                          every time -- the expensive version)
  3. ec_knn_st_modified - the proposed efficiency + accuracy fix:
                          - density-gated candidate search (skip full 109 pool
                            when local density is high)
                          - two-stage validation pruning (cheap pass -> shortlist -> full CV)
                          - per-point (not global) two-pass trigger
                          - adaptive sigma(x), lambda(x) based on local density

All normalized-error (NE) numbers this script produces are computed the same
way as Eq. 5.1 in the report: NE = sum(|actual-pred|) / sum(|actual|).
"""

import numpy as np
from scipy.spatial import cKDTree
from dataclasses import dataclass
from typing import Optional
import time


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class STData:
    xy: np.ndarray      # (N,2) normalized spatial coords
    t: np.ndarray        # (N,)  normalized time coords
    v: np.ndarray         # (N,)  values


def spatiotemporal_dist(xy, t, xy_q, t_q, alpha):
    """Vectorized d_ST(x,p) = alpha*dS + (1-alpha)*dT   (Eq. 3.1)"""
    ds = np.sqrt(((xy - xy_q) ** 2).sum(axis=1))
    dt = np.abs(t - t_q)
    return alpha * ds + (1 - alpha) * dt


# ---------------------------------------------------------------------------
# 1. Original KNN-ST  (Eq. 3.1 / 3.2)
# ---------------------------------------------------------------------------
def knn_st_predict(obs: STData, x_xy, x_t, alpha=0.5, k=4, eps=1e-3):
    d = spatiotemporal_dist(obs.xy, obs.t, x_xy, x_t, alpha)
    idx = np.argsort(d)[:k]
    w = 1.0 / (d[idx] + eps)
    return float(np.sum(w * obs.v[idx]) / np.sum(w))


# ---------------------------------------------------------------------------
# 2. EC-KNN-ST v2  (as in the report: fixed sigma, fixed lambda, ALL candidates
#    ALWAYS evaluated -- this is the expensive baseline we are improving on)
# ---------------------------------------------------------------------------
ALPHAS = [0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85]
KS = [2, 3, 4, 5, 6, 8]
SIGMAS = [0.2, 0.3, 0.5]

def build_v2_candidates():
    """Return list of candidate dicts (~109-ish pool, condensed for tractability
    while preserving the same structure: tuned KNN-ST, Gaussian-kernel variants)."""
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

V2_CANDIDATES = build_v2_candidates()  # ~1+42+27 = 70 (condensed from 109, same spirit)


def _predict_one(cand, obs: STData, x_xy, x_t, eps=1e-3):
    alpha, k = cand["alpha"], cand["k"]
    d = spatiotemporal_dist(obs.xy, obs.t, x_xy, x_t, alpha)
    k = min(k, len(d))
    idx = np.argsort(d)[:k]
    if cand["kind"] == "gauss":
        w = np.exp(-(d[idx] ** 2) / (2 * cand["sigma"] ** 2)) + 1e-9
    else:
        w = 1.0 / (d[idx] + eps)
    return float(np.sum(w * obs.v[idx]) / np.sum(w))


def _ne(actual, pred):
    actual, pred = np.asarray(actual), np.asarray(pred)
    denom = np.sum(np.abs(actual))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(actual - pred)) / denom)


def _validate_candidates(cands, obs: STData, n_splits=3, val_frac=0.15, seed=0):
    """Stratified-ish random validation: hide val_frac of obs, score every candidate.
    Returns candidate NE scores averaged over n_splits."""
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
        for ci, cand in enumerate(cands):
            preds = [_predict_one(cand, train, obs.xy[i], obs.t[i]) for i in val_idx]
            scores[ci] += _ne(obs.v[val_idx], preds)
    return scores / n_splits


def ec_knn_st_v2_predict(obs: STData, miss_xy, miss_t, missing_rate,
                          n_top=5, lam=0.7, two_pass_threshold=0.5, seed=0):
    """Full v2: validate ALL candidates every time, ensemble top-5,
    global two-pass trigger if missing_rate >= 0.5. This is the EXPENSIVE version."""
    cands = V2_CANDIDATES
    scores = _validate_candidates(cands, obs, n_splits=3, seed=seed)
    order = np.argsort(scores)[:n_top]
    top_cands = [cands[i] for i in order]
    top_scores = scores[order]
    inv = 1.0 / (top_scores + 1e-6)
    weights = inv / inv.sum()

    def ensemble_predict(o, xy, t):
        preds = [_predict_one(c, o, xy, t) for c in top_cands]
        return float(np.dot(weights, preds))

    pass1 = np.array([ensemble_predict(obs, x, t) for x, t in zip(miss_xy, miss_t)])

    if missing_rate >= two_pass_threshold:
        aug_xy = np.vstack([obs.xy, miss_xy])
        aug_t = np.concatenate([obs.t, miss_t])
        aug_v = np.concatenate([obs.v, pass1])
        aug = STData(aug_xy, aug_t, aug_v)
        pass2 = np.array([ensemble_predict(aug, x, t) for x, t in zip(miss_xy, miss_t)])
        final = lam * pass1 + (1 - lam) * pass2
    else:
        final = pass1

    n_eval = len(cands) * 3  # candidate evaluations spent on validation
    return final, n_eval


# ---------------------------------------------------------------------------
# 3. Modified version: density-gated + validation-pruned + adaptive sigma/lambda
# ---------------------------------------------------------------------------
SHORTLIST = [  # cheap pre-tuned shortlist used when density is high (Point 1)
    dict(kind="orig", alpha=0.5, k=4),
    dict(kind="tuned", alpha=0.35, k=4),
    dict(kind="tuned", alpha=0.5, k=6),
    dict(kind="gauss", alpha=0.5, k=5, sigma=0.3),
    dict(kind="gauss", alpha=0.35, k=6, sigma=0.3),
]


def local_density(tree: cKDTree, obs_n, query_pts, radius):
    """rho(x): count of observed points within radius r0 of x (Point 1/3)."""
    counts = tree.query_ball_point(query_pts, r=radius, return_length=True)
    return np.asarray(counts)


def adaptive_sigma(rho, sigma0=0.3, rho_ref=6.0, gamma=0.5, smin=0.15, smax=0.9):
    return np.clip(sigma0 * (rho_ref / (rho + 1)) ** gamma, smin, smax)


def adaptive_lambda(rho, lam0=0.7, delta=0.35, rho_max=10.0, lmin=0.35, lmax=0.85):
    return np.clip(lam0 - delta * (1 - np.minimum(rho, rho_max) / rho_max), lmin, lmax)


def ec_knn_st_modified_predict(obs: STData, miss_xy, miss_t, missing_rate,
                                density_radius=0.15, density_threshold=6,
                                n_top=5, seed=0):
    """
    Point 1: density gate -> shortlist vs full pool
    Point 2: two-stage pruning inside the full-pool path
    Point 3: per-point two-pass trigger (based on local neighbour count, not
             the global missing rate)
    Point 4 (Gaussian kernel weight) uses adaptive sigma(x); blend uses
             adaptive lambda(x)
    """
    tree = cKDTree(obs.xy)
    n_eval_total = 0

    rho = local_density(tree, len(obs.v), miss_xy, density_radius)  # per-point density
    sigma_x = adaptive_sigma(rho)
    lambda_x = adaptive_lambda(rho)

    high_density_mask = rho >= density_threshold
    final = np.zeros(len(miss_xy))

    # ---- high-density points: cheap shortlist, single pass ----
    if high_density_mask.any():
        idxs = np.where(high_density_mask)[0]
        scores = _validate_candidates(SHORTLIST, obs, n_splits=1, seed=seed)
        n_eval_total += len(SHORTLIST) * 1
        order = np.argsort(scores)[:min(n_top, len(SHORTLIST))]
        cands = [SHORTLIST[i] for i in order]
        inv = 1.0 / (scores[order] + 1e-6)
        w = inv / inv.sum()
        for i in idxs:
            preds = []
            for c in cands:
                if c["kind"] == "gauss":
                    c = dict(c, sigma=float(sigma_x[i]))  # adaptive sigma injected
                preds.append(_predict_one(c, obs, miss_xy[i], miss_t[i]))
            final[i] = float(np.dot(w, preds))

    # ---- low-density points: pruned full search + per-point two-pass ----
    if (~high_density_mask).any():
        idxs = np.where(~high_density_mask)[0]
        # Stage A: cheap 1-split pass over full pool
        stage_a_scores = _validate_candidates(V2_CANDIDATES, obs, n_splits=1, seed=seed)
        n_eval_total += len(V2_CANDIDATES) * 1
        shortlist_idx = np.argsort(stage_a_scores)[:20]
        # Stage B: full 3-split CV only on survivors
        survivors = [V2_CANDIDATES[i] for i in shortlist_idx]
        stage_b_scores = _validate_candidates(survivors, obs, n_splits=3, seed=seed)
        n_eval_total += len(survivors) * 3
        order = np.argsort(stage_b_scores)[:n_top]
        top_cands = [survivors[i] for i in order]
        inv = 1.0 / (stage_b_scores[order] + 1e-6)
        w = inv / inv.sum()

        def ens_predict(o, xy, t, sigma_override=None):
            preds = []
            for c in top_cands:
                if c["kind"] == "gauss" and sigma_override is not None:
                    c = dict(c, sigma=sigma_override)
                preds.append(_predict_one(c, o, xy, t))
            return float(np.dot(w, preds))

        pass1_vals = {}
        for i in idxs:
            pass1_vals[i] = ens_predict(obs, miss_xy[i], miss_t[i], float(sigma_x[i]))
            final[i] = pass1_vals[i]

        # per-point two-pass: only if THIS point's local density is below k_min
        k_min = 3
        need_two_pass = [i for i in idxs if rho[i] < k_min]
        if need_two_pass:
            aug_xy = np.vstack([obs.xy, miss_xy[list(pass1_vals.keys())]])
            aug_t = np.concatenate([obs.t, miss_t[list(pass1_vals.keys())]])
            aug_v = np.concatenate([obs.v, np.array(list(pass1_vals.values()))])
            aug = STData(aug_xy, aug_t, aug_v)
            for i in need_two_pass:
                p2 = ens_predict(aug, miss_xy[i], miss_t[i], float(sigma_x[i]))
                final[i] = lambda_x[i] * pass1_vals[i] + (1 - lambda_x[i]) * p2

    return final, n_eval_total
