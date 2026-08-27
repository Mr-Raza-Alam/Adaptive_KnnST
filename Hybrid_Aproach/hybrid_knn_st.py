import numpy as np
from collections import defaultdict
from utils import STData, spatiotemporal_dist

class FastVoxelGrid:
    """
    O(N) Hash-Map based Indexer for scalable spatiotemporal searching.
    Provides fast O(1) local density estimation and O(1) candidate superset fetching.
    """
    def __init__(self, obs: STData, bins_xy=10, bins_t=24):
        self.bins_xy = bins_xy
        self.bins_t = bins_t
        self.grid = defaultdict(list)
        
        # We assume coords are normalized [0,1]
        for i in range(len(obs.v)):
            x, y = obs.xy[i]
            t = obs.t[i]
            
            # Map [0,1] to [0, bins-1] safely
            ix = max(0, min(int(x * bins_xy), bins_xy - 1))
            iy = max(0, min(int(y * bins_xy), bins_xy - 1))
            it = max(0, min(int(t * bins_t), bins_t - 1))
            
            self.grid[(ix, iy, it)].append(i)
            
    def get_voxel_coords(self, x, y, t):
        ix = max(0, min(int(x * self.bins_xy), self.bins_xy - 1))
        iy = max(0, min(int(y * self.bins_xy), self.bins_xy - 1))
        it = max(0, min(int(t * self.bins_t), self.bins_t - 1))
        return ix, iy, it
        
    def get_superset_and_density(self, x, y, t, radius=1, max_candidates=100):
        """
        Returns the immediate local density (rho) and a superset of candidate points.
        rho is defined strictly as points in the immediate vicinity (radius <= 1).
        If the immediate vicinity is too sparse, it expands the radius to fetch a superset.
        """
        ix, iy, it = self.get_voxel_coords(x, y, t)
        
        candidates = []
        rho = 0
        current_radius = 0
        
        # 1. Calculate local density (rho) strictly within radius 1
        for rx in range(ix - 1, ix + 2):
            if rx < 0 or rx >= self.bins_xy: continue
            for ry in range(iy - 1, iy + 2):
                if ry < 0 or ry >= self.bins_xy: continue
                for rt in range(it - 1, it + 2):
                    if rt < 0 or rt >= self.bins_t: continue
                    rho += len(self.grid.get((rx, ry, rt), []))
        
        # 2. Fetch superset of candidates (expanding radius if necessary)
        max_radius = max(self.bins_xy, self.bins_t)
        while len(candidates) < max_candidates and current_radius <= max_radius:
            for rx in range(ix - current_radius, ix + current_radius + 1):
                if rx < 0 or rx >= self.bins_xy: continue
                for ry in range(iy - current_radius, iy + current_radius + 1):
                    if ry < 0 or ry >= self.bins_xy: continue
                    for rt in range(it - current_radius, it + current_radius + 1):
                        if rt < 0 or rt >= self.bins_t: continue
                        
                        # Only grab voxels on the boundary of the current radius
                        if current_radius == 0 or max(abs(rx - ix), abs(ry - iy), abs(rt - it)) == current_radius:
                            candidates.extend(self.grid.get((rx, ry, rt), []))
            current_radius += 1
            
        return rho, np.array(candidates)


# --- Adaptive Candidate Definitions ---
ALPHAS = [0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85]
KS = [3, 4, 5, 6, 8]
SIGMAS = [0.2, 0.3, 0.5]

def build_candidates():
    cands = []
    # Broad search for sparse areas
    for a in [0.35, 0.5, 0.65]:
        for k in [4, 5, 6]:
            for s in SIGMAS:
                cands.append(dict(kind="gauss", alpha=a, k=k, sigma=s))
    return cands

CANDIDATES = build_candidates()
# Fast shortlist for dense areas
SHORTLIST = [
    dict(kind="gauss", alpha=0.5, k=5, sigma=0.2),
    dict(kind="gauss", alpha=0.35, k=6, sigma=0.3),
    dict(kind="gauss", alpha=0.65, k=4, sigma=0.2),
]


def _predict_on_superset(cand, obs: STData, query_xy, query_t, superset_idx, eps=1e-3, override_sigma=None):
    """
    Evaluates a specific hyperparameter candidate purely on the pre-fetched superset.
    """
    alpha, k = cand["alpha"], cand["k"]
    
    # Calculate EXACT custom distance ONLY for the superset points
    exact_d = spatiotemporal_dist(obs.xy[superset_idx], obs.t[superset_idx], query_xy, query_t, alpha)
    
    # Sort and pick the true top K from the superset
    k_final = min(k, len(exact_d))
    if k_final == 0:
        return 0.0
        
    idx_sort = np.argsort(exact_d)[:k_final]
    best_idx = superset_idx[idx_sort]
    best_d = exact_d[idx_sort]
    
    # Apply Gaussian weighting
    sigma = override_sigma if override_sigma is not None else cand["sigma"]
    w = np.exp(-(best_d ** 2) / (2 * sigma ** 2)) + 1e-9
        
    return float(np.sum(w * obs.v[best_idx]) / np.sum(w))


def _evaluate_candidate(cand, obs: STData, val_idx_list, val_supersets, override_sigma_list=None):
    """
    Evaluates a candidate across a list of validation points (1-split cross validation).
    """
    errors = 0.0
    for i, (v_idx, superset) in enumerate(zip(val_idx_list, val_supersets)):
        if len(superset) == 0: continue
        # Remove the validation point itself from its superset to prevent cheating
        clean_superset = superset[superset != v_idx]
        
        sig = override_sigma_list[i] if override_sigma_list else None
        pred = _predict_on_superset(cand, obs, obs.xy[v_idx], obs.t[v_idx], clean_superset, override_sigma=sig)
        errors += abs(pred - obs.v[v_idx])
    
    denom = np.sum(np.abs(obs.v[val_idx_list]))
    if denom == 0: return 0.0
    return float(errors / denom)


def adaptive_sigma(rho, sigma0=0.3, rho_ref=10.0, gamma=0.5, smin=0.15, smax=0.8):
    """Dynamically stretches the Gaussian kernel for empty voxels."""
    return np.clip(sigma0 * (rho_ref / (rho + 1)) ** gamma, smin, smax)

def adaptive_lambda(rho, lam0=0.7, delta=0.35, rho_max=15.0, lmin=0.35, lmax=0.85):
    """Dynamically scales confidence between Pass 1 and Pass 2."""
    return np.clip(lam0 - delta * (1 - np.minimum(rho, rho_max) / rho_max), lmin, lmax)


def hybrid_knn_st_predict(obs: STData, miss_xy, miss_t, missing_rate, k_min=3, density_threshold=10, seed=0):
    """
    Fast Hybrid Ad-KNN-ST algorithm.
    """
    rng = np.random.default_rng(seed)
    
    # 1. Build the O(N) Scalability Engine (VoxelGrid)
    voxel_grid = FastVoxelGrid(obs, bins_xy=10, bins_t=24)
    
    final_preds = np.zeros(len(miss_xy))
    rhos = np.zeros(len(miss_xy))
    supersets = []
    
    # Fetch supersets and density for all missing points
    for i in range(len(miss_xy)):
        r, sup = voxel_grid.get_superset_and_density(miss_xy[i][0], miss_xy[i][1], miss_t[i], radius=1)
        rhos[i] = r
        supersets.append(sup)
        
    sigma_x = adaptive_sigma(rhos)
    lambda_x = adaptive_lambda(rhos)
    
    high_density_mask = rhos >= density_threshold
    
    # --- PASS 1 ---
    pass1_vals = np.zeros(len(miss_xy))
    
    # For validation, we grab a 15% random sample from the known points
    n_val = max(5, int(len(obs.v) * 0.15))
    val_idx = rng.choice(len(obs.v), size=n_val, replace=False)
    
    # Pre-fetch supersets for validation points to keep validation lightning fast
    val_supersets = []
    val_rhos = []
    for vi in val_idx:
        r, sup = voxel_grid.get_superset_and_density(obs.xy[vi][0], obs.xy[vi][1], obs.t[vi])
        val_supersets.append(sup)
        val_rhos.append(r)
    val_sigma_x = adaptive_sigma(np.array(val_rhos))
    
    # High Density: Fast Shortlist Ensemble
    if high_density_mask.any():
        scores = [_evaluate_candidate(c, obs, val_idx, val_supersets) for c in SHORTLIST]
        order = np.argsort(scores)[:3] # top 3
        cands = [SHORTLIST[i] for i in order]
        
        inv = 1.0 / (np.array([scores[i] for i in order]) + 1e-6)
        w = inv / inv.sum()
        
        idxs = np.where(high_density_mask)[0]
        for i in idxs:
            if len(supersets[i]) == 0: continue
            preds = [_predict_on_superset(c, obs, miss_xy[i], miss_t[i], supersets[i], override_sigma=float(sigma_x[i])) for c in cands]
            pass1_vals[i] = float(np.dot(w, preds))
            final_preds[i] = pass1_vals[i]
            
    # Low Density: Exhaustive Tuning
    if (~high_density_mask).any():
        scores = [_evaluate_candidate(c, obs, val_idx, val_supersets) for c in CANDIDATES]
        order = np.argsort(scores)[:5] # top 5 survivors
        cands = [CANDIDATES[i] for i in order]
        
        inv = 1.0 / (np.array([scores[i] for i in order]) + 1e-6)
        w = inv / inv.sum()
        
        idxs = np.where(~high_density_mask)[0]
        for i in idxs:
            if len(supersets[i]) == 0: continue
            preds = [_predict_on_superset(c, obs, miss_xy[i], miss_t[i], supersets[i], override_sigma=float(sigma_x[i])) for c in cands]
            pass1_vals[i] = float(np.dot(w, preds))
            final_preds[i] = pass1_vals[i]
            
            
    # --- PASS 2 (Extremely Sparse Fallback) ---
    need_two_pass = [i for i in range(len(miss_xy)) if rhos[i] < k_min]
    
    if need_two_pass:
        # Augment the known dataset with Pass 1 estimates
        aug_xy = np.vstack([obs.xy, miss_xy])
        aug_t = np.concatenate([obs.t, miss_t])
        aug_v = np.concatenate([obs.v, pass1_vals])
        aug = STData(aug_xy, aug_t, aug_v)
        
        # Fast Re-indexing: Rebuild O(N) VoxelGrid instantly
        aug_grid = FastVoxelGrid(aug, bins_xy=10, bins_t=24)
        
        # Since these are isolated points, we use the rigorous candidates found in Phase 1
        for i in need_two_pass:
            _, aug_superset = aug_grid.get_superset_and_density(miss_xy[i][0], miss_xy[i][1], miss_t[i])
            if len(aug_superset) == 0: continue
            
            preds = [_predict_on_superset(c, aug, miss_xy[i], miss_t[i], aug_superset, override_sigma=float(sigma_x[i])) for c in cands]
            p2 = float(np.dot(w, preds))
            
            # Blend Pass 1 and Pass 2 based on density confidence
            final_preds[i] = lambda_x[i] * pass1_vals[i] + (1 - lambda_x[i]) * p2
            
    return final_preds
