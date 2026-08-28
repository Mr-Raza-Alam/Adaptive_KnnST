"""
Station-Aware Temporal-Spatial KNN-ST (Improved Algorithm)

Key insight: The original KNN-ST by Marchang (2021) treats every data point
as an anonymous (x, y, t, v) tuple and mixes spatial + temporal distance
into one single number d_ST = α·dS + (1-α)·dT. This loses critical
information — it cannot distinguish between:
  - "same station, 1 hour ago" (very informative)
  - "different station 5 km away, same hour" (less informative)

This improved algorithm SEPARATES the two dimensions:
  1. Temporal Prediction: If the SAME station has nearby readings in time,
     interpolate between them (extremely accurate for short gaps).
  2. Spatial Prediction:  If no temporal neighbors exist, use readings from
     NEARBY stations at the SAME time (standard spatial IDW).
  3. Adaptive Blending:   When both are available, blend them with
     data-driven weights based on neighbor quality.

This exploits the STRUCTURE of MCS data that plain KNN-ST ignores.
"""

import numpy as np
from collections import defaultdict
from utils import STData, spatiotemporal_dist


# ---------------------------------------------------------------------------
# 1.  Station-Indexed Temporal Lookup  (O(log N) per query)
# ---------------------------------------------------------------------------
class StationTemporalIndex:
    """
    Groups observed readings by station_id and sorts them by time.
    Allows O(log N) lookup of the nearest temporal neighbors for any
    (station, time) query.
    """
    def __init__(self, station_ids, times, values, xy_coords):
        self.station_data = {}           # station_id -> sorted arrays
        self.station_xy = {}             # station_id -> (x, y)

        unique_stations = np.unique(station_ids)
        for sid in unique_stations:
            mask = station_ids == sid
            t_arr = times[mask]
            v_arr = values[mask]

            order = np.argsort(t_arr)
            self.station_data[sid] = (t_arr[order], v_arr[order])
            # All rows for one station share the same (x,y)
            self.station_xy[sid] = xy_coords[mask][0]

    def temporal_neighbors(self, station_id, query_t, max_dt=None):
        """
        Find the closest reading BEFORE and AFTER query_t at the same station.
        Returns (t_before, v_before, t_after, v_after) or None for each side
        if no neighbor exists (or is beyond max_dt).
        """
        if station_id not in self.station_data:
            return None, None, None, None

        t_arr, v_arr = self.station_data[station_id]
        # Binary search for insertion point
        idx = np.searchsorted(t_arr, query_t)

        t_before = v_before = t_after = v_after = None

        # Left neighbor (before)
        if idx > 0:
            tb = t_arr[idx - 1]
            if max_dt is None or abs(query_t - tb) <= max_dt:
                t_before = tb
                v_before = v_arr[idx - 1]

        # Right neighbor (after)
        if idx < len(t_arr):
            ta = t_arr[idx]
            if max_dt is None or abs(ta - query_t) <= max_dt:
                t_after = ta
                v_after = v_arr[idx]

        return t_before, v_before, t_after, v_after

    def temporal_predict(self, station_id, query_t, max_dt=None):
        """
        Predict value at (station_id, query_t) using linear interpolation
        between the two closest temporal neighbors at the SAME station.

        Returns (prediction, confidence) where confidence indicates quality:
          - 1.0  = interpolation between two bracketing neighbors
          - 0.7  = extrapolation from one-sided neighbor
          - 0.0  = no temporal neighbor available
        """
        tb, vb, ta, va = self.temporal_neighbors(station_id, query_t, max_dt)

        if tb is not None and ta is not None:
            # Interpolation (best case): linearly blend between before & after
            dt_total = ta - tb
            if dt_total < 1e-12:
                return float(vb), 1.0
            w = (query_t - tb) / dt_total
            return float((1 - w) * vb + w * va), 1.0

        elif tb is not None:
            # Extrapolation from left only
            return float(vb), 0.7

        elif ta is not None:
            # Extrapolation from right only
            return float(va), 0.7

        else:
            return 0.0, 0.0


# ---------------------------------------------------------------------------
# 2.  Spatial IDW Prediction  (Same as paper's weighted KNN-ST)
# ---------------------------------------------------------------------------
def spatial_idw_predict(obs: STData, query_xy, query_t, alpha, k, eps=1e-3):
    """
    Standard KNN-ST weighted prediction from Marchang (2021).
    Finds k nearest neighbors by d_ST and applies IDW.
    """
    d = spatiotemporal_dist(obs.xy, obs.t, query_xy, query_t, alpha)
    k_final = min(k, len(d))
    if k_final == 0:
        return 0.0, 0.0

    idx_sort = np.argsort(d)[:k_final]
    best_d = d[idx_sort]

    w = 1.0 / (best_d + eps)
    pred = float(np.sum(w * obs.v[idx_sort]) / np.sum(w))

    # Confidence based on how close the neighbors are (closer = higher)
    avg_d = np.mean(best_d)
    conf = float(np.clip(1.0 / (1.0 + 5.0 * avg_d), 0.1, 0.9))

    return pred, conf


# ---------------------------------------------------------------------------
# 3.  Fast Spatial IDW using VoxelGrid  (for scalability)
# ---------------------------------------------------------------------------
class SpatialVoxelGrid:
    """Simple voxel grid for fast spatial neighbor lookup."""
    def __init__(self, obs: STData, bins_xy=10, bins_t=24):
        self.bins_xy = bins_xy
        self.bins_t = bins_t
        self.grid = defaultdict(list)
        self.obs = obs

        for i in range(len(obs.v)):
            x, y = obs.xy[i]
            t = obs.t[i]
            ix = max(0, min(int(x * bins_xy), bins_xy - 1))
            iy = max(0, min(int(y * bins_xy), bins_xy - 1))
            it = max(0, min(int(t * bins_t), bins_t - 1))
            self.grid[(ix, iy, it)].append(i)

    def get_candidates(self, x, y, t, k_min=6):
        """Expand rings until we have at least k_min candidates."""
        ix = max(0, min(int(x * self.bins_xy), self.bins_xy - 1))
        iy = max(0, min(int(y * self.bins_xy), self.bins_xy - 1))
        it = max(0, min(int(t * self.bins_t), self.bins_t - 1))

        candidates = []
        radius = 0
        max_radius = max(self.bins_xy, self.bins_t)

        while len(candidates) < k_min and radius <= max_radius:
            for rx in range(ix - radius, ix + radius + 1):
                if rx < 0 or rx >= self.bins_xy: continue
                for ry in range(iy - radius, iy + radius + 1):
                    if ry < 0 or ry >= self.bins_xy: continue
                    for rt in range(it - radius, it + radius + 1):
                        if rt < 0 or rt >= self.bins_t: continue
                        if radius == 0 or max(abs(rx-ix), abs(ry-iy), abs(rt-it)) == radius:
                            candidates.extend(self.grid.get((rx, ry, rt), []))
            radius += 1

        return np.array(candidates) if candidates else np.array([], dtype=int)

    def spatial_idw_predict(self, query_xy, query_t, alpha=0.5, k=5, eps=1e-3):
        """Fast spatial IDW using voxel-accelerated neighbor search."""
        cand_idx = self.get_candidates(query_xy[0], query_xy[1], query_t, k_min=k)

        if len(cand_idx) == 0:
            return 0.0, 0.0

        d = spatiotemporal_dist(
            self.obs.xy[cand_idx], self.obs.t[cand_idx],
            query_xy, query_t, alpha
        )

        k_final = min(k, len(d))
        idx_sort = np.argsort(d)[:k_final]
        best_idx = cand_idx[idx_sort]
        best_d = d[idx_sort]

        w = 1.0 / (best_d + eps)
        pred = float(np.sum(w * self.obs.v[best_idx]) / np.sum(w))

        avg_d = np.mean(best_d)
        conf = float(np.clip(1.0 / (1.0 + 5.0 * avg_d), 0.1, 0.9))

        return pred, conf


# ---------------------------------------------------------------------------
# 4.  Main Prediction Engine
# ---------------------------------------------------------------------------
def hybrid_knn_st_predict(obs: STData, miss_xy, miss_t,
                          station_ids_obs, station_ids_miss,
                          alpha=0.5, k=5, temporal_max_dt=None,
                          seed=0):
    """
    Station-Aware Temporal-Spatial KNN-ST.

    For each missing point:
      1. Try temporal interpolation at the SAME station first.
      2. If temporal is unavailable/weak, compute spatial IDW.
      3. Blend both predictions based on their confidence scores.

    Parameters
    ----------
    obs : STData
        Known observed readings (xy, t, v arrays).
    miss_xy : ndarray (M, 2)
        Spatial coordinates of missing points.
    miss_t : ndarray (M,)
        Temporal coordinates of missing points.
    station_ids_obs : ndarray (N,)
        Station ID for each observed reading.
    station_ids_miss : ndarray (M,)
        Station ID for each missing point.
    alpha : float
        Spatial weight for d_ST (same as paper).
    k : int
        Number of spatial neighbors (same as paper).
    temporal_max_dt : float or None
        Maximum temporal gap (in normalized time) for temporal neighbors.
        None = no limit (use any temporal neighbor from same station).
    seed : int
        Random seed (unused here, kept for API compatibility).

    Returns
    -------
    preds : ndarray (M,)
        Predicted values for each missing point.
    """
    n_miss = len(miss_t)
    preds = np.zeros(n_miss)

    # Build the two lookup structures
    temporal_index = StationTemporalIndex(
        station_ids_obs, obs.t, obs.v, obs.xy
    )
    spatial_grid = SpatialVoxelGrid(obs, bins_xy=10, bins_t=24)

    for i in range(n_miss):
        sid = station_ids_miss[i]
        qt = miss_t[i]
        qxy = miss_xy[i]

        # --- Step 1: Temporal prediction (same station) ---
        t_pred, t_conf = temporal_index.temporal_predict(sid, qt, max_dt=temporal_max_dt)

        # --- Step 2: Spatial prediction (nearby stations, same time) ---
        s_pred, s_conf = spatial_grid.spatial_idw_predict(qxy, qt, alpha=alpha, k=k)

        # --- Step 3: Adaptive blending ---
        if t_conf > 0 and s_conf > 0:
            # Both available: blend by confidence
            total = t_conf + s_conf
            preds[i] = (t_conf / total) * t_pred + (s_conf / total) * s_pred
        elif t_conf > 0:
            # Only temporal available
            preds[i] = t_pred
        elif s_conf > 0:
            # Only spatial available
            preds[i] = s_pred
        else:
            # Absolute fallback (very rare)
            preds[i] = 0.0

    return preds
