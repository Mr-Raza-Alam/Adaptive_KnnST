"""
Station-Aware Temporal-Spatial KNN-ST (v3 — Balanced Fix)

Changes from v2:
  - Extrapolation (one-sided temporal) → conf_T = 0.0 (completely ignored)
  - Interpolation (both sides) → conf_T stays HIGH for short gaps,
    gradually decreases only for very long gaps (24h+ scale)
  - k=4 for spatial (matches baseline exactly)
  - Non-inferiority guard kept at conf_T < 0.15 (rarely triggers)
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
    Allows O(log N) lookup of the nearest temporal neighbors.
    """
    def __init__(self, station_ids, times, values, xy_coords, total_hours):
        self.station_data = {}
        self.station_xy = {}
        self.total_hours = total_hours

        unique_stations = np.unique(station_ids)
        for sid in unique_stations:
            mask = station_ids == sid
            t_arr = times[mask]
            v_arr = values[mask]

            order = np.argsort(t_arr)
            self.station_data[sid] = (t_arr[order], v_arr[order])
            self.station_xy[sid] = xy_coords[mask][0]

    def temporal_predict(self, station_id, query_t):
        """
        Predict using temporal neighbors at the SAME station.

        v3 logic:
          - INTERPOLATION (both before & after exist):
              conf_T = 1.0 / (1.0 + gap_hours / 24.0)
              Short gaps → conf_T ≈ 1.0 (dominant)
              Long gaps  → conf_T decreases gradually
          - EXTRAPOLATION (only one side):
              conf_T = 0.0 (completely ignored — this was the root
              cause of v1's 33% failure rate)
        """
        if station_id not in self.station_data:
            return 0.0, 0.0

        t_arr, v_arr = self.station_data[station_id]
        idx = np.searchsorted(t_arr, query_t)

        t_before = v_before = t_after = v_after = None

        if idx > 0:
            t_before = t_arr[idx - 1]
            v_before = v_arr[idx - 1]

        if idx < len(t_arr):
            t_after = t_arr[idx]
            v_after = v_arr[idx]

        if t_before is not None and t_after is not None:
            # INTERPOLATION — reliable, keep high confidence
            dt_total = t_after - t_before
            if dt_total < 1e-12:
                return float(v_before), 1.0
            w = (query_t - t_before) / dt_total
            pred = float((1 - w) * v_before + w * v_after)

            # Gap-aware but generous: only penalize for very long gaps
            gap_hours = dt_total * self.total_hours
            conf = 1.0 / (1.0 + gap_hours / 24.0)
            return pred, conf

        else:
            # EXTRAPOLATION (one side or no side) → conf_T = 0.0
            # This completely eliminates the root cause of v1 failures
            return 0.0, 0.0


# ---------------------------------------------------------------------------
# 2.  Fast Spatial IDW using VoxelGrid
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

    def get_candidates(self, x, y, t, k_min=4):
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

    def spatial_idw_predict(self, query_xy, query_t, alpha=0.5, k=4, eps=1e-3):
        """Spatial IDW with k=4 — identical to baseline."""
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
# 3.  Main Prediction Engine (v3)
# ---------------------------------------------------------------------------
NON_INFERIORITY_THRESHOLD = 0.15

def hybrid_knn_st_predict(obs: STData, miss_xy, miss_t,
                          station_ids_obs, station_ids_miss,
                          alpha=0.5, k=4, total_hours=8759.0,
                          seed=0):
    """
    Station-Aware Temporal-Spatial KNN-ST (v3).

    v3 design:
      - Extrapolation completely eliminated (conf_T = 0.0)
      - Interpolation keeps high confidence for short gaps
      - k=4 matches baseline exactly
      - Non-inferiority guard at conf_T < 0.15 (safety net)
    """
    n_miss = len(miss_t)
    preds = np.zeros(n_miss)

    # Build the two lookup structures
    temporal_index = StationTemporalIndex(
        station_ids_obs, obs.t, obs.v, obs.xy, total_hours
    )
    spatial_grid = SpatialVoxelGrid(obs, bins_xy=10, bins_t=24)

    for i in range(n_miss):
        sid = station_ids_miss[i]
        qt = miss_t[i]
        qxy = miss_xy[i]

        # --- Step 1: Temporal prediction (same station) ---
        t_pred, t_conf = temporal_index.temporal_predict(sid, qt)

        # --- Step 2: Spatial prediction (k=4, identical to baseline) ---
        s_pred, s_conf = spatial_grid.spatial_idw_predict(qxy, qt, alpha=alpha, k=k)

        # --- Step 3: Blend or guard ---
        if t_conf < NON_INFERIORITY_THRESHOLD:
            # Temporal unreliable or unavailable → pure spatial (= baseline)
            preds[i] = s_pred
        elif s_conf > 0:
            # Both available → blend by confidence
            total = t_conf + s_conf
            preds[i] = (t_conf / total) * t_pred + (s_conf / total) * s_pred
        else:
            # Only temporal available (rare)
            preds[i] = t_pred

    return preds
