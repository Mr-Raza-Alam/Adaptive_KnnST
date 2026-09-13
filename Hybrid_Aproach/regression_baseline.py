"""
Regression-based missing-data imputation for Sparse Mobile Crowd Sensing.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

__all__ = [
    "make_regressor",
    "REGRESSORS",
    "regression_impute",
    "nmae",
    "random_loss",
    "reduce_sensing_tasks",
]


# --------------------------------------------------------------------- #
#  1. The seven regression algorithms (Sect. 3.2)                       #
# --------------------------------------------------------------------- #
def make_regressor(name="RF", random_state=0):
    name = name.strip()
    if name == "LR":
        return LinearRegression()
    if name == "LASSO":
        return Lasso(random_state=random_state)
    if name == "Ridge":
        return Ridge(random_state=random_state)
    if name == "EN":
        return ElasticNet(random_state=random_state)
    if name.startswith("uni-"):
        k = int(name.split("-")[1])
        return KNeighborsRegressor(n_neighbors=k, weights="uniform")
    if name.startswith("dis-"):
        k = int(name.split("-")[1])
        return KNeighborsRegressor(n_neighbors=k, weights="distance")
    if name == "DT":
        return DecisionTreeRegressor(random_state=random_state)
    if name == "RF":
        return RandomForestRegressor(
            n_estimators=100, random_state=random_state, n_jobs=-1
        )
    raise ValueError(f"unknown regressor name: {name!r}")


# The set used in the paper's experiments (K values as in Figs. 5-7).
REGRESSORS = [
    "LR", "LASSO", "Ridge", "EN",
    "uni-2", "uni-4", "uni-6", "uni-8",
    "dis-2", "dis-4", "dis-6", "dis-8",
    "DT", "RF",
]


# --------------------------------------------------------------------- #
#  2. Regression-based missing-data inference                          #
# --------------------------------------------------------------------- #
def regression_impute(X_train, y_train, X_missing, model="RF",
                       random_state=0):
    if isinstance(model, str):
        model = make_regressor(model, random_state=random_state)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_missing)
    return y_hat, model


# --------------------------------------------------------------------- #
#  3. Random loss model + NMAE metric (Sects. 3.1 / 5)                  #
# --------------------------------------------------------------------- #
def random_loss(mask_length, loss_prob, rng=None):
    """Purely random loss model (Sect. 5): each data point is removed
    independently with probability `loss_prob`.

    Returns a boolean array `deleted` of length `mask_length`
    (True = the point is deleted and must be imputed).
    """
    rng = np.random.default_rng(rng)
    return rng.random(mask_length) < loss_prob


def nmae(y_true, y_hat):
    """Normalized Mean Absolute Error (Eq. 4 of the paper):

        NMAE = sum_i |T_i - T_hat_i|  /  sum_i |T_i|
    """
    y_true = np.asarray(y_true, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    denom = np.abs(y_true).sum()
    if denom == 0:
        return np.nan
    return np.abs(y_true - y_hat).sum() / denom


# --------------------------------------------------------------------- #
#  4. ReduceSensingTasks: divide-and-conquer task reduction (Sect. 4)  #
# --------------------------------------------------------------------- #
def reduce_sensing_tasks(cells, model="RF", loss_prob=0.2,
                          nmae_limit=0.1, S=4, rng=None,
                          prev_slot_values=None, random_state=0):
    rng = np.random.default_rng(rng)
    n = len(cells)
    keep_mask = np.ones(n, dtype=bool)
    stats = {"deleted": 0, "calls": 0, "levels": 0}

    if n == 0:
        return keep_mask, stats

    feats = np.asarray([np.asarray(c["features"], dtype=float)
                        for c in cells])
    prev = (np.asarray(prev_slot_values, dtype=float)
            if prev_slot_values is not None
            else np.asarray([c["value"] for c in cells], dtype=float))

    def _recurse(idx, depth):
        # 1) stop if the problem is small
        if len(idx) <= S:
            return
        stats["levels"] = max(stats["levels"], depth)

        # 2) randomly delete tasks with loss probability
        delete = rng.random(len(idx)) < loss_prob
        # never delete everything: need data to infer from
        if delete.all():
            delete[rng.integers(len(delete))] = False
        del_idx = idx[delete]
        keep_idx = idx[~delete]
        if len(del_idx) == 0 or len(keep_idx) == 0:
            return

        # 3) infer the deleted tasks' values with the candidate method
        stats["calls"] += 1
        y_hat, _ = regression_impute(
            feats[keep_idx], prev[keep_idx], feats[del_idx],
            model=model, random_state=random_state,
        )

        # 4) inference error against previous-slot values
        err = nmae(prev[del_idx], y_hat)

        # 5) accept / reject, then divide AOI into 4 and recurse
        if err <= nmae_limit:
            keep_mask[del_idx] = False          # confirm the deletion
            stats["deleted"] += len(del_idx)

            # Divide the AOI (this sub-region) into 4 equal subareas
            # at the mid-points of latitude / longitude, then recurse
            # on the REMAINING (kept) tasks in each subarea
            # (per the recurrence R(n) = 4 R((1-p)n/4) + p n).
            lat = feats[idx, -2]
            lon = feats[idx, -1]
            lat_mid = (lat.min() + lat.max()) / 2.0
            lon_mid = (lon.min() + lon.max()) / 2.0
            klat = feats[keep_idx, -2]
            klon = feats[keep_idx, -1]
            quadrants = [
                keep_idx[(klat <= lat_mid) & (klon <= lon_mid)],
                keep_idx[(klat <= lat_mid) & (klon > lon_mid)],
                keep_idx[(klat > lat_mid) & (klon <= lon_mid)],
                keep_idx[(klat > lat_mid) & (klon > lon_mid)],
            ]
            for q in quadrants:
                _recurse(q, depth + 1)
        # else: undo the deletion -- keep_mask untouched for these tasks

    _recurse(np.arange(n), 0)
    return keep_mask, stats


# --------------------------------------------------------------------- #
#  5. Demo / smoke test on synthetic environmental data                 #
# --------------------------------------------------------------------- #
def _demo():
    rng = np.random.default_rng(42)

    # Synthetic "crowdad-style" data points:
    # features = [day, time, latitude, longitude], target = temperature
    n_points = 4000
    day = rng.integers(1, 5, n_points)            # 4 days
    hour = rng.integers(0, 24, n_points)
    lat = rng.uniform(41.80, 41.95, n_points)     # Rome-ish
    lon = rng.uniform(12.40, 12.60, n_points)
    X = np.column_stack([day, hour, lat, lon])

    temperature = (
        15.0
        + 6.0 * np.sin((hour - 6) / 24 * 2 * np.pi)      # daily cycle
        - 8.0 * (lat - 41.80) / 0.15                     # spatial gradient
        + 4.0 * (lon - 12.40) / 0.20
        + 0.5 * day
        + 0.3 * rng.standard_normal(n_points)           # sensing noise
    )

    print("Regression-based missing data inference "
          "(Marchang et al., J. Supercomputing 2022)")
    print("synthetic temperature data, features = "
          "[day, hour, latitude, longitude]\n")
    print(f"{'model':<8} | " + " | ".join(f"p={p:<4}" for p in
                                          (0.3, 0.5, 0.7, 0.8, 0.9)))
    print("-" * 52)
    for name in ("LR", "LASSO", "Ridge", "EN",
                 "uni-4", "dis-4", "DT", "RF"):
        row = []
        for p in (0.3, 0.5, 0.7, 0.8, 0.9):
            deleted = random_loss(n_points, p, rng)
            y_hat, _ = regression_impute(
                X[~deleted], temperature[~deleted], X[deleted], model=name)
            row.append(f"{nmae(temperature[deleted], y_hat):.3f}")
        print(f"{name:<8} | " + " | ".join(f"{v:<5}" for v in row))

    # ---- ReduceSensingTasks demonstration ----
    print("\nReduceSensingTasks (divide-and-conquer, p=0.2, "
          "NMAE limit=0.10, S=4):")
    cells = [
        {"features": X[i], "value": temperature[i]}
        for i in range(n_points)
    ]
    keep_mask, stats = reduce_sensing_tasks(
        cells, model="RF", loss_prob=0.2, nmae_limit=0.10, S=4,
        rng=7, random_state=0,
    )
    n = len(cells)
    print(f"  total tasks        : {n}")
    print(f"  tasks kept         : {keep_mask.sum()}")
    print(f"  tasks deleted      : {stats['deleted']} "
          f"({stats['deleted'] / n:.1%} reduction)")
    print(f"  inference calls    : {stats['calls']}")
    print(f"  recursion depth    : {stats['levels']}")


if __name__ == "__main__":
    _demo()
