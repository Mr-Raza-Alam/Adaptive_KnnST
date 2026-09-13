import numpy as np
import time
import argparse
from utils import load_seoul_dataset, load_temperature_dataset, STData, _ne
from base_knn_st import VoxelGrid, knn_st_voxel_predict
from regression_baseline import regression_impute

# change these settings to run different tests
POLLUTANTS = ["PM2.5", "PM10", "NO2"]       # which pollutants to test
DATASET_SIZES = [10000, 20000, 50000, 100000] # how many rows to use
MISSING_RATES = [0.4, 0.6, 0.8, 0.9]        # how much data to drop (40% to 90%)
N_REPEATS = 3                               # run it 3 times to get an average


def run_missing_rate_test(xy, t, v, sids, missing_rate, model_name="RF", seed=42):
    rng = np.random.default_rng(seed)
    n = len(v)
    n_miss = int(n * missing_rate)
    
    # randomly pick which data points will be "missing" for our test
    miss_idx = rng.choice(n, size=n_miss, replace=False)
    mask = np.ones(n, dtype=bool)
    mask[miss_idx] = False

    obs = STData(xy[mask], t[mask], v[mask])

    miss_xy = xy[miss_idx]
    miss_t = t[miss_idx]
    miss_v = v[miss_idx]

    # --- Regression Baseline (Marchang 2022) ---
    # Construct feature matrix X: [time, lon, lat]
    X_full = np.column_stack([t, xy[:, 0], xy[:, 1]])
    X_train = X_full[mask]
    y_train = v[mask]
    X_missing = X_full[miss_idx]

    t0 = time.time()
    # we use Random Forest (RF) as the strongest baseline from the paper
    reg_pred, _ = regression_impute(X_train, y_train, X_missing, model=model_name, random_state=seed)
    time_reg = time.time() - t0
    ne_reg = _ne(miss_v, reg_pred)

    # --- Base KNN-ST (Marchang 2021) ---
    t0 = time.time()
    base_grid = VoxelGrid(obs)
    base_pred = np.array([
        knn_st_voxel_predict(base_grid, obs, miss_xy[i], miss_t[i])
        for i in range(len(miss_xy))
    ])
    time_base = time.time() - t0
    ne_base = _ne(miss_v, base_pred)
    
    return ne_reg, time_reg, ne_base, time_base


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="seoul", choices=["seoul", "temperature"])
    args = parser.parse_args()

    if args.dataset == "seoul":
        features_to_test = POLLUTANTS
    else:
        features_to_test = ["Temperature"]

    for ds_size in DATASET_SIZES:
        for feature in features_to_test:
            print(f"\n{'='*60}")
            print(f"  Dataset: {args.dataset.capitalize()} | Feature: {feature} | Size: {ds_size}")
            print(f"{'='*60}")

            try:
                if args.dataset == "seoul":
                    xy, t, v, sids = load_seoul_dataset(
                        summary_csv="Seoul_dataset_2017.csv",
                        pollutant=feature,
                        n_rows=ds_size
                    )
                else:
                    xy, t, v, sids = load_temperature_dataset(
                        csv_file="crowd_temperature.csv",
                        n_rows=ds_size
                    )
            except Exception as e:
                print(f"  ERROR loading {feature}: {e}")
                continue

            print(f"{'Rate':>5} | {'Reg(RF) NE':>11} | {'Reg t(s)':>10} | {'Base NE':>10} | {'Base t(s)':>11}")
            print("-" * 60)

            for r in MISSING_RATES:
                err_reg, t_reg, err_base, t_base = [], [], [], []
                for s in range(N_REPEATS):
                    ereg, treg, ebase, tbase = run_missing_rate_test(xy, t, v, sids, r, model_name="RF", seed=42+s)
                    err_reg.append(ereg)
                    t_reg.append(treg)
                    err_base.append(ebase)
                    t_base.append(tbase)

                print(f"{r:>5.0%} | {np.mean(err_reg):>11.4f} | {np.mean(t_reg):>10.2f} | {np.mean(err_base):>10.4f} | {np.mean(t_base):>11.2f}")

    print(f"\n{'='*60}")
    print("  Regression vs Base KNN-ST tests completed.")
    print(f"{'='*60}")
