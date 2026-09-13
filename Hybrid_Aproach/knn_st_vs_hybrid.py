import numpy as np
import time
import argparse
from utils import load_seoul_dataset, load_temperature_dataset, STData, _ne
from hybrid_knn_st import hybrid_knn_st_predict
from base_knn_st import VoxelGrid, knn_st_voxel_predict

# change these settings to run different tests
POLLUTANTS = ["PM2.5", "PM10", "NO2"]       # which pollutants to test
DATASET_SIZES = [10000,20000,50000,150000,210000] # how many rows to use
MISSING_RATES = [0.4, 0.6, 0.8, 0.9]        # how much data to drop (40% to 90%)
N_REPEATS = 3                                # run it 3 times to get an average


def run_missing_rate_test(xy, t, v, sids, missing_rate, seed=42):
    rng = np.random.default_rng(seed)
    n = len(v)
    n_miss = int(n * missing_rate)
    
    # randomly pick which data points will be "missing" for our test
    miss_idx = rng.choice(n, size=n_miss, replace=False)
    mask = np.ones(n, dtype=bool)
    mask[miss_idx] = False

    obs = STData(xy[mask], t[mask], v[mask])
    obs_sids = sids[mask]

    miss_xy = xy[miss_idx]
    miss_t = t[miss_idx]
    miss_v = v[miss_idx]
    miss_sids = sids[miss_idx]

    # run the baseline algorithm first so we can compare
    # --- Baseline (KNN-ST from Marchang 2021) ---
    t0 = time.time()
    base_grid = VoxelGrid(obs)
    base_pred = np.array([
        knn_st_voxel_predict(base_grid, obs, miss_xy[i], miss_t[i])
        for i in range(len(miss_xy))
    ])
    time_base = time.time() - t0
    ne_base = _ne(miss_v, base_pred)

    # now run our new hybrid algorithm
    # --- Proposed: Station-Aware Temporal-Spatial KNN-ST ---
    t0 = time.time()
    hybrid_pred = hybrid_knn_st_predict(
        obs, miss_xy, miss_t,
        station_ids_obs=obs_sids,
        station_ids_miss=miss_sids,
        alpha=0.5, k=4, total_hours=8759.0
    )
    time_hybrid = time.time() - t0
    ne_hybrid = _ne(miss_v, hybrid_pred)
    
    return ne_base, time_base, ne_hybrid, time_hybrid


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

            print(f"{'Rate':>5} | {'Base NE':>10} | {'Base t(s)':>10} | {'Hybrid NE':>10} | {'Hybrid t(s)':>11}")
            print("-" * 60)

            for r in MISSING_RATES:
                err_base, t_base, err_hyb, t_hyb = [], [], [], []
                for s in range(N_REPEATS):
                    eb, tb, eh, th = run_missing_rate_test(xy, t, v, sids, r, seed=42+s)
                    err_base.append(eb)
                    t_base.append(tb)
                    err_hyb.append(eh)
                    t_hyb.append(th)

                print(f"{r:>5.0%} | {np.mean(err_base):>10.4f} | {np.mean(t_base):>10.2f} | {np.mean(err_hyb):>10.4f} | {np.mean(t_hyb):>11.2f}")

    print(f"\n{'='*60}")
    print("  All tests completed.")
    print(f"{'='*60}")
