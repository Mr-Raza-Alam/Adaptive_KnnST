import numpy as np
import time
from utils import load_seoul_dataset, STData, _ne
from hybrid_knn_st import hybrid_knn_st_predict
from cs_baseline import cs_impute

# change these settings to run different tests
POLLUTANTS = ["PM2.5", "PM10", "NO2"]       # which pollutants to test
DATASET_SIZES = [10000, 20000, 50000,100000] # how many rows to use
MISSING_RATES = [0.4, 0.6, 0.8, 0.9]        # how much data to drop (40% to 90%)
N_REPEATS = 3                               # run it 3 times to get an average


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

    # --- CS Baseline (Zhu et al. 2013) ---
    # Convert flat data to a matrix (time slots x stations)
    unique_t = np.unique(t)
    unique_sids = np.unique(sids)
    
    # map real values to matrix indices
    t_to_row = {val: i for i, val in enumerate(unique_t)}
    sid_to_col = {val: i for i, val in enumerate(unique_sids)}
    
    m, num_stations = len(unique_t), len(unique_sids)
    M = np.zeros((m, num_stations))
    B = np.zeros((m, num_stations))
    
    # fill the matrix with only observed data
    for i in range(len(v)):
        if mask[i]:
            r_idx = t_to_row[t[i]]
            c_idx = sid_to_col[sids[i]]
            M[r_idx, c_idx] = v[i]
            B[r_idx, c_idx] = 1.0

    t0 = time.time()
    # run CS imputation (hardcoded r=5, lam=1.0 for speed)
    X_hat = cs_impute(M, B, r=5, lam=1.0, t=100, seed=seed)
    
    # extract predictions for the missing entries
    cs_pred = np.zeros(len(miss_v))
    for i in range(len(miss_v)):
        r_idx = t_to_row[miss_t[i]]
        c_idx = sid_to_col[miss_sids[i]]
        cs_pred[i] = X_hat[r_idx, c_idx]

    time_cs = time.time() - t0
    ne_cs = _ne(miss_v, cs_pred)

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
    
    return ne_cs, time_cs, ne_hybrid, time_hybrid


if __name__ == "__main__":
    for ds_size in DATASET_SIZES:
        for pollutant in POLLUTANTS:
            print(f"\n{'='*60}")
            print(f"  Pollutant: {pollutant}  |  Dataset Size: {ds_size}")
            print(f"{'='*60}")

            try:
                xy, t, v, sids = load_seoul_dataset(
                    summary_csv="Seoul_dataset_2017.csv",
                    pollutant=pollutant,
                    n_rows=ds_size
                )
            except Exception as e:
                print(f"  ERROR loading {pollutant}: {e}")
                continue

            print(f"{'Rate':>5} | {'CS NE':>10} | {'CS t(s)':>10} | {'Hybrid NE':>10} | {'Hybrid t(s)':>11}")
            print("-" * 60)

            for r in MISSING_RATES:
                err_cs, t_cs, err_hyb, t_hyb = [], [], [], []
                for s in range(N_REPEATS):
                    ecs, tcs, eh, th = run_missing_rate_test(xy, t, v, sids, r, seed=42+s)
                    err_cs.append(ecs)
                    t_cs.append(tcs)
                    err_hyb.append(eh)
                    t_hyb.append(th)

                print(f"{r:>5.0%} | {np.mean(err_cs):>10.4f} | {np.mean(t_cs):>10.2f} | {np.mean(err_hyb):>10.4f} | {np.mean(t_hyb):>11.2f}")

    print(f"\n{'='*60}")
    print("  CS vs Hybrid tests completed.")
    print(f"{'='*60}")
