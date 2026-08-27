import numpy as np
import time
from utils import load_seoul_month_dataset, STData, _ne
from hybrid_knn_st import hybrid_knn_st_predict

def run_missing_rate_test(xy, t, v, missing_rate, seed=42):
    rng = np.random.default_rng(seed)
    n = len(v)
    n_miss = int(n * missing_rate)
    
    # Randomly select missing indices
    miss_idx = rng.choice(n, size=n_miss, replace=False)
    mask = np.ones(n, dtype=bool)
    mask[miss_idx] = False

    obs = STData(xy[mask], t[mask], v[mask])
    miss_xy, miss_t, miss_v = xy[miss_idx], t[miss_idx], v[miss_idx]

    t0 = time.time()
    pred = hybrid_knn_st_predict(obs, miss_xy, miss_t, missing_rate=missing_rate, seed=seed)
    t_elapsed = time.time() - t0
    
    error = _ne(miss_v, pred)
    
    return error, t_elapsed

if __name__ == "__main__":
    print("=== Loading Dataset ===")
    try:
        # Load exactly one month from the 2017 dataset (~18,000 readings)
        xy, t, v = load_seoul_month_dataset(summary_csv="Seoul_dataset_2017.csv", month_str="2017-01", pollutant="PM2.5")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure 'Seoul_dataset_2017.csv' is in the Hybrid_Aproach folder.")
        exit(1)
        
    rates = [0.4, 0.6, 0.8, 0.9]
    n_repeats = 3  # Multiple repeats for robust timing/accuracy average
    
    print("\n=== Fast Hybrid Adaptive k-NN ST Performance Sweep ===")
    print(f"Dataset Size: {len(v)} rows")
    print(f"{'Missing Rate':>12} | {'Normalized Error':>18} | {'Avg Time (s)':>14}")
    print("-" * 52)
    
    for r in rates:
        errors = []
        times = []
        for s in range(n_repeats):
            err, t_elap = run_missing_rate_test(xy, t, v, r, seed=42+s)
            errors.append(err)
            times.append(t_elap)
            
        print(f"{r:>12.0%} | {np.mean(errors):>18.4f} | {np.mean(times):>14.2f}")
