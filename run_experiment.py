import numpy as np
import time
from make_demo_dataset import load_real_station_coords, synthetic_pollutant_field, real_pollutant_field, REPORT_DATES
from Ad_knn_st import STData, _ne, ad_knn_st_predict
from base_knn_st import VoxelGrid, knn_st_voxel_predict

def run_at_missing_rate(xy, t, v, missing_rate, seed=0):
    rng = np.random.default_rng(seed)
    n = len(v)
    n_miss = int(n * missing_rate)
    miss_idx = rng.choice(n, size=n_miss, replace=False)
    mask = np.ones(n, dtype=bool)
    mask[miss_idx] = False

    obs = STData(xy[mask], t[mask], v[mask])
    miss_xy, miss_t, miss_v = xy[miss_idx], t[miss_idx], v[miss_idx]

    t0 = time.time()
    vox_grid = VoxelGrid(obs)
    pred_vox = np.array([knn_st_voxel_predict(vox_grid, obs, x, tt) for x, tt in zip(miss_xy, miss_t)])
    time_vox = time.time() - t0
    ne_vox = _ne(miss_v, pred_vox)

    t0 = time.time()
    pred_ad, n_eval_ad = ad_knn_st_predict(obs, miss_xy, miss_t, missing_rate, seed=seed)
    time_ad = time.time() - t0
    ne_ad = _ne(miss_v, pred_ad)

    return dict(
        missing_rate=missing_rate,
        ne_vox=ne_vox, time_vox=time_vox,
        ne_ad=ne_ad, time_ad=time_ad, eval_ad=n_eval_ad,
    )


def run_full_sweep(xy, t, v, label, n_repeats=2):
    rates = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]
    print(f"\n=== {label} ===")
    print(f"{'Rate':>6} | {'NE vox':>9} | {'NE ad':>9} | {'t_vox(s)':>8} | {'t_ad(s)':>8} | {'eval_ad':>8}")
    print("-" * 75)
    for r in rates:
        results = [run_at_missing_rate(xy, t, v, r, seed=s) for s in range(n_repeats)]
        ne_vox = np.mean([res['ne_vox'] for res in results])
        ne_ad = np.mean([res['ne_ad'] for res in results])
        t_vox = np.mean([res['time_vox'] for res in results])
        t_ad = np.mean([res['time_ad'] for res in results])
        ead = int(np.mean([res['eval_ad'] for res in results]))
        print(f"{r:>6.0%} | {ne_vox:>9.4f} | {ne_ad:>9.4f} | {t_vox:>8.2f} | {t_ad:>8.2f} | {ead:>8d}")


if __name__ == "__main__":
    USE_REAL_DATA = True   # set False to fall back to the synthetic demo

    if USE_REAL_DATA:
        for date in REPORT_DATES:
            for pollutant in ["PM10", "PM2.5", "CO"]:
                xy, t, v = real_pollutant_field(
                    summary_csv="Seoul_dataset.csv",
                    date=date, pollutant=pollutant
                )
                run_full_sweep(xy, t, v, label=f"{pollutant} - {date}", n_repeats=5)
    else:
        xy, names = load_real_station_coords()
        sxy, st, sv = synthetic_pollutant_field(xy, n_hours=24, seed=42)
        run_full_sweep(sxy, st, sv, label="SYNTHETIC DEMO", n_repeats=1)
