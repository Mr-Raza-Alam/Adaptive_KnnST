import numpy as np
import time
from make_demo_dataset import load_real_station_coords, synthetic_pollutant_field, real_pollutant_field, REPORT_DATES
from ec_knn_st import STData, knn_st_predict, ec_knn_st_v2_predict, ec_knn_st_modified_predict, _ne

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
    pred_base = np.array([knn_st_predict(obs, x, tt) for x, tt in zip(miss_xy, miss_t)])
    time_base = time.time() - t0
    ne_base = _ne(miss_v, pred_base)

    t0 = time.time()
    pred_v2, n_eval_v2 = ec_knn_st_v2_predict(obs, miss_xy, miss_t, missing_rate, seed=seed)
    time_v2 = time.time() - t0
    ne_v2 = _ne(miss_v, pred_v2)

    t0 = time.time()
    pred_mod, n_eval_mod = ec_knn_st_modified_predict(obs, miss_xy, miss_t, missing_rate, seed=seed)
    time_mod = time.time() - t0
    ne_mod = _ne(miss_v, pred_mod)

    return dict(
        missing_rate=missing_rate,
        ne_baseline=ne_base, time_baseline=time_base,
        ne_v2=ne_v2, time_v2=time_v2, eval_v2=n_eval_v2,
        ne_modified=ne_mod, time_modified=time_mod, eval_modified=n_eval_mod,
    )


def run_full_sweep(xy, t, v, label, n_repeats=5):
    rates = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]
    print(f"\n=== {label} ===")
    print(f"{'Rate':>6} | {'NE base':>9} | {'NE v2':>9} | {'NE mod':>9} | "
          f"{'t_v2(s)':>8} | {'t_mod(s)':>8} | {'eval_v2':>8} | {'eval_mod':>8}")
    print("-" * 90)
    for r in rates:
        results = [run_at_missing_rate(xy, t, v, r, seed=s) for s in range(n_repeats)]
        ne_base = np.mean([res['ne_baseline'] for res in results])
        ne_v2 = np.mean([res['ne_v2'] for res in results])
        ne_mod = np.mean([res['ne_modified'] for res in results])
        t_v2 = np.mean([res['time_v2'] for res in results])
        t_mod = np.mean([res['time_modified'] for res in results])
        ev2 = results[0]['eval_v2']
        emod = int(np.mean([res['eval_modified'] for res in results]))
        print(f"{r:>6.0%} | {ne_base:>9.4f} | {ne_v2:>9.4f} | {ne_mod:>9.4f} | "
              f"{t_v2:>8.2f} | {t_mod:>8.2f} | {ev2:>8d} | {emod:>8d}")


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
