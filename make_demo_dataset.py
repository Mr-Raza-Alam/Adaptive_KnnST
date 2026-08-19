import numpy as np
import pandas as pd

def load_real_station_coords(path="station_info.csv"):
    df = pd.read_csv(path)
    lat = df["Latitude"].values
    lon = df["Longitude"].values
    # normalize to [0,1] same way the report does (Sec 5.3)
    lat_n = (lat - lat.min()) / (lat.max() - lat.min())
    lon_n = (lon - lon.min()) / (lon.max() - lon.min())
    xy = np.stack([lon_n, lat_n], axis=1)
    return xy, df["Station name(district)"].values


def synthetic_pollutant_field(xy, n_hours=24, seed=42, pollutant="PM2.5"):
    """
    A smooth-ish spatiotemporal signal with diurnal pattern + spatial gradient +
    noise, loosely mimicking the qualitative shape described in the report
    (diurnal peaks for PM2.5, spatial clustering for PM10, etc.) -- but this is
    a STAND-IN signal, not real Seoul measurements.
    """
    rng = np.random.default_rng(seed)
    n_stations = xy.shape[0]
    t = np.arange(n_hours) / max(n_hours - 1, 1)

    # spatial gradient (e.g. more traffic-dense center -> higher baseline)
    center = xy.mean(axis=0)
    spatial_base = 40 + 30 * np.exp(-5 * ((xy - center) ** 2).sum(axis=1))

    # diurnal component (rush-hour peaks around t=0.33 and t=0.75)
    diurnal = 15 * (np.exp(-((t - 0.33) ** 2) / 0.01) + np.exp(-((t - 0.75) ** 2) / 0.01))

    records = []
    for si in range(n_stations):
        for ti in range(n_hours):
            val = spatial_base[si] + diurnal[ti] + rng.normal(0, 3.0)
            records.append((xy[si, 0], xy[si, 1], t[ti], max(val, 1.0)))
    arr = np.array(records)
    return arr[:, 0:2], arr[:, 2], arr[:, 3]  # xy, t, v


if __name__ == "__main__":
    xy, names = load_real_station_coords()
    print(f"Loaded {len(xy)} REAL Seoul station coordinates.")
    sxy, st, sv = synthetic_pollutant_field(xy)
    print(f"Generated {len(sv)} SYNTHETIC (station x hour) readings for demo purposes.")


def real_pollutant_field(summary_csv="Measurement_summary.csv",
                          date="2017-01-01", pollutant="PM2.5"):
    """
    Loads REAL Seoul measurements for a single day (24 hourly readings across
    all stations) from Measurement_summary.csv (bappekim/air-pollution-in-seoul).

    Known schema: 'Measurement date', 'Station code', 'Latitude', 'Longitude',
    'SO2','NO2','O3','CO','PM10','PM2.5'

    -1 values in this dataset mark sensor error/missing readings (per the
    dataset's own documentation) and are dropped before use.
    """
    df = pd.read_csv(summary_csv, parse_dates=["Measurement date"])

    day_mask = df["Measurement date"].dt.strftime("%Y-%m-%d") == date
    day_df = df.loc[day_mask].copy()
    if day_df.empty:
        raise ValueError(f"No rows found for date={date}. Check the date exists "
                          f"in the file (format should be YYYY-MM-DD).")

    # drop sensor-error / missing readings (-1 sentinel)
    day_df = day_df[day_df[pollutant] > 0]

    hour = day_df["Measurement date"].dt.hour.values.astype(float)
    lat = day_df["Latitude"].values.astype(float)
    lon = day_df["Longitude"].values.astype(float)
    val = day_df[pollutant].values.astype(float)

    # normalize exactly as Section 5.3 of the report describes: each to [0,1]
    lat_n = (lat - lat.min()) / (lat.max() - lat.min() + 1e-9)
    lon_n = (lon - lon.min()) / (lon.max() - lon.min() + 1e-9)
    t_n = hour / 23.0  # hour index 0-23 -> [0,1]

    xy = np.stack([lon_n, lat_n], axis=1)
    print(f"Loaded {len(val)} REAL readings for {pollutant} on {date} "
          f"from {day_df['Station code'].nunique()} stations.")
    return xy, t_n, val


REPORT_DATES = ["2017-01-01", "2017-04-15", "2017-07-20", "2017-10-10", "2018-03-05"]
