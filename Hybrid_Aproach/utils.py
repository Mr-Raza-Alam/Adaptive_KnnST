import numpy as np
import pandas as pd
from dataclasses import dataclass
import os

@dataclass
class STData:
    xy: np.ndarray      # (N,2) normalized spatial coords
    t: np.ndarray       # (N,)  normalized time coords
    v: np.ndarray       # (N,)  values


def spatiotemporal_dist(xy, t, xy_q, t_q, alpha):
    """Exact d_ST(x,p) = alpha*dS + (1-alpha)*dT"""
    if xy.ndim == 1:
        xy = xy.reshape(1, 2)
        t = np.array([t])
    ds = np.sqrt(((xy - xy_q) ** 2).sum(axis=1))
    dt = np.abs(t - t_q)
    return alpha * ds + (1 - alpha) * dt


def _ne(actual, pred):
    """Normalized Error metric for evaluating imputation."""
    actual, pred = np.asarray(actual), np.asarray(pred)
    denom = np.sum(np.abs(actual))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(actual - pred)) / denom)


def load_seoul_dataset(summary_csv="Seoul_dataset_2017.csv", pollutant="PM2.5", n_rows=None, seed=42):
    """
    Loads REAL Seoul measurements across all stations.
    Returns xy, t, v, station_ids (all as numpy arrays).
    Station IDs are preserved so the hybrid algorithm can do
    station-aware temporal interpolation.
    """
    if not os.path.exists(summary_csv):
        raise FileNotFoundError(f"Could not find dataset: {summary_csv}")
        
    df = pd.read_csv(summary_csv, parse_dates=["Measurement date"])

    # Drop missing values / sensor errors marked as -1 or 0
    clean_df = df[df[pollutant] > 0].copy()
    
    # Optionally sample an exact number of rows for scaling tests
    if n_rows is not None and n_rows < len(clean_df):
        clean_df = clean_df.sample(n=n_rows, random_state=seed).copy()

    # Normalize time across the selected range
    time_series = clean_df["Measurement date"]
    t_seconds = (time_series - time_series.min()).dt.total_seconds().values
    
    lat = clean_df["Latitude"].values.astype(float)
    lon = clean_df["Longitude"].values.astype(float)
    val = clean_df[pollutant].values.astype(float)
    sids = clean_df["Station code"].values.astype(int)

    # Normalize space to [0,1]
    lat_n = (lat - lat.min()) / (lat.max() - lat.min() + 1e-9)
    lon_n = (lon - lon.min()) / (lon.max() - lon.min() + 1e-9)
    
    # Normalize time to [0,1]
    t_n = t_seconds / (t_seconds.max() + 1e-9)

    xy = np.stack([lon_n, lat_n], axis=1)
    print(f"Loaded {len(val)} REAL readings for {pollutant} "
          f"from {clean_df['Station code'].nunique()} stations.")
    return xy, t_n, val, sids
