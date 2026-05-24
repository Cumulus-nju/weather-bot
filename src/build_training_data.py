"""Build multi-variate training data from ERA5 for ML interpolation.

For each ERA5 time step:
  1. Interpolate all 5 ERA5 variables to 0.05° target grid (bicubic)
  2. Extract values at station locations + add realistic observation noise
  3. Generate IDW first-guess from (noisy) station values for each variable
  4. Store (IDW_input, ERA5_target) pairs — shape (N, 5, H, W)

Channels: 0=t2m, 1=d2m, 2=u10, 3=v10, 4=msl
"""

import json
import logging
from pathlib import Path
import sys

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build-training-data")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import YANGTZE_DELTA_EXTENT, DEFAULT_GRID_RES
from src.interpolation import make_grid, idw

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ERA5_PATH = ROOT / "data" / "era5" / "era5_yangtze_delta.nc"
STATION_PATH = ROOT / "data" / "stations_yangtze_delta.json"
OUTPUT_DIR = ROOT / "data" / "training"
LAND_SEA_MASK_PATH = OUTPUT_DIR / "land_sea_mask.npy"
TRAIN_SPLIT = 0.8
IDW_RADIUS = 2.0

# Variables and channel order
VARIABLES = ["t2m", "d2m", "u10", "v10", "msl"]
# Realistic station observation noise (1-sigma)
OBS_NOISE = {
    "t2m": 0.5,    # °C
    "d2m": 0.5,    # °C
    "u10": 1.0,    # m/s
    "v10": 1.0,    # m/s
    "msl": 0.5,    # hPa
}

EXTENT = YANGTZE_DELTA_EXTENT
GRID_RES = DEFAULT_GRID_RES


def load_stations():
    with open(STATION_PATH, encoding="utf-8") as f:
        stations = json.load(f)
    lons = np.array([s["lon"] for s in stations])
    lats = np.array([s["lat"] for s in stations])
    logger.info(f"Loaded {len(stations)} stations")
    return lons, lats


def build_dataset():
    logger.info(f"Loading ERA5: {ERA5_PATH}")
    ds = xr.open_dataset(ERA5_PATH)
    n_times = ds.sizes["time"]
    logger.info(f"Time steps: {n_times}, variables: {VARIABLES}")

    lons_s, lats_s = load_stations()
    lon_g, lat_g = make_grid(EXTENT, GRID_RES)
    ny, nx = lon_g.shape
    logger.info(f"Target grid: {ny}×{nx} = {ny * nx} points")

    # Load land-sea mask (static, 0=sea, 1=land)
    lsm = np.load(LAND_SEA_MASK_PATH).astype(np.float32)
    logger.info(f"Land-sea mask loaded: shape={lsm.shape}, land frac={lsm.mean():.1%}")

    lon1d = lon_g[0, :]
    lat1d = lat_g[:, 0]

    X_list, Y_list = [], []
    n_vars = len(VARIABLES)

    for t_idx in range(n_times):
        if (t_idx + 1) % 200 == 0:
            logger.info(f"  Processing {t_idx + 1}/{n_times} ...")

        era5_slice = ds[VARIABLES].isel(time=t_idx)

        # Interpolate all variables to fine grid at once
        target = era5_slice.interp(
            lon=lon1d, lat=lat1d,
            method="cubic", kwargs={"fill_value": "extrapolate"}
        )
        # target shape: (5, H, W) after converting from xarray

        # Check for NaN in t2m (primary variable)
        if np.isnan(target["t2m"].values).any():
            continue

        idw_channels = []
        target_channels = []
        rng = np.random.default_rng(t_idx)

        for v in VARIABLES:
            target_field = target[v].values.astype(np.float32)

            # Sample station values from target + add noise
            interp = RegularGridInterpolator(
                (lat1d, lon1d), target_field,
                bounds_error=False, fill_value=None
            )
            station_vals = np.array([interp([lat, lon])[0]
                                     for lon, lat in zip(lons_s, lats_s)])
            station_vals += rng.normal(0, OBS_NOISE[v], len(station_vals))

            # IDW first-guess
            idw_field = idw(lons_s, lats_s, station_vals, lon_g, lat_g,
                            power=2.0, min_neighbors=1, max_radius=IDW_RADIUS)
            nan_mask = np.isnan(idw_field)
            if nan_mask.any():
                idw_field[nan_mask] = np.nanmean(station_vals)

            idw_channels.append(idw_field.astype(np.float32))
            target_channels.append(target_field)

        idw_channels.append(lsm)  # 6th channel: land-sea mask (static)
        X_list.append(np.stack(idw_channels, axis=0))   # (6, H, W)
        Y_list.append(np.stack(target_channels, axis=0))  # (5, H, W)

    # --- Concatenate and split ---
    X = np.stack(X_list, axis=0)  # (N, 5, H, W)
    Y = np.stack(Y_list, axis=0)  # (N, 5, H, W)
    logger.info(f"Valid samples: {len(X_list)} / {n_times}")

    n_train = int(len(X) * TRAIN_SPLIT)
    X_train, X_val = X[:n_train], X[n_train:]
    Y_train, Y_val = Y[:n_train], Y[n_train:]

    # --- Save ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "X_train.npy", X_train)
    np.save(OUTPUT_DIR / "Y_train.npy", Y_train)
    np.save(OUTPUT_DIR / "X_val.npy", X_val)
    np.save(OUTPUT_DIR / "Y_val.npy", Y_val)

    logger.info(f"Train: X={X_train.shape} Y={Y_train.shape}")
    logger.info(f"Val:   X={X_val.shape} Y={Y_val.shape}")
    total_mb = sum(p.stat().st_size for p in OUTPUT_DIR.glob("*.npy")) / 1e6
    logger.info(f"Total: {total_mb:.1f} MB saved to {OUTPUT_DIR}")
    logger.info(f"Channels: 0=t2m 1=d2m 2=u10 3=v10 4=msl 5=land_sea_mask")


if __name__ == "__main__":
    build_dataset()
