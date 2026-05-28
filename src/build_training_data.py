"""Build seasonal multi-variate training data from ERA5 for ML interpolation.

Loads 3 years of ERA5 (2022/2023/2024), unifies coordinate names,
tags each time step by season, and generates IDW→target pairs with
realistic station observation noise. Precipitation (tp) is loaded
from a separate file and time-aligned with the instantaneous variables.

For each ERA5 time step:
  1. Interpolate all 6 ERA5 variables to 0.05° target grid (bicubic)
  2. Extract values at station locations + add observation noise
  3. Generate IDW first-guess from (noisy) station values
  4. Store (IDW_input, ERA5_target) pairs, grouped by season

Output per season (spring/summer/autumn/winter):
  data/training/{season}/X_train.npy  — IDW input  (N, 7, H, W)
  data/training/{season}/Y_train.npy  — ERA5 target (N, 6, H, W)
  data/training/{season}/X_val.npy / Y_val.npy
"""

import json
import logging
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build-training-data")

import sys
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import YANGTZE_DELTA_EXTENT, DEFAULT_GRID_RES
from src.interpolation import make_grid

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ERA5_FILES = {
    2022: ROOT / "data" / "era5" / "era5_yangtze_delta.nc",
    2023: ROOT / "data" / "era5" / "era5_2023" / "data_stream-oper_stepType-instant.nc",
    2024: ROOT / "data" / "era5" / "era5_2024" / "data_stream-oper_stepType-instant.nc",
}
ERA5_TP_PATH = ROOT / "data" / "era5" / "era5_tp.nc"
STATION_PATH = ROOT / "data" / "stations_yangtze_delta.json"
LAND_SEA_MASK_PATH = ROOT / "data" / "training" / "land_sea_mask.npy"
TRAIN_SPLIT = 0.8
IDW_RADIUS = 2.0

VARIABLES = ["t2m", "d2m", "u10", "v10", "msl", "tp"]
OBS_NOISE = {
    "t2m": 0.5,
    "d2m": 0.5,
    "u10": 1.0,
    "v10": 1.0,
    "msl": 0.5,
    "tp": 0.3,      # mm — rain gauge error
}

EXTENT = YANGTZE_DELTA_EXTENT
GRID_RES = DEFAULT_GRID_RES


class _PrecomputedIDW:
    """IDW interpolator with pre-computed neighbor lists.

    Reproduces the EXACT behaviour of src.interpolation.idw() — same KDTree,
    same query_ball_point, same _euclidean_deg distance, same weighting —
    but runs the spatial search only once. Each subsequent call just
    computes a weighted average with the same neighbours and weights,
    which is ~500x faster and mathematically identical.
    """

    def __init__(self, lons_s, lats_s, lon_g, lat_g, max_radius=2.0, power=2.0, min_neighbors=1):
        from scipy.spatial import KDTree
        cos_mean = np.cos(np.radians(np.mean(lats_s)))
        self._tree = KDTree(np.column_stack([lons_s * cos_mean, lats_s]))
        self._lons_s = lons_s
        self._lats_s = lats_s
        self._max_radius = max_radius
        self._power = power
        self._min_neighbors = min_neighbors
        self._cos_mean = cos_mean
        self._shape = lon_g.shape

        ny, nx = lon_g.shape
        self._neighbor_idxs = [[None] * nx for _ in range(ny)]
        self._neighbor_weights = [[None] * nx for _ in range(ny)]

        for j in range(ny):
            for i in range(nx):
                idxs = self._tree.query_ball_point(
                    [lon_g[j, i] * cos_mean, lat_g[j, i]], max_radius
                )
                if len(idxs) < min_neighbors:
                    continue
                dists = _euclidean_deg(
                    lons_s[idxs], lats_s[idxs], lon_g[j, i], lat_g[j, i]
                )
                dists = np.maximum(dists, 1e-6)
                w = dists ** (-power)
                self._neighbor_idxs[j][i] = np.array(idxs, dtype=np.intp)
                self._neighbor_weights[j][i] = w / w.sum()  # normalized

    def __call__(self, values):
        """Return (ny, nx) grid interpolated from station values."""
        ny, nx = self._shape
        grid = np.full((ny, nx), np.nan, dtype=np.float64)
        for j in range(ny):
            for i in range(nx):
                if self._neighbor_idxs[j][i] is None:
                    continue
                idxs = self._neighbor_idxs[j][i]
                w = self._neighbor_weights[j][i]
                grid[j, i] = (values[idxs] * w).sum()
        return grid


def _euclidean_deg(lon1, lat1, lon2, lat2):
    """Exact copy of interpolation._euclidean_deg."""
    dx = (lon2 - lon1) * np.cos(np.radians((lat1 + lat2) / 2))
    dy = lat2 - lat1
    return np.sqrt(dx ** 2 + dy ** 2)


# Coordinate name mappings for each year
COORD_MAP = {
    2022: {"time": "time", "lat": "lat", "lon": "lon"},
    2023: {"time": "valid_time", "lat": "latitude", "lon": "longitude"},
    2024: {"time": "valid_time", "lat": "latitude", "lon": "longitude"},
}

SEASON_MONTHS = {
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "autumn": [9, 10, 11],
    "winter": [12, 1, 2],
}


def _get_season(month: int) -> str:
    for name, months in SEASON_MONTHS.items():
        if month in months:
            return name
    return "winter"


def load_stations():
    with open(STATION_PATH, encoding="utf-8") as f:
        stations = json.load(f)
    lons = np.array([s["lon"] for s in stations])
    lats = np.array([s["lat"] for s in stations])
    logger.info(f"Loaded {len(stations)} stations")
    return lons, lats


def _load_era5_with_tp(year: int, tp_ds: xr.Dataset) -> xr.Dataset:
    """Load ERA5 instantaneous vars for a year, merge with tp from combined file."""
    path = ERA5_FILES[year]
    logger.info(f"Loading {year} ERA5: {path}")
    ds = xr.open_dataset(path)
    cmap = COORD_MAP[year]
    ds = ds.rename({cmap["time"]: "time", cmap["lat"]: "lat", cmap["lon"]: "lon"})

    # Keep only the 5 instantaneous variables (tp comes from separate file)
    inst_vars = [v for v in VARIABLES if v != "tp"]
    ds = ds[inst_vars]

    # Slice matching year from precipitation file
    tp_slice = tp_ds.sel(valid_time=str(year))
    tp_slice = tp_slice.rename({"valid_time": "time", "latitude": "lat", "longitude": "lon"})

    # Merge tp into the dataset
    ds["tp"] = tp_slice["tp"]
    logger.info(f"  {year}: merged tp ({ds.sizes['time']} steps, {len(ds.data_vars)} vars)")
    return ds


def build_dataset():
    """Build seasonal training data from all ERA5 years, now including tp."""
    lons_s, lats_s = load_stations()
    lon_g, lat_g = make_grid(EXTENT, GRID_RES)
    ny, nx = lon_g.shape
    logger.info(f"Target grid: {ny}×{nx} = {ny * nx} points")

    lsm = np.load(LAND_SEA_MASK_PATH).astype(np.float32)
    logger.info(f"Land-sea mask: shape={lsm.shape}, land frac={lsm.mean():.1%}")

    # Load precipitation data (all 3 years in one file)
    logger.info(f"Loading precipitation: {ERA5_TP_PATH}")
    tp_ds = xr.open_dataset(ERA5_TP_PATH)
    logger.info(f"  tp: {tp_ds.sizes}")

    lon1d = lon_g[0, :]
    lat1d = lat_g[:, 0]

    # Pre-compute IDW neighbour lists (mathematically identical to original idw())
    idw_interp = _PrecomputedIDW(lons_s, lats_s, lon_g, lat_g,
                                 max_radius=IDW_RADIUS, power=2.0, min_neighbors=1)
    logger.info("Pre-computed IDW neighbour indices and weights")

    # Collect samples per season
    season_data = {s: {"X": [], "Y": []} for s in SEASON_MONTHS}

    total_samples = 0

    for year in sorted(ERA5_FILES):
        ds = _load_era5_with_tp(year, tp_ds)
        time_coord = ds["time"]
        n_times = ds.sizes["time"]
        logger.info(f"  {year}: {n_times} time steps, "
                    f"{time_coord.values[0]} → {time_coord.values[-1]}")

        for t_idx in range(n_times):
            if (t_idx + 1) % 500 == 0:
                logger.info(f"    {year} processing {t_idx + 1}/{n_times} ...")

            era5_slice = ds.isel(time=t_idx)

            # Interpolate all variables to fine grid
            target = era5_slice.interp(
                lon=lon1d, lat=lat1d,
                method="cubic", kwargs={"fill_value": "extrapolate"}
            )

            if np.isnan(target["t2m"].values).any():
                continue

            # Skip if precipitation is NaN
            if np.isnan(target["tp"].values).all():
                continue

            # Determine season from timestamp
            dt = time_coord.values[t_idx]
            month = dt.astype("datetime64[M]").astype(int) % 12 + 1
            season = _get_season(month)

            idw_channels = []
            target_channels = []
            rng = np.random.default_rng(t_idx + year * 10000)

            for v in VARIABLES:
                target_field = target[v].values.astype(np.float32)

                # Convert tp from metres to mm, clip interpolation artifacts
                if v == "tp":
                    target_field = np.maximum(target_field * 1000.0, 0)

                interp = RegularGridInterpolator(
                    (lat1d, lon1d), target_field,
                    bounds_error=False, fill_value=None
                )
                station_vals = np.array([interp([lat, lon])[0]
                                         for lon, lat in zip(lons_s, lats_s)])
                station_vals += rng.normal(0, OBS_NOISE[v], len(station_vals))

                # Precipitation can't be negative
                if v == "tp":
                    station_vals = np.maximum(station_vals, 0)

                idw_field = idw_interp(station_vals)
                nan_mask = np.isnan(idw_field)
                if nan_mask.any():
                    idw_field[nan_mask] = np.nanmean(station_vals)

                idw_channels.append(idw_field.astype(np.float32))
                target_channels.append(target_field)

            idw_channels.append(lsm)
            season_data[season]["X"].append(np.stack(idw_channels, axis=0))
            season_data[season]["Y"].append(np.stack(target_channels, axis=0))
            total_samples += 1

        ds.close()

    tp_ds.close()
    logger.info(f"Total valid samples: {total_samples}")

    # Split and save per season
    base_dir = ROOT / "data" / "training"
    grand_total = 0

    for season in SEASON_MONTHS:
        X_all = np.stack(season_data[season]["X"], axis=0)
        Y_all = np.stack(season_data[season]["Y"], axis=0)
        n = len(X_all)

        if n == 0:
            logger.warning(f"  {season}: 0 samples, skipping")
            continue

        n_train = int(n * TRAIN_SPLIT)
        X_train, X_val = X_all[:n_train], X_all[n_train:]
        Y_train, Y_val = Y_all[:n_train], Y_all[n_train:]

        out_dir = base_dir / season
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "X_train.npy", X_train)
        np.save(out_dir / "Y_train.npy", Y_train)
        np.save(out_dir / "X_val.npy", X_val)
        np.save(out_dir / "Y_val.npy", Y_val)

        mb = sum(p.stat().st_size for p in out_dir.glob("*.npy")) / 1e6
        logger.info(f"  {season}: train={X_train.shape} val={X_val.shape} ({mb:.0f} MB)")
        grand_total += mb

    logger.info(f"Total saved: {grand_total:.0f} MB")


if __name__ == "__main__":
    build_dataset()
