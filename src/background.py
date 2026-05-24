"""Background field fetcher — ERA5 / GFS with local caching.

For the bot use-case, we wrap the heavy downloads with:
  - File-based cache in data/
  - Graceful fallback (returns None if unavailable)
  - No blocking of the main pipeline
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from config import DATA_DIR, BACKGROUND_SOURCE, BACKGROUND_CACHE_TTL

logger = logging.getLogger("weather-bot.background")

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(label: str) -> Path:
    return DATA_DIR / f"bg_{label}.nc"


def _cache_valid(path: Path) -> bool:
    if not path.exists():
        return False
    import time
    return (time.time() - path.stat().st_mtime) < BACKGROUND_CACHE_TTL


# ---------------------------------------------------------------------------
# ERA5-Land via CDS
# ---------------------------------------------------------------------------

def get_era5_background(variable: str, lon_g, lat_g) -> np.ndarray | None:
    """Get 2D background field from ERA5-Land, interpolated to target grid.

    Uses a 5-day-old analysis since ERA5-Land real-time has a delay.
    """
    from datetime import datetime, timedelta

    try:
        import cdsapi
    except ImportError:
        logger.warning("cdsapi not installed — ERA5 background unavailable")
        return None

    # ERA5-Land: use 5-day-old data (the latest generally available)
    date = (datetime.utcnow() - timedelta(days=5)).strftime("%Y-%m-%d")
    time_str = "12:00"
    cache_file = _cache_path(f"era5land_{variable}_{date.replace('-','')}_{time_str.replace(':','')}")

    if _cache_valid(cache_file):
        logger.info(f"ERA5 cache hit: {cache_file}")
        ds = xr.open_dataset(cache_file)
        return _interp_to_grid(ds, variable, lon_g, lat_g)

    var_cds = _var_era5(variable)
    if var_cds is None:
        return None

    lon_min, lon_max = float(lon_g.min()), float(lon_g.max())
    lat_min, lat_max = float(lat_g.min()), float(lat_g.max())

    try:
        client = cdsapi.Client(quiet=True)
        request = {
            "product_type": "reanalysis",
            "variable": var_cds,
            "year": date[:4],
            "month": date[5:7],
            "day": date[8:10],
            "time": [time_str],
            "area": [lat_max, lon_min, lat_min, lon_max],
            "format": "netcdf",
        }
        client.retrieve("reanalysis-era5-land", request, str(cache_file))
        ds = xr.open_dataset(cache_file)
        logger.info(f"ERA5 downloaded: {cache_file}")
        return _interp_to_grid(ds, variable, lon_g, lat_g)
    except Exception as e:
        logger.warning(f"ERA5 download failed: {e}")
        # Try a 6-day-old cache file as fallback
        date2 = (datetime.utcnow() - timedelta(days=6)).strftime("%Y-%m-%d")
        cache_file2 = _cache_path(f"era5land_{variable}_{date2.replace('-','')}_{time_str.replace(':','')}")
        if cache_file2.exists():
            ds = xr.open_dataset(cache_file2)
            return _interp_to_grid(ds, variable, lon_g, lat_g)
        return None


# ---------------------------------------------------------------------------
# GFS via NOMADS
# ---------------------------------------------------------------------------

def get_gfs_background(variable: str, lon_g, lat_g) -> np.ndarray | None:
    """Get GFS analysis background field."""
    from datetime import datetime

    # Latest synoptic hour
    now = datetime.utcnow()
    # GFS analysis at 00/06/12/18Z
    hours = [0, 6, 12, 18]
    latest = max(h for h in hours if h <= now.hour + (now.hour < 6 and 18 or 0))  # approximation
    if now.hour < 6:
        latest = 18  # previous day

    date_str = now.strftime("%Y%m%d")
    hour_str = f"{latest:02d}"
    cache_file = _cache_path(f"gfs_{date_str}_{hour_str}")

    if _cache_valid(cache_file):
        logger.info(f"GFS cache hit: {cache_file}")
        ds = xr.open_dataset(cache_file)
        return _interp_to_grid(ds, variable, lon_g, lat_g)

    var_gfs = _var_gfs(variable)
    if var_gfs is None:
        return None

    lon_min, lon_max = int(lon_g.min()), int(lon_g.max())
    lat_min, lat_max = int(lat_g.min()), int(lat_g.max())

    try:
        import requests
        import tempfile

        base_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
        params = {
            "file": f"gfs.t{hour_str}z.pgrb2.0p25.anl",
            "lev_2_m_above_ground": "on",
            "lev_10_m_above_ground": "on",
            "var_TMP": "on", "var_RH": "on", "var_UGRD": "on", "var_VGRD": "on",
            "var_PRES": "on",
            "subregion": "on",
            "leftlon": lon_min, "rightlon": lon_max,
            "toplat": lat_max, "bottomlat": lat_min,
            "dir": f"/gfs.{date_str}/{hour_str}/atmos",
        }
        resp = requests.get(base_url, params=params, stream=True, timeout=60)
        resp.raise_for_status()

        tmp = Path(tempfile.gettempdir()) / f"gfs_tmp_{date_str}_{hour_str}.grib2"
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)

        ds = xr.open_dataset(tmp, engine="cfgrib",
                             backend_kwargs={"filter_by_keys": {"typeOfLevel": "heightAboveGround"}})
        ds.to_netcdf(cache_file)
        tmp.unlink(missing_ok=True)
        logger.info(f"GFS downloaded: {cache_file}")
        return _interp_to_grid(ds, variable, lon_g, lat_g)
    except Exception as e:
        logger.warning(f"GFS download failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Public API used by pipeline
# ---------------------------------------------------------------------------

def try_get_background(lon_g, lat_g, variable: str) -> np.ndarray | None:
    """Best-effort background field fetch. Returns None if unavailable."""
    if BACKGROUND_SOURCE == "none":
        return None

    if variable in ("precipitation", "comprehensive"):
        return None  # precipitation is too discontinuous for OI

    if BACKGROUND_SOURCE == "era5":
        return get_era5_background(variable, lon_g, lat_g)
    elif BACKGROUND_SOURCE == "gfs":
        return get_gfs_background(variable, lon_g, lat_g)

    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _var_era5(var: str) -> str | None:
    return {
        "temperature": "2m_temperature",
        "humidity":    "2m_dewpoint_temperature",
        "pressure":    "surface_pressure",
        "wind":        "10m_u_component_of_wind",  # needs both u/v
    }.get(var)


def _var_gfs(var: str) -> str | None:
    return {
        "temperature": "t2m",
        "humidity":    "r2",
        "pressure":    "pres",
        "wind":        "u10",
    }.get(var)


def _interp_to_grid(ds, variable: str, lon_g, lat_g) -> np.ndarray | None:
    """Bilinearly interpolate xarray field to target grid."""
    try:
        lon1d = lon_g[0, :]
        lat1d = lat_g[:, 0]

        # Map our variable name → dataset variable
        var_era5_map = {
            "temperature": "t2m",
            "humidity":    "d2m",
            "pressure":    "sp",
            "wind":        "u10",
        }

        # Try common CF names
        ds_var = None
        for candidate in [var_era5_map.get(variable, variable),
                          _var_era5(variable), _var_gfs(variable)]:
            if candidate and candidate in ds:
                ds_var = candidate
                break
            if candidate and candidate in ds.data_vars:
                ds_var = candidate
                break

        # Brute-force: use first 2D data variable
        if ds_var is None:
            for name, da in ds.data_vars.items():
                if "lat" in da.dims and "lon" in da.dims:
                    ds_var = name
                    break

        if ds_var is None:
            logger.warning("No plottable variable found in dataset")
            return None

        # Rename coords to standard names for interpolation
        ds_renamed = ds
        # cartopy-typical coordinate names
        for lname in ["latitude", "lat", "LAT", "Latitude"]:
            if lname in ds.coords:
                break
        for loname in ["longitude", "lon", "LON", "Longitude"]:
            if loname in ds.coords:
                break

        interp = ds[ds_var].interp(
            latitude=lat1d, longitude=lon1d, method="linear",
            kwargs={"fill_value": None}
        )
        bg = interp.values

        # dewpoint → relative humidity conversion (simplified)
        if variable == "humidity" and "dewpoint" in str(ds[ds_var].attrs.get("long_name", "")).lower():
            temp_interp = ds["t2m"].interp(latitude=lat1d, longitude=lon1d)
            t = temp_interp.values
            td = bg
            bg = 100 * np.exp(17.625 * (td - t) / (243.04 + td - t - 1e-6))

        return bg
    except Exception as e:
        logger.warning(f"Background interpolation failed: {e}")
        return None
