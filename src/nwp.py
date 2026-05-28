"""NWP data pipeline: ECMWF IFS and GFS analysis / forecast.

Data flow:
  1. Determine latest available model cycle
  2. Download GRIB2 -> subset to China -> cache as netCDF
  3. Extract requested variable, apply unit conversion
  4. Plot with Cartopy (reuses src.plotter)

ECMWF: ecmwf-opendata (CC-BY-4.0, free since 2025-10)
GFS:   NOAA NOMADS GRIB filter with server-side BBOX subset
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from config import (
    CHINA_EXTENT,
    NWP_CACHE_DIR,
    NWP_CACHE_TTL,
    NWP_MAX_CACHE_AGE_DAYS,
    OUTPUT_DIR,
    OUTPUT_DPI,
)
from src.interpolation import make_grid
from src.plotter import plot_grid, plot_wind_barbs, plot_multi_panel

logger = logging.getLogger("weather-bot.nwp")

# ---------------------------------------------------------------------------
# cfgrib is not thread-safe; serialise all GRIB reads through this lock
# ---------------------------------------------------------------------------
_grib_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _dewpoint_to_rh(temp_k: np.ndarray, dewpoint_k: np.ndarray) -> np.ndarray:
    """Convert temperature (K) + dewpoint (K) -> relative humidity (%).

    Uses Magnus formula with Sonntag coefficients (same as pipeline.py inverse).
    """
    a, b = 17.62, 243.12
    temp_c = temp_k - 273.15
    dew_c = dewpoint_k - 273.15
    es = 6.112 * np.exp(a * temp_c / (temp_c + np.clip(b, 1e-6, None)))
    e = 6.112 * np.exp(a * dew_c / (dew_c + np.clip(b, 1e-6, None)))
    return np.clip(100.0 * e / np.maximum(es, 1e-6), 0, 100)


# Ensure cfgrib backend is registered before any xarray GRIB reads
try:
    import cfgrib  # noqa: F401
except ImportError:
    pass


def _subset_china(ds: "xr.Dataset") -> "xr.Dataset":
    """Subset an xarray Dataset to China extent [73-136E, 16-55N].

    Handles both 0-360 and -180-180 longitude conventions.
    """
    lon_min, lon_max, lat_min, lat_max = CHINA_EXTENT  # -180:180

    import xarray as xr

    # Determine longitude convention
    ds_lon_min = float(ds.longitude.min())
    ds_lon_max = float(ds.longitude.max())

    if ds_lon_max > 180:
        # 0:360 convention — convert China bbox to 0:360
        subset_lon_min, subset_lon_max = lon_min, lon_max
        if subset_lon_min < 0:
            subset_lon_min += 360
        if subset_lon_max < 0:
            subset_lon_max += 360
    else:
        subset_lon_min, subset_lon_max = lon_min, lon_max

    # lat may be stored descending (N->S) — handle both
    lat_ascending = float(ds.latitude[0]) < float(ds.latitude[-1])
    if lat_ascending:
        lat_slice = slice(float(lat_min), float(lat_max))
    else:
        lat_slice = slice(float(lat_max), float(lat_min))

    # Select region
    ds_sub = ds.sel(longitude=slice(subset_lon_min, subset_lon_max),
                    latitude=lat_slice)

    # Convert to -180:180 if needed
    if ds_lon_max > 180:
        lon_vals = ds_sub.longitude.values
        lon_vals[lon_vals > 180] -= 360
        ds_sub = ds_sub.assign_coords(longitude=lon_vals)
        ds_sub = ds_sub.sortby("longitude")

    return ds_sub


def _extract_2d(ds: "xr.Dataset", short_names: list[str]) -> np.ndarray:
    """Extract first matching 2D field from a cfgrib/xarray Dataset by GRIB shortName."""
    for vn in list(ds.data_vars):
        da = ds[vn]
        sn = da.attrs.get("GRIB_shortName", vn)
        if sn in short_names:
            arr = np.squeeze(da.values)
            if arr.ndim != 2:
                raise ValueError(
                    f"Expected 2D field for {sn}, got shape {da.values.shape} "
                    f"after squeeze. Dataset dims: {list(ds.dims)}"
                )
            return arr.astype(np.float32)

    # Fallback: try variable name directly
    for sn in short_names:
        if sn in ds.data_vars:
            arr = np.squeeze(ds[sn].values)
            if arr.ndim != 2:
                raise ValueError(
                    f"Expected 2D field for {sn}, got shape {ds[sn].values.shape} "
                    f"after squeeze."
                )
            return arr.astype(np.float32)

    raise KeyError(f"Variable not found: {short_names}. Available: {list(ds.data_vars)}")


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class NWPSource(ABC):
    """Abstract base for NWP data sources."""

    def __init__(self, name: str, cache_subdir: str):
        self.name = name
        self.cache_dir = NWP_CACHE_DIR / cache_subdir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Cycle tracking
    # ------------------------------------------------------------------

    @abstractmethod
    def get_latest_cycle(self) -> tuple[datetime, int]:
        """Return (date, hour_utc) of the most recent available model run."""
        ...

    # ------------------------------------------------------------------
    # Download (abstract)
    # ------------------------------------------------------------------

    @abstractmethod
    def download(self, date: datetime, hour: int, step: int) -> Path:
        """Download data for one model cycle and forecast step.

        Returns path to the cached netCDF file.
        """
        ...

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, date: datetime, hour: int, step: int) -> Path:
        label = f"{date.strftime('%Y%m%d')}_{hour:02d}z_f{step:03d}.nc"
        return self.cache_dir / label

    def is_cached(self, date: datetime, hour: int, step: int) -> bool:
        p = self._cache_path(date, hour, step)
        if not p.exists():
            return False
        age = time.time() - p.stat().st_mtime
        return age < NWP_CACHE_TTL

    def get_dataset(self, date: datetime, hour: int, step: int) -> "xr.Dataset":
        """Return an xarray Dataset for the given cycle+step, from cache or download."""
        import xarray as xr

        p = self._cache_path(date, hour, step)
        if self.is_cached(date, hour, step):
            logger.info(f"[{self.name}] Cache hit: {p.name}")
            with _grib_lock:
                return xr.open_dataset(p)

        logger.info(f"[{self.name}] Downloading: {date.strftime('%Y%m%d')}_{hour:02d}z f{step:03d}")
        p = self.download(date, hour, step)
        with _grib_lock:
            return xr.open_dataset(p)

    def cleanup(self, max_age_days: int = NWP_MAX_CACHE_AGE_DAYS):
        """Delete cached netCDF files older than *max_age_days*."""
        now = time.time()
        max_age_s = max_age_days * 86400
        deleted = 0
        for f in sorted(self.cache_dir.glob("*.nc")):
            try:
                if now - f.stat().st_mtime > max_age_s:
                    f.unlink()
                    deleted += 1
            except OSError:
                pass
        if deleted:
            logger.info(f"[{self.name}] Cleaned up {deleted} old cache file(s)")


# ---------------------------------------------------------------------------
# ECMWF IFS source (via ecmwf-opendata)
# ---------------------------------------------------------------------------

class ECMWFSource(NWPSource):

    # ECMWF grib2: shortNames as decoded by cfgrib (first) or ecmwf-opendata
    # We download: 2t, 10u, 10v, msl, tp, 2d
    VAR_SHORTNAMES = {
        "temperature":   ["t2m", "2t"],
        "precipitation": ["tp"],
        "wind_u":        ["u10", "10u"],
        "wind_v":        ["v10", "10v"],
        "pressure":      ["msl"],
        "dewpoint":      ["d2m", "2d"],
    }

    def __init__(self):
        super().__init__("ECMWF", "ecmwf")

    def get_latest_cycle(self) -> tuple[datetime, int]:
        """ECMWF IFS runs at 00z and 12z, available ~6h after run time."""
        now = datetime.now(timezone.utc)
        h = now.hour
        if h >= 18:
            return (now.date(), 12)
        elif h >= 6:
            return (now.date(), 0)
        else:
            return ((now - timedelta(days=1)).date(), 12)

    def download(self, date: datetime, hour: int, step: int) -> Path:
        import xarray as xr

        cache_path = self._cache_path(date, hour, step)
        tmp_grib = cache_path.with_suffix(".grib2")

        try:
            from ecmwf.opendata import Client

            client = Client(source="ecmwf")
            param_str = "2t/10u/10v/msl/tp/2d"

            logger.info(f"[ECMWF] Fetching {date.strftime('%Y%m%d')}_{hour:02d}z step={step} ...")
            client.retrieve(
                date=date.strftime("%Y-%m-%d"),
                time=hour,
                type="fc",
                step=step,
                param=param_str,
                target=str(tmp_grib),
            )

            logger.info(f"[ECMWF] Downloaded {tmp_grib.stat().st_size/1e6:.1f} MB, decoding ...")

            with _grib_lock:
                # cfgrib.open_datasets splits on hypercube (surface / heightAboveGround / meanSea)
                # compat='override' handles conflicting heightAboveGround coords (2m vs 10m)
                datasets = cfgrib.open_datasets(tmp_grib)
                if len(datasets) > 1:
                    ds = xr.merge(datasets, compat="override")
                elif len(datasets) == 1:
                    ds = datasets[0]
                else:
                    raise RuntimeError("No variables decoded from ECMWF GRIB2")

            # Subset to China
            ds = _subset_china(ds)

            # Save as netCDF
            ds.to_netcdf(cache_path)
            logger.info(f"[ECMWF] Cached → {cache_path.name} ({cache_path.stat().st_size/1e3:.0f} KB)")

            return cache_path

        finally:
            if tmp_grib.exists():
                tmp_grib.unlink()
            # Clean up cfgrib index files left alongside the grib
            for idx in tmp_grib.parent.glob(f"{tmp_grib.name}.*.idx"):
                idx.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# GFS source (via NOAA NOMADS GRIB filter)
# ---------------------------------------------------------------------------

class GFSSource(NWPSource):

    VAR_SHORTNAMES = {
        "temperature":   ["2t", "t2m"],
        "precipitation": ["tp"],
        "wind_u":        ["10u", "u10"],
        "wind_v":        ["10v", "v10"],
        "pressure":      ["prmsl", "msl", "mslma"],
        "humidity":      ["2r", "r2"],
    }

    # NOAA filter URL parameters
    _GFS_LEVELS = [
        "lev_2_m_above_ground=on",
        "lev_10_m_above_ground=on",
        "lev_mean_sea_level=on",
        "lev_surface=on",
    ]
    _GFS_VARS = [
        "var_TMP=on",
        "var_UGRD=on",
        "var_VGRD=on",
        "var_PRMSL=on",
        "var_RH=on",
        "var_APCP=on",
    ]

    def __init__(self):
        super().__init__("GFS", "gfs")

    def get_latest_cycle(self) -> tuple[datetime, int]:
        """GFS runs 4x daily: 00, 06, 12, 18z. Available ~4h after run.

        Pick the most recent available cycle.
        """
        now = datetime.now(timezone.utc)
        candidates = []
        for rh in [0, 6, 12, 18]:
            run_dt = now.replace(hour=rh, minute=0, second=0, microsecond=0)
            if now < run_dt:
                run_dt -= timedelta(days=1)
            available_at = run_dt + timedelta(hours=4)
            if now >= available_at:
                candidates.append((available_at, run_dt.date(), rh))

        if not candidates:
            yday = (now - timedelta(days=1)).date()
            return (yday, 18)

        candidates.sort(reverse=True)  # most recent first
        _, date, rh = candidates[0]
        return (date, rh)

    def download(self, date: datetime, hour: int, step: int) -> Path:
        import xarray as xr

        cache_path = self._cache_path(date, hour, step)
        tmp_grib = cache_path.with_suffix(".grib2")

        # Clean up stale .grib2 and .idx files from previous runs (cfgrib cache
        # confusion when old .idx is incompatible with a freshly-downloaded file)
        if tmp_grib.exists():
            tmp_grib.unlink()
        for idx in tmp_grib.parent.glob(f"{tmp_grib.name}.*.idx"):
            idx.unlink(missing_ok=True)

        date_str = date.strftime("%Y%m%d")
        dir_str = f"gfs.{date_str}/{hour:02d}/atmos"

        if step == 0:
            file_str = f"gfs.t{hour:02d}z.pgrb2.0p25.f000"
        else:
            file_str = f"gfs.t{hour:02d}z.pgrb2.0p25.f{step:03d}"

        url = (
            f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
            f"?file={file_str}"
            f"&{'&'.join(self._GFS_LEVELS)}"
            f"&{'&'.join(self._GFS_VARS)}"
            f"&subregion=&toplat={CHINA_EXTENT[3]}"
            f"&leftlon={CHINA_EXTENT[0]}"
            f"&rightlon={CHINA_EXTENT[1]}"
            f"&bottomlat={CHINA_EXTENT[2]}"
            f"&dir=%2F{dir_str}"
        )

        import httpx

        for attempt in range(1, 6):
            try:
                resp = httpx.get(url, timeout=120)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    # Verify GRIB2 magic bytes — NOAA may return HTML on error
                    if resp.content[:4] != b"GRIB":
                        raise RuntimeError(
                            f"Response is not GRIB2 (starts with {resp.content[:80]!r})"
                        )
                    tmp_grib.write_bytes(resp.content)
                    logger.info(f"[GFS] Downloaded {len(resp.content)/1e3:.0f} KB "
                                f"(attempt {attempt})")
                    break
                else:
                    logger.warning(f"[GFS] HTTP {resp.status_code}, size={len(resp.content)} "
                                   f"(attempt {attempt})")
            except Exception as e:
                logger.warning(f"[GFS] attempt {attempt}: {e}")

            if attempt < 5:
                time.sleep(10 * attempt)
        else:
            raise RuntimeError(f"GFS download failed after 5 attempts: {url}")

        # Decode GRIB2 -> subset -> netCDF
        logger.info(f"[GFS] Decoding GRIB2 ({tmp_grib.stat().st_size/1e6:.1f} MB) ...")
        try:
            with _grib_lock:
                datasets = cfgrib.open_datasets(tmp_grib)
                logger.info(f"[GFS] cfgrib returned {len(datasets)} hypercube(s)")
                for i, d in enumerate(datasets):
                    logger.info(f"[GFS]   hypercube[{i}]: {list(d.data_vars)} "
                                f"dims={dict(d.dims)}")
                if len(datasets) > 1:
                    ds = xr.merge(datasets, compat="override")
                elif len(datasets) == 1:
                    ds = datasets[0]
                else:
                    raise RuntimeError("No variables decoded from GFS GRIB2")

            logger.info(f"[GFS] Merged variables: {list(ds.data_vars)}")

            # GFS filter URL subsets server-side, but we may need lon conversion
            if ds.longitude.max() > 180:
                lon_vals = ds.longitude.values
                lon_vals[lon_vals > 180] -= 360
                ds = ds.assign_coords(longitude=lon_vals)
                ds = ds.sortby("longitude")

            ds.to_netcdf(cache_path)
            logger.info(f"[GFS] Cached → {cache_path.name} "
                        f"({cache_path.stat().st_size/1e3:.0f} KB)")

            return cache_path

        finally:
            if tmp_grib.exists():
                tmp_grib.unlink()
            for idx in tmp_grib.parent.glob(f"{tmp_grib.name}.*.idx"):
                idx.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# NWP Pipeline
# ---------------------------------------------------------------------------

_SOURCE_MAP = {
    "ecmwf": ECMWFSource,
    "ec":    ECMWFSource,
    "gfs":   GFSSource,
}

# Variables that default to forecast (no meaningful analysis field)
_FORECAST_DEFAULT_VARS = {"precipitation"}


class NWPPipeline:
    """Orchestrates NWP data fetching and plotting.

    Uses a single 0.25° grid covering China, matching the native resolution
    of both ECMWF open data and GFS.
    """

    def __init__(self):
        self._sources: dict[str, NWPSource] = {
            "ecmwf": ECMWFSource(),
            "gfs":   GFSSource(),
        }
        # Pre-build grid at 0.25° over China for plotting
        self.china_lon_g, self.china_lat_g = make_grid(CHINA_EXTENT, 0.25)
        self.extent = CHINA_EXTENT
        logger.info(f"NWPPipeline init: "
                    f"extent={CHINA_EXTENT}, grid={self.china_lon_g.shape}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, source: str, variable: str,
                 forecast_hour: int | None = None) -> str:
        """Run NWP pipeline and return path to the generated PNG.

        Args:
            source: "ecmwf" | "gfs"
            variable: "temperature" | "precipitation" | "wind" |
                      "pressure" | "humidity" | "comprehensive"
            forecast_hour: None=auto (0 for most, 24 for precip),
                           0=analysis, 24/48/72=forecast
        """
        t0 = time.perf_counter()
        src = self._sources.get(source.lower())
        if src is None:
            raise ValueError(f"Unknown source: {source}. Use: ecmwf, gfs")

        # Resolve forecast hour
        if forecast_hour is None:
            if variable in _FORECAST_DEFAULT_VARS:
                forecast_hour = 24
            else:
                forecast_hour = 0

        # Ensure precip never uses analysis (tp=0 meaningless)
        if variable == "precipitation" and forecast_hour == 0:
            logger.info(f"[{src.name}] Precipitation analysis is zero — "
                        f"falling back to 24h forecast")
            forecast_hour = 24

        # Get cycle + dataset
        date, hour = src.get_latest_cycle()
        logger.info(f"[{src.name}] Latest cycle: {date.strftime('%Y-%m-%d')} "
                    f"{hour:02d}z, step={forecast_hour}")

        src.cleanup()
        ds = src.get_dataset(date, hour, forecast_hour)

        # Build title with cycle time and valid time
        cycle_ts = f"{date.strftime('%Y-%m-%d')} {hour:02d}z"
        if forecast_hour == 0:
            mode_str = "分析场"
            time_str = cycle_ts
        else:
            mode_str = f"+{forecast_hour}h预报"
            valid_dt = datetime(date.year, date.month, date.day, hour,
                                tzinfo=timezone.utc) + timedelta(hours=forecast_hour)
            valid_ts = f"{valid_dt.strftime('%Y-%m-%d')} {valid_dt.hour:02d}z"
            time_str = f"起报 {cycle_ts}  |  有效 {valid_ts}"
        base_title = f"{'ECMWF IFS' if isinstance(src, ECMWFSource) else 'GFS'} {mode_str}"

        # Dispatch
        if variable == "comprehensive":
            path = self._comprehensive(ds, src, base_title, time_str)
        elif variable == "wind":
            path = self._wind(ds, src, base_title, time_str)
        else:
            path = self._scalar(ds, src, variable, base_title, time_str)

        elapsed = time.perf_counter() - t0
        logger.info(f"NWP done: {source}/{variable}/f{forecast_hour} → {path} "
                    f"({elapsed:.1f}s)")
        return path

    # ------------------------------------------------------------------
    # Plot dispatch
    # ------------------------------------------------------------------

    def _scalar(self, ds, src: NWPSource, variable: str,
                base_title: str, time_str: str) -> str:
        lon_g, lat_g = self._grid_from_ds(ds)
        field = self._get_field(ds, src, variable)
        out_name = f"nwp_{src.name.lower()}_{variable}_{int(time.time())}.png"
        title = f"{base_title} {self._var_cn(variable)}\n{time_str}"

        return plot_grid(lon_g, lat_g, field,
                         title=title,
                         var_type=self._plot_var_type(variable),
                         extent=self._extent_from_ds(ds),
                         out_path=out_name,
                         dpi=OUTPUT_DPI)

    def _wind(self, ds, src: NWPSource, base_title: str,
              time_str: str) -> str:
        lon_g, lat_g = self._grid_from_ds(ds)
        u_field = self._get_field(ds, src, "wind_u")
        v_field = self._get_field(ds, src, "wind_v")
        speed = np.sqrt(u_field ** 2 + v_field ** 2)

        out_name = f"nwp_{src.name.lower()}_wind_{int(time.time())}.png"
        title = f"{base_title} 10m风场\n{time_str}"

        return plot_wind_barbs(lon_g, lat_g, speed, u_field, v_field,
                               title=title,
                               extent=self._extent_from_ds(ds),
                               out_path=out_name,
                               dpi=OUTPUT_DPI, skip=5)

    def _comprehensive(self, ds, src: NWPSource, base_title: str,
                       time_str: str) -> str:
        lon_g, lat_g = self._grid_from_ds(ds)

        fields = {}
        for var_name, var_key in [("temperature", "temperature"),
                                   ("pressure", "pressure"),
                                   ("wind_speed", "wind_speed"),
                                   ("humidity", "humidity")]:
            if var_key == "wind_speed":
                u = self._get_field(ds, src, "wind_u")
                v = self._get_field(ds, src, "wind_v")
                fields["wind_speed"] = np.sqrt(u ** 2 + v ** 2)
            else:
                fields[var_key] = self._get_field(ds, src, var_name)

        out_name = f"nwp_{src.name.lower()}_comprehensive_{int(time.time())}.png"
        title = f"{base_title} 综合分析\n{time_str}"

        return plot_multi_panel(lon_g, lat_g, fields,
                                title=title,
                                extent=self._extent_from_ds(ds),
                                out_path=out_name,
                                dpi=OUTPUT_DPI)

    # ------------------------------------------------------------------
    # Field extraction
    # ------------------------------------------------------------------

    def _get_field(self, ds, src: NWPSource, variable: str) -> np.ndarray:
        """Extract a 2D numpy field from the dataset with unit conversion.

        Returns data in ascending-lat order, matching plotter expectation.
        """
        # ECMWF has no direct humidity variable — compute from dewpoint + temperature.
        # Bypass _get_field unit conversions: _dewpoint_to_rh expects Kelvin for both.
        if variable == "humidity" and isinstance(src, ECMWFSource):
            t_short = src.VAR_SHORTNAMES.get("temperature", ["t2m"])
            d_short = src.VAR_SHORTNAMES.get("dewpoint", ["d2m"])
            temp_k = _extract_2d(ds, t_short)
            dew_k = _extract_2d(ds, d_short)
            if float(ds.latitude[0]) > float(ds.latitude[-1]):
                temp_k = temp_k[::-1, ...]
                dew_k = dew_k[::-1, ...]
            field = _dewpoint_to_rh(temp_k, dew_k)
            return field.astype(np.float32)

        short_names = src.VAR_SHORTNAMES.get(variable, [variable])
        field = _extract_2d(ds, short_names)

        # Flip to ascending lat if the dataset is stored N→S
        if float(ds.latitude[0]) > float(ds.latitude[-1]):
            field = field[::-1, ...]

        # Unit conversion
        if variable in ("temperature",) and np.nanmax(field) > 100:
            # Kelvin → Celsius (water freezes at 273K)
            field = field - 273.15

        if variable == "pressure" and np.nanmax(field) > 5000:
            # Pa → hPa (standard SLP ~101300 Pa vs ~1013 hPa)
            field = field / 100.0

        if variable == "precipitation" and np.nanmax(field) < 1.0:
            # m → mm (ECMWF tp is in metres)
            field = field * 1000.0

        return field.astype(np.float32)

    # ------------------------------------------------------------------
    # Grid helpers
    # ------------------------------------------------------------------

    def _grid_from_ds(self, ds) -> tuple[np.ndarray, np.ndarray]:
        """Build 2D lon/lat mesh from dataset coordinates (ascending lat)."""
        lon1d = ds.longitude.values.astype(np.float64)
        lat1d = ds.latitude.values.astype(np.float64)

        if lat1d[0] > lat1d[-1]:
            lat1d = lat1d[::-1]

        lon2d, lat2d = np.meshgrid(lon1d, lat1d)
        return lon2d, lat2d

    def _extent_from_ds(self, ds) -> list[float]:
        lon1d = ds.longitude.values.astype(np.float64)
        lat1d = ds.latitude.values.astype(np.float64)
        return [float(lon1d.min()), float(lon1d.max()),
                float(lat1d.min()), float(lat1d.max())]

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------

    @staticmethod
    def _var_cn(var: str) -> str:
        return {
            "temperature":   "2m温度",
            "precipitation": "降水量",
            "wind":          "10m风场",
            "pressure":      "海平面气压",
            "humidity":      "相对湿度",
        }.get(var, var)

    @staticmethod
    def _plot_var_type(var: str) -> str:
        """Map internal variable name → plotter var_type for color ranges."""
        return {
            "temperature":   "temperature",
            "precipitation": "precipitation",
            "wind":          "wind_speed",
            "pressure":      "pressure",
            "humidity":      "humidity",
        }.get(var, "default")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_nwp_pipeline: NWPPipeline | None = None


def get_nwp_pipeline() -> NWPPipeline:
    global _nwp_pipeline
    if _nwp_pipeline is None:
        _nwp_pipeline = NWPPipeline()
    return _nwp_pipeline
