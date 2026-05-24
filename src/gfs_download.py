"""Download GFS analysis fields for ML training.

GFS (Global Forecast System) 0.25° analysis, NOAA NOMADS.
No auth, no queue — direct HTTPS download from public servers.

Target: Yangtze Delta (116-123°E, 27-35°N)
Period: 2024-01-01 to 2024-12-31, 6-hourly (00/06/12/18 UTC)
Variables: 2m temperature, 10m u/v wind, MSLP, 2m RH

Output: data/gfs/*.grib2 per time step, merged to data/gfs/gfs_yangtze_delta.nc
"""

import logging
from pathlib import Path
import sys
from datetime import datetime, timedelta

import numpy as np

try:
    import xarray as xr
    import cfgrib
except ImportError:
    pass  # checked below

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gfs-download")

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "data" / "gfs"
OUTPUT_FILE = ROOT / "data" / "gfs" / "gfs_yangtze_delta.nc"
TEMP_DIR = OUTPUT_DIR / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Domain: subset in lat/lon — NOAA NOMADS serves data in GRIB2, we subset after download
# GFS is global 0.25°, so each file is ~300 MB for all vars
# We use pgrb2.0p25 (pressure level + surface) analysis files

BASE_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 3, 31)  # start with 3 months for speed
HOURS = [0, 6, 12, 18]
FORECAST_HOUR = 0  # analysis (f000)

# Bounding box: [leftlon, rightlon, top(lat), bottom(lat)]
BBOX = "116,123,35,27"

# Variables to extract: GFS grib2 variable names
# 2m temp, u10, v10, prmsl, rh2m
VARS = [
    "TMP:2 m above ground",
    "UGRD:10 m above ground",
    "VGRD:10 m above ground",
    "PRMSL:mean sea level",
    "RH:2 m above ground",
]
VAR_STRING = " -var ".join(VARS)


def build_url(date: datetime, hour: int) -> str:
    """Build NOAA NOMADS filter URL for a single GFS analysis time."""
    date_str = date.strftime("%Y%m%d")
    # GFS directory structure: gfs.YYYYMMDD/HH/atmos/
    dir_str = f"gfs.{date_str}/{hour:02d}/atmos"
    file_str = f"gfs.t{hour:02d}z.pgrb2.0p25.f{FORECAST_HOUR:03d}"
    url = (
        f"{BASE_URL}"
        f"?file={file_str}"
        f"&lev_2_m_above_ground=on"
        f"&lev_mean_sea_level=on"
        f"&var_TMP=on&var_UGRD=on&var_VGRD=on&var_PRMSL=on&var_RH=on"
        f"&subregion=&toplat={BBOX.split(',')[2]}"
        f"&leftlon={BBOX.split(',')[0]}"
        f"&rightlon={BBOX.split(',')[1]}"
        f"&bottomlat={BBOX.split(',')[3]}"
        f"&dir=%2F{dir_str}"
    )
    return url


def download_one(date: datetime, hour: int, out_path: Path) -> bool:
    """Download a single GFS analysis field as GRIB2."""
    import requests

    url = build_url(date, hour)
    label = f"{date.strftime('%Y%m%d')}_{hour:02d}z"

    for attempt in range(1, 6):
        try:
            # Disable proxy for NOAA access
            resp = requests.get(url, timeout=120, proxies={"http": None, "https": None})
            if resp.status_code == 200 and len(resp.content) > 1000:
                out_path.write_bytes(resp.content)
                logger.info(f"  {label}: {len(resp.content)/1e3:.0f} KB")
                return True
            else:
                logger.warning(f"  {label}: HTTP {resp.status_code}, size={len(resp.content)}")
        except Exception as e:
            logger.warning(f"  {label} attempt {attempt}: {e}")

        if attempt < 5:
            import time
            time.sleep(10 * attempt)
    return False


def main():
    # Check dependencies
    try:
        import cfgrib  # noqa: F401
    except ImportError:
        logger.error("cfgrib not installed. Run: pip install cfgrib")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"GFS download: {START_DATE.date()} to {END_DATE.date()}")
    logger.info(f"  Bounding box: {BBOX}")
    logger.info(f"  Hours: {[f'{h:02d}z' for h in HOURS]}")

    # Download time steps
    current = START_DATE
    grib_files = []
    skipped = 0
    failed = 0

    while current <= END_DATE:
        for hour in HOURS:
            label = f"{current.strftime('%Y%m%d')}_{hour:02d}z"
            grib_path = TEMP_DIR / f"gfs_{label}.grib2"
            nc_path = TEMP_DIR / f"gfs_{label}.nc"

            if nc_path.exists() and nc_path.stat().st_size > 1000:
                grib_files.append(nc_path)
                skipped += 1
                continue

            ok = download_one(current, hour, grib_path)
            if ok:
                # Convert GRIB2 to netCDF for easier stacking
                try:
                    ds = xr.open_dataset(grib_path, engine="cfgrib",
                                         backend_kwargs={"filter_by_keys": {"typeOfLevel": "surface"}})
                    ds = ds.reset_coords(drop=True)
                    ds.to_netcdf(nc_path)
                    grib_files.append(nc_path)
                    grib_path.unlink()  # remove temp grib
                except Exception as e:
                    logger.warning(f"  {label} grib→nc failed: {e}")
                    # Try alternative: keep as grib
                    grib_files.append(grib_path)
            else:
                failed += 1
                if failed > 20:
                    logger.error("Too many failures, aborting.")
                    sys.exit(1)

        current += timedelta(days=1)

    logger.info(f"Downloaded: {len(grib_files)} files, skipped: {skipped}, failed: {failed}")

    if len(grib_files) < 10:
        logger.error("Not enough data to proceed.")
        sys.exit(1)

    # Merge all time steps into one netCDF
    logger.info(f"Merging {len(grib_files)} files → {OUTPUT_FILE} ...")
    datasets = []
    for f in grib_files:
        try:
            ds = xr.open_dataset(f).expand_dims("time")
            datasets.append(ds)
        except Exception:
            continue

    merged = xr.concat(datasets, dim="time")
    merged = merged.sortby("time")
    merged.to_netcdf(OUTPUT_FILE)
    logger.info(f"Merged: {OUTPUT_FILE} ({OUTPUT_FILE.stat().st_size/1e6:.1f} MB)")

    # Cleanup temp files
    for f in grib_files:
        f.unlink()
    TEMP_DIR.rmdir()
    logger.info("Done!")


if __name__ == "__main__":
    main()
