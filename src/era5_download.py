"""Download ERA5 reanalysis data for ML training.

Target: Yangtze Delta (116-123°E, 27-35°N), 0.25° resolution
Period: 2022-01-01 to 2024-12-31, 6-hourly (00/06/12/18 UTC)
Variables: 2m_temperature, 10m_u_component_of_wind, 10m_v_component_of_wind,
           mean_sea_level_pressure, 2m_dewpoint_temperature

Output: data/era5/era5_yangtze_delta.nc (~80 MB)

Requires CDS API credentials in ~/.cdsapirc (see README).
"""

# ── Proxy: comment/uncomment to toggle ──────────────────────────────────
# If your proxy (127.0.0.1:7897) is stable, comment out this block to use it.
# If the proxy causes SSL errors, uncomment to bypass it.
"""
import os
os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""

import requests as _requests
_orig_send = _requests.Session.send
def _no_proxy_send(self, req, **kw):
    kw["proxies"] = {}
    return _orig_send(self, req, **kw)
_requests.Session.send = _no_proxy_send
"""
# ─────────────────────────────────────────────────────────────────────────

import logging
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("era5-download")

try:
    import cdsapi
except ImportError:
    logger.error("cdsapi not installed. Run: pip install cdsapi")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REGION = "Yangtze Delta"
AREA = [35, 116, 27, 123]  # [North, West, South, East] — CDS convention
YEARS = [2022, 2023, 2024]
MONTHS = list(range(1, 13))
DAYS = list(range(1, 32))
TIMES = ["00:00", "06:00", "12:00", "18:00"]

VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "2m_dewpoint_temperature",
]

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "data" / "era5"
OUTPUT_FILE = OUTPUT_DIR / "era5_yangtze_delta.nc"


import time as _time

def download_year(client, year, out_path, max_retries=5):
    """Download a single year of ERA5 data with retries for network issues."""
    request = {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": VARIABLES,
        "year": [str(year)],
        "month": [str(m).zfill(2) for m in MONTHS],
        "day": [str(d).zfill(2) for d in DAYS],
        "time": TIMES,
        "area": AREA,
    }

    for attempt in range(1, max_retries + 1):
        logger.info(f"Downloading {year} → {out_path.name} (attempt {attempt}/{max_retries}) ...")
        try:
            # Remove partial file from previous attempt
            if out_path.exists():
                out_path.unlink()

            client.retrieve(
                "reanalysis-era5-single-levels",
                request,
                str(out_path),
            )
            size_mb = out_path.stat().st_size / 1e6
            logger.info(f"  {year} done ({size_mb:.1f} MB)")
            return True
        except Exception as e:
            logger.warning(f"  Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                wait = 15 * attempt
                logger.info(f"  Retrying in {wait}s ...")
                _time.sleep(wait)
    return False


def merge_years(year_files):
    """Merge yearly netCDF files into one."""
    import xarray as xr
    datasets = [xr.open_dataset(f) for f in year_files]
    merged = xr.concat(datasets, dim="time")
    merged.to_netcdf(OUTPUT_FILE)
    logger.info(f"Merged → {OUTPUT_FILE} ({OUTPUT_FILE.stat().st_size / 1e6:.1f} MB)")
    # Remove yearly files
    for f in year_files:
        f.unlink()
        logger.debug(f"Removed {f.name}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    logger.info(f"Target: {REGION}, {VARIABLES}")
    logger.info(f"Years: {YEARS}")

    year_files = []
    for year in YEARS:
        out_path = OUTPUT_DIR / f"era5_{year}.nc"
        if out_path.exists() and out_path.stat().st_size > 1e6:
            logger.info(f"  {year} already downloaded ({out_path.stat().st_size/1e6:.1f} MB), skipping")
            year_files.append(out_path)
            continue
        ok = download_year(client, year, out_path)
        if ok:
            year_files.append(out_path)
        else:
            logger.error(f"  {year} failed after all retries")
            sys.exit(1)

    if len(year_files) == len(YEARS):
        merge_years(year_files)
        logger.info(f"All done: {OUTPUT_FILE}")
    else:
        logger.error(f"Only {len(year_files)}/{len(YEARS)} years downloaded")


if __name__ == "__main__":
    main()
