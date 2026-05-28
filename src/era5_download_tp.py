"""Download ERA5 total_precipitation — all years in one request, matching existing data.

Same region / times as era5_download.py.
Output: data/era5/era5_tp.nc (~15 MB)
"""

import logging
import time as _time
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("era5-tp-download")

try:
    import cdsapi
except ImportError:
    logger.error("cdsapi not installed. Run: pip install cdsapi")
    sys.exit(1)

AREA = [35, 116, 27, 123]        # [North, West, South, East]
YEARS = ["2022", "2023", "2024"]
MONTHS = [str(m).zfill(2) for m in range(1, 13)]
DAYS = [str(d).zfill(2) for d in range(1, 32)]
TIMES = ["00:00", "06:00", "12:00", "18:00"]

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "data" / "era5"
OUTPUT_PATH = OUTPUT_DIR / "era5_tp.nc"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size > 1e5:
        logger.info(f"{OUTPUT_PATH.name} already exists, skipping")
        return

    request = {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": ["total_precipitation"],
        "year": YEARS,
        "month": MONTHS,
        "day": DAYS,
        "time": TIMES,
        "area": AREA,
    }

    client = cdsapi.Client()
    logger.info(f"Target: total_precipitation, area={AREA}, years={YEARS}")

    max_retries = 5
    for attempt in range(1, max_retries + 1):
        logger.info(f"Downloading tp → {OUTPUT_PATH.name} (attempt {attempt}/{max_retries}) ...")
        try:
            if OUTPUT_PATH.exists():
                OUTPUT_PATH.unlink()
            client.retrieve("reanalysis-era5-single-levels", request, str(OUTPUT_PATH))
            size_mb = OUTPUT_PATH.stat().st_size / 1e6
            logger.info(f"Done ({size_mb:.1f} MB)")
            return
        except Exception as e:
            logger.warning(f"  Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                wait = 15 * attempt
                logger.info(f"  Retrying in {wait}s ...")
                _time.sleep(wait)
    logger.error("All attempts failed")
    sys.exit(1)


if __name__ == "__main__":
    main()
