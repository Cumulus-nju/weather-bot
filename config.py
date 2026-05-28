"""Global configuration for weather-bot — Yangtze River Delta edition."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — use system env vars

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent
OUTPUT_DIR = ROOT_DIR / "output"
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Region definitions (lon_min, lon_max, lat_min, lat_max)
# ---------------------------------------------------------------------------
CHINA_EXTENT = [73.0, 136.0, 16.0, 55.0]
EAST_CHINA_EXTENT = [105.0, 125.0, 18.0, 42.0]
YANGTZE_DELTA_EXTENT = [116.0, 123.0, 27.0, 35.0]  # 长三角

DEFAULT_EXTENT = YANGTZE_DELTA_EXTENT

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
DEFAULT_GRID_RES = 0.05  # ~5.5 km at this latitude

# ---------------------------------------------------------------------------
# Barnes interpolation
# ---------------------------------------------------------------------------
BARNES_SIGMA = 1.5         # degrees — broad first pass for smooth large-scale field
BARNES_PASSES = 2
BARNES_GAMMA = 0.5         # 1.5 → 0.75, overlapping corrections between stations

# ---------------------------------------------------------------------------
# Cressman
# ---------------------------------------------------------------------------
CRESSMAN_RADII = [3.0, 1.5, 0.75]  # tightened for regional domain

# ---------------------------------------------------------------------------
# OI (Optimal Interpolation)
# ---------------------------------------------------------------------------
OI_BG_ERROR_VAR = 4.0
OI_OBS_ERROR_VAR = 1.0
OI_CORR_LENGTH = 1.5         # degrees — shorter for regional scale
OI_MAX_SCAN_RADIUS = 3.0     # degrees — local neighborhood only, avoid distant-station pollution
OI_MAX_STATIONS = 50

# ---------------------------------------------------------------------------
# Kriging — disabled for real-time bot (too slow); kept for offline use
# ---------------------------------------------------------------------------
KRIGING_RANGE = 2.0
KRIGING_SILL = 1.0
KRIGING_NUGGET = 0.1
KRIGING_MAX_STATIONS = 40

# ---------------------------------------------------------------------------
# Station data source
# ---------------------------------------------------------------------------
# "openweathermap" | "qweather" | "ogimet" | "synthetic"
STATION_DATA_SOURCE = os.getenv("STATION_DATA_SOURCE", "openweathermap")
STATION_DATA_CACHE_TTL = 900  # 15 minutes
STATION_LIST_FILE = DATA_DIR / "stations_yangtze_delta.json"

# API keys
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")
QWATHER_API_KEY = os.getenv("QWATHER_API_KEY", "")
QWATHER_API_HOST = os.getenv("QWATHER_API_HOST", "")

# HTTP
HTTP_TIMEOUT = 10            # seconds per API call
HTTP_USER_AGENT = "weather-bot-nju/1.0"

# ---------------------------------------------------------------------------
# Background field
# ---------------------------------------------------------------------------
BACKGROUND_SOURCE = os.getenv("BACKGROUND_SOURCE", "none")  # "era5" | "gfs" | "none"
BACKGROUND_CACHE_TTL = 21600  # 6 hours

# ERA5 / CDS
CDSAPI_URL = os.getenv("CDSAPI_URL", "")
CDSAPI_KEY = os.getenv("CDSAPI_KEY", "")

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
DEFAULT_METHOD = "ml"        # "barnes" | "cressman" | "oi" | "kriging" | "ml"
PIPELINE_MAX_TIME = 30       # seconds
MAX_CONCURRENT = 4           # max simultaneous pipeline runs

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUTPUT_DPI = 150
OUTPUT_FORMAT = "png"

# Light post-processing Gaussian smoothing (grid-cell units, 0=disabled)
POST_SMOOTH_SIGMA = 1.2  # ~0.06° at 0.05° res, light smoothing to remove residual noise

# ---------------------------------------------------------------------------
# NWP (Numerical Weather Prediction) — ECMWF IFS & GFS
# ---------------------------------------------------------------------------
NWP_CACHE_DIR = DATA_DIR / "nwp"
NWP_CACHE_TTL = 21600           # 6h per synoptic cycle
NWP_MAX_CACHE_AGE_DAYS = 7      # auto-delete cached files older than 7 days
NWP_GRID_RES = 0.25             # native resolution of ECMWF open data and GFS 0.25°

# ==========================  LocalWords:  YANGTZE  cressman  barnes  CDSAPI

# ---------------------------------------------------------------------------
# Seasonal ML models
# ---------------------------------------------------------------------------
SEASONS = {
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "autumn": [9, 10, 11],
    "winter": [12, 1, 2],
}


def get_season(month: int | None = None) -> str:
    """Return season name for a given month (1-12). Defaults to current month."""
    if month is None:
        month = datetime.now().month
    for name, months in SEASONS.items():
        if month in months:
            return name
    return "spring"  # unreachable
