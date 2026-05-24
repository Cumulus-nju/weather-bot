"""Real-time weather station data fetchers.

Supports multiple backends:
  - OpenWeatherMap   (generous free tier, ~10-min updates)
  - QWeather / 和风天气 (China-optimised, tighter rate limits)
  - OGIMET SYNOP      (raw WMO station reports, no key needed, 3-hourly)
  - Synthetic          (demo mode — works offline)
"""

from __future__ import annotations

import json
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import httpx

from config import (
    STATION_LIST_FILE, STATION_DATA_CACHE_TTL, HTTP_TIMEOUT,
    OPENWEATHERMAP_API_KEY, QWATHER_API_KEY, QWATHER_API_HOST, HTTP_USER_AGENT,
)

logger = logging.getLogger("weather-bot.station_data")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StationObs:
    lons: np.ndarray       # degrees
    lats: np.ndarray
    names: list
    temp: np.ndarray       # °C
    humidity: np.ndarray   # %
    pressure: np.ndarray   # hPa
    wind_speed: np.ndarray # m/s
    wind_dir: np.ndarray   # degrees (meteorological)
    wind_u: np.ndarray     # m/s (eastward)
    wind_v: np.ndarray     # m/s (northward)
    precip: np.ndarray     # mm (last hour, where available)

    def to_dict(self, variable: str):
        """Return {'lons','lats','values','errors'} for a given variable."""
        var_map = {
            "temperature": (self.temp, np.full_like(self.temp, 1.0)),
            "humidity": (self.humidity, np.full_like(self.humidity, 5.0)),
            "pressure": (self.pressure, np.full_like(self.pressure, 2.0)),
            "wind_speed": (self.wind_speed, np.full_like(self.wind_speed, 1.5)),
            "precipitation": (self.precip, np.full_like(self.precip, 0.5)),
        }
        values, errors = var_map.get(variable, (None, None))
        if values is None:
            raise ValueError(f"Unknown variable: {variable}")
        return {"lon": self.lons, "lat": self.lats, "values": values, "errors": errors}

    def mask_valid(self):
        """Remove stations where temperature is missing."""
        valid = ~np.isnan(self.temp)
        self.lons = self.lons[valid]
        self.lats = self.lats[valid]
        self.names = [n for n, v in zip(self.names, valid) if v]
        self.temp = self.temp[valid]
        self.humidity = self.humidity[valid]
        self.pressure = self.pressure[valid]
        self.wind_speed = self.wind_speed[valid]
        self.wind_dir = self.wind_dir[valid]
        self.wind_u = self.wind_u[valid]
        self.wind_v = self.wind_v[valid]
        self.precip = self.precip[valid]
        return self


# ---------------------------------------------------------------------------
# Abstract source
# ---------------------------------------------------------------------------

class StationDataSource(ABC):
    @abstractmethod
    def fetch(self, extent: tuple) -> StationObs:
        """Return observations within the given [lon_min, lon_max, lat_min, lat_max]."""
        ...


# ---------------------------------------------------------------------------
# Base class with caching and station-list loading
# ---------------------------------------------------------------------------

class CachedSource(StationDataSource):
    def __init__(self):
        self._cache: Optional[tuple[float, StationObs]] = None

    def _load_stations(self) -> list[dict]:
        with open(STATION_LIST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def _filter_by_extent(self, stations: list[dict], extent: tuple) -> list[dict]:
        lon_min, lon_max, lat_min, lat_max = extent
        return [s for s in stations
                if lon_min <= s["lon"] <= lon_max and lat_min <= s["lat"] <= lat_max]

    def fetch(self, extent: tuple) -> StationObs:
        now = time.time()
        if self._cache and (now - self._cache[0] < STATION_DATA_CACHE_TTL):
            obs = self._cache[1]
            return self._filter_obs_by_extent(obs, extent)
        obs = self._do_fetch(extent)
        self._cache = (now, obs)
        return obs

    def _filter_obs_by_extent(self, obs: StationObs, extent: tuple) -> StationObs:
        lon_min, lon_max, lat_min, lat_max = extent
        mask = (obs.lons >= lon_min) & (obs.lons <= lon_max) & \
               (obs.lats >= lat_min) & (obs.lats <= lat_max)
        import copy
        out = copy.copy(obs)
        out.lons = obs.lons[mask]
        out.lats = obs.lats[mask]
        out.names = [n for n, m in zip(obs.names, mask) if m]
        out.temp = obs.temp[mask]
        out.humidity = obs.humidity[mask]
        out.pressure = obs.pressure[mask]
        out.wind_speed = obs.wind_speed[mask]
        out.wind_dir = obs.wind_dir[mask]
        out.wind_u = obs.wind_u[mask]
        out.wind_v = obs.wind_v[mask]
        out.precip = obs.precip[mask]
        return out

    @abstractmethod
    def _do_fetch(self, extent: tuple) -> StationObs:
        ...


# ---------------------------------------------------------------------------
# OpenWeatherMap
# ---------------------------------------------------------------------------

class OpenWeatherMapSource(CachedSource):
    """Fetch current weather for each city via OpenWeatherMap API.

    Free tier: 1,000,000 calls/month, 60 calls/minute.
    """

    BASE = "https://api.openweathermap.org/data/2.5/weather"

    def __init__(self, api_key: str = ""):
        super().__init__()
        self.api_key = api_key or OPENWEATHERMAP_API_KEY

    def _do_fetch(self, extent: tuple) -> StationObs:
        stations = self._load_stations()
        stations = self._filter_by_extent(stations, extent)
        logger.info(f"OWM: fetching {len(stations)} stations ...")

        names, lons, lats = [], [], []
        temps, hums, press, ws, wd, precip = [], [], [], [], [], []

        with httpx.Client(timeout=HTTP_TIMEOUT, headers={"User-Agent": HTTP_USER_AGENT},
                           trust_env=False) as client:
            for s in stations:
                try:
                    params = {"q": s["owm"], "appid": self.api_key,
                              "units": "metric", "lang": "zh_cn"}
                    r = client.get(self.BASE, params=params)
                    r.raise_for_status()
                    d = r.json()
                    names.append(s["name"])
                    lons.append(s["lon"])
                    lats.append(s["lat"])
                    temps.append(d["main"]["temp"])
                    hums.append(d["main"]["humidity"])
                    press.append(d["main"]["pressure"])
                    wd_deg = d["wind"].get("deg", 0)
                    ws_mps = d["wind"].get("speed", 0)
                    wd.append(wd_deg)
                    ws.append(ws_mps)
                    # Precip — OWM "rain.1h" if available
                    p = d.get("rain", {}).get("1h", 0.0)
                    precip.append(p)
                except Exception as e:
                    logger.warning(f"OWM: {s['name']} failed: {e}")

        arr = np.array
        obs = StationObs(
            lons=arr(lons, dtype=float),
            lats=arr(lats, dtype=float),
            names=names,
            temp=arr(temps, dtype=float),
            humidity=arr(hums, dtype=float),
            pressure=arr(press, dtype=float),
            wind_speed=arr(ws, dtype=float),
            wind_dir=arr(wd, dtype=float),
            wind_u=np.zeros(len(ws)),
            wind_v=np.zeros(len(ws)),
            precip=arr(precip, dtype=float),
        )
        # Convert wind dir/speed → u, v
        wd_rad = np.radians(270 - obs.wind_dir)  # meteorological → math convention
        obs.wind_u = obs.wind_speed * np.cos(wd_rad)
        obs.wind_v = obs.wind_speed * np.sin(wd_rad)
        obs.mask_valid()
        logger.info(f"OWM: got {len(obs.names)} valid stations")
        return obs


# ---------------------------------------------------------------------------
# QWeather / 和风天气
# ---------------------------------------------------------------------------

class QWeatherSource(CachedSource):
    """Fetch via QWeather dev API. Free tier: 1,000 calls/day."""

    def __init__(self, api_key: str = "", api_host: str = ""):
        super().__init__()
        self.api_key = api_key or QWATHER_API_KEY
        self.api_host = api_host or QWATHER_API_HOST
        # Use custom API Host if available, otherwise old shared domain
        if self.api_host:
            self.BASE = f"https://{self.api_host}/v7/weather/now"
        else:
            self.BASE = "https://devapi.qweather.com/v7/weather/now"

    def _do_fetch(self, extent: tuple) -> StationObs:
        stations = self._load_stations()
        stations = self._filter_by_extent(stations, extent)

        names, lons, lats = [], [], []
        temps, hums, press, ws, wd, precip = [], [], [], [], [], []

        with httpx.Client(timeout=HTTP_TIMEOUT, headers={"User-Agent": HTTP_USER_AGENT},
                           trust_env=False) as client:
            for s in stations:
                try:
                    params = {"location": s["qw"], "key": self.api_key}
                    r = client.get(self.BASE, params=params)
                    r.raise_for_status()
                    d = r.json()
                    now = d.get("now", {})
                    names.append(s["name"])
                    lons.append(s["lon"])
                    lats.append(s["lat"])
                    temps.append(float(now["temp"]))
                    hums.append(float(now["humidity"]))
                    press.append(float(now["pressure"]))
                    wd_deg = float(now.get("wind360", 0))
                    ws_mps = float(now.get("windSpeed", 0))
                    wd.append(wd_deg)
                    ws.append(ws_mps)
                    precip.append(float(now.get("precip", 0.0)))
                except Exception as e:
                    logger.warning(f"QWeather: {s['name']} failed: {e}")

        arr = np.array
        obs = StationObs(
            lons=arr(lons, dtype=float), lats=arr(lats, dtype=float), names=names,
            temp=arr(temps, dtype=float), humidity=arr(hums, dtype=float),
            pressure=arr(press, dtype=float), wind_speed=arr(ws, dtype=float),
            wind_dir=arr(wd, dtype=float), wind_u=np.zeros(len(ws)),
            wind_v=np.zeros(len(ws)), precip=arr(precip, dtype=float),
        )
        wd_rad = np.radians(270 - obs.wind_dir)
        obs.wind_u = obs.wind_speed * np.cos(wd_rad)
        obs.wind_v = obs.wind_speed * np.sin(wd_rad)
        obs.mask_valid()
        return obs


# ---------------------------------------------------------------------------
# OGIMET SYNOP (free, no key, real station data)
# ---------------------------------------------------------------------------

class OGIMETSource(StationDataSource):
    """Parse OGIMET SYNOP reports for Chinese stations.

    Updates every 3 hours. No API key required.
    """

    BASE = "http://www.ogimet.com/cgi-bin/gsynop"

    def fetch(self, extent: tuple) -> StationObs:
        import re
        lon_min, lon_max, lat_min, lat_max = extent

        names, lons, lats = [], [], []
        temps, hums, press, ws, wd, precip = [], [], [], [], [], []

        try:
            with httpx.Client(timeout=HTTP_TIMEOUT + 10, headers={"User-Agent": HTTP_USER_AGENT}) as client:
                r = client.get(self.BASE, params={"lang": "en", "estado": "China"})
                r.raise_for_status()
                text = r.text
        except Exception as e:
            logger.error(f"OGIMET: fetch failed: {e}")
            return self._empty()

        # Parse HTML table rows
        # Each row: <tr>...<td>station_id</td>...<td>lat</td><td>lon</td>...
        # The format is messy; we use regex to extract data blocks
        blocks = re.findall(r'<tr[^>]*>.*?</tr>', text, re.DOTALL)
        for blk in blocks:
            tds = re.findall(r'<td[^>]*>(.*?)</td>', blk, re.DOTALL)
            if len(tds) < 10:
                continue
            try:
                # td[0]: WMO id, td[2]: lat, td[3]: lon, td[5]: temp, td[6]: dewpoint
                # td[7]: pressure, td[8]: wind info, ...
                slat = float(tds[2].strip()) if tds[2].strip() else None
                slon = float(tds[3].strip()) if tds[3].strip() else None
                if slat is None or slon is None:
                    continue
                if not (lon_min <= slon <= lon_max and lat_min <= slat <= lat_max):
                    continue

                temp_str = tds[5].strip() if len(tds) > 5 else ""
                temp = float(temp_str) if temp_str and temp_str != "-" else np.nan
                humidity = np.nan  # SYNOP table doesn't directly give RH easily
                pres_str = tds[7].strip() if len(tds) > 7 else ""
                pres = float(pres_str) if pres_str and pres_str != "-" else np.nan
                # Wind — not robustly parseable from this format; skip
                wspeed = np.nan
                wdir = np.nan
                p = 0.0

                names.append(tds[0].strip())
                lons.append(slon)
                lats.append(slat)
                temps.append(temp)
                hums.append(humidity)
                press.append(pres)
                ws.append(wspeed)
                wd.append(wdir)
                precip.append(p)
            except (ValueError, IndexError):
                continue

        arr = np.array
        obs = StationObs(
            lons=arr(lons, dtype=float), lats=arr(lats, dtype=float), names=names,
            temp=arr(temps, dtype=float), humidity=arr(hums, dtype=float),
            pressure=arr(press, dtype=float), wind_speed=arr(ws, dtype=float),
            wind_dir=arr(wd, dtype=float), wind_u=np.full(len(ws), np.nan),
            wind_v=np.full(len(ws), np.nan), precip=arr(precip, dtype=float),
        )
        obs.mask_valid()
        logger.info(f"OGIMET: parsed {len(obs.names)} stations in extent")
        return obs

    def _empty(self) -> StationObs:
        arr = np.array([])
        return StationObs(arr, arr, [], arr, arr, arr, arr, arr, arr, arr, arr)


# ---------------------------------------------------------------------------
# Synthetic (offline demo)
# ---------------------------------------------------------------------------

class SyntheticSource(StationDataSource):
    """Generate fake-but-plausible observations for testing."""

    def fetch(self, extent: tuple) -> StationObs:
        stations = self._load_stations()
        stations = self._filter_by_extent(stations, extent)
        rng = np.random.default_rng(int(time.time()) // 600)  # stable within 10 min

        n = len(stations)
        lons = np.array([s["lon"] for s in stations])
        lats = np.array([s["lat"] for s in stations])
        names = [s["name"] for s in stations]

        # Base temperature field: meriodional + zonal gradient
        temp = 25 - 0.4 * (lats - 30) - 0.15 * (lons - 119) + rng.normal(0, 1.2, n)
        humidity = 65 + 2 * (lats - 30) + rng.normal(0, 8, n)
        humidity = np.clip(humidity, 20, 100)
        pressure = 1013 - 2 * (lats - 30) + rng.normal(0, 2, n)
        ws = np.abs(rng.normal(3, 3, n))
        wd = rng.uniform(0, 360, n)
        wd_rad = np.radians(270 - wd)
        precip = np.maximum(0, rng.exponential(1.5, n))

        obs = StationObs(
            lons=lons, lats=lats, names=names,
            temp=temp, humidity=humidity, pressure=pressure,
            wind_speed=ws, wind_dir=wd,
            wind_u=ws * np.cos(wd_rad), wind_v=ws * np.sin(wd_rad),
            precip=precip,
        )
        logger.info(f"Synthetic: generated {n} fake stations")
        return obs

    def _load_stations(self):
        with open(STATION_LIST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def _filter_by_extent(self, stations, extent):
        lon_min, lon_max, lat_min, lat_max = extent
        return [s for s in stations
                if lon_min <= s["lon"] <= lon_max and lat_min <= s["lat"] <= lat_max]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_station_source(name: str = "synthetic") -> StationDataSource:
    sources = {
        "openweathermap": OpenWeatherMapSource,
        "qweather": QWeatherSource,
        "ogimet": OGIMETSource,
        "synthetic": SyntheticSource,
    }
    cls = sources.get(name)
    if cls is None:
        logger.warning(f"Unknown source '{name}', falling back to synthetic")
        cls = SyntheticSource
    return cls()
