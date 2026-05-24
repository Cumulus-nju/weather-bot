"""Pipeline orchestrator: station data → interpolation → plot.

Data flow:
  1. Fetch real-time station observations (cached 15 min)
  2. Optionally fetch ERA5/GFS background field (cached 6 h)
  3. Run ML multi-variate refinement (all variables jointly)
  4. Plot with Cartopy → return path to PNG
"""

from __future__ import annotations

import time
import logging
import traceback
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from config import (
    DEFAULT_EXTENT, DEFAULT_GRID_RES, DEFAULT_METHOD,
    OUTPUT_DIR, OUTPUT_DPI, STATION_DATA_SOURCE,
    BARNES_SIGMA, BARNES_PASSES, BARNES_GAMMA,
    CRESSMAN_RADII,
    OI_BG_ERROR_VAR, OI_OBS_ERROR_VAR, OI_CORR_LENGTH, OI_MAX_SCAN_RADIUS, OI_MAX_STATIONS,
    PIPELINE_MAX_TIME, POST_SMOOTH_SIGMA,
)
from src.interpolation import make_grid, barnes, cressman, idw, ordinary_kriging
from src.assimilation import oi_analyze
from src.plotter import plot_grid, plot_comparison, plot_wind_barbs, plot_multi_panel
from src.station_data import create_station_source, StationObs
from src.background import try_get_background
from src.ml_model import load_refiner, refine

logger = logging.getLogger("weather-bot.pipeline")


def _temp_rh_to_dewpoint(temp_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    """Convert temperature (°C) + relative humidity (%) → dewpoint (°C).
    Uses the Magnus formula with Sonntag coefficients.
    """
    a, b = 17.62, 243.12
    es = 6.112 * np.exp(a * temp_c / (temp_c + b))  # saturation vapour pressure (hPa)
    e = es * rh_pct / 100.0
    # Clip to avoid log of zero or negative
    e = np.clip(e, 1e-6, None)
    return (b * np.log(e / 6.112)) / (a - np.log(e / 6.112))


class Pipeline:
    def __init__(self):
        self.extent = DEFAULT_EXTENT
        self.grid_res = DEFAULT_GRID_RES
        self.method = DEFAULT_METHOD
        self.lon_g, self.lat_g = make_grid(self.extent, self.grid_res)
        self.station_source = create_station_source(STATION_DATA_SOURCE)
        self.refiner = load_refiner() if self.method == "ml" else None
        logger.info(f"Pipeline init: extent={self.extent}, res={self.grid_res}, "
                    f"grid={self.lon_g.shape}, source={STATION_DATA_SOURCE}, method={self.method}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, variable: str) -> str:
        """Run full pipeline and return path to the generated PNG.

        variable: 'temperature' | 'precipitation' | 'wind' | 'pressure' | 'humidity' | 'comprehensive'
        """
        t0 = time.perf_counter()
        logger.info(f"Pipeline start: variable={variable}")

        # 1. Fetch station data
        obs = self.station_source.fetch(self.extent)
        if len(obs.names) < 3:
            raise RuntimeError(f"仅有 {len(obs.names)} 个站点有数据，无法插值。请稍后再试。")

        # 2. Get background if available
        bg_field = try_get_background(self.lon_g, self.lat_g, variable)
        has_bg = bg_field is not None
        logger.info(f"Background: {'available' if has_bg else 'unavailable — using station-only interpolation'}")

        # 3. Dispatch
        if variable == "comprehensive":
            path = self._comprehensive(obs, bg_field)
        elif variable == "wind":
            path = self._wind(obs, bg_field)
        else:
            path = self._scalar(variable, obs, bg_field)
            if has_bg:
                path = self._scalar_with_bg(variable, obs, bg_field)

        elapsed = time.perf_counter() - t0
        logger.info(f"Pipeline done: {variable} → {path} ({elapsed:.1f}s)")
        return path

    # ------------------------------------------------------------------
    # ML refinement (shared)
    # ------------------------------------------------------------------

    def _ml_refine(self, obs: StationObs) -> dict[str, np.ndarray]:
        """Build all 5 IDW channels and run ML refinement once. Returns refined dict."""
        dewpoint = _temp_rh_to_dewpoint(obs.temp, obs.humidity)

        var_data = {
            "t2m": obs.temp,
            "d2m": dewpoint,
            "u10": obs.wind_u,
            "v10": obs.wind_v,
            "msl": obs.pressure,
        }

        idw_fields = {}
        for name, values in var_data.items():
            f = idw(obs.lons, obs.lats, values, self.lon_g, self.lat_g,
                    power=2.0, min_neighbors=1, max_radius=2.0)
            nan_mask = np.isnan(f)
            if nan_mask.any():
                f[nan_mask] = np.nanmean(values)
            idw_fields[name] = f.astype(np.float32)

        lsm = np.load(Path(__file__).parent.parent / "data" / "training" / "land_sea_mask.npy")
        return refine(self.refiner, idw_fields, lsm)

    # ------------------------------------------------------------------
    # Scalar variable (temperature, humidity, pressure, precipitation)
    # ------------------------------------------------------------------

    def _scalar(self, variable: str, obs: StationObs, bg_field=None) -> str:
        out_name = f"weather_{variable}_{int(time.time())}.png"

        if bg_field is not None:
            data = obs.to_dict(variable)
            lons_s, lats_s, values = data["lon"], data["lat"], data["values"]
            errors = data.get("errors", np.full_like(values, 1.0))
            analysis, _ = oi_analyze(
                lons_s, lats_s, values, errors,
                self.lon_g, self.lat_g, bg_field,
                bg_error_var=OI_BG_ERROR_VAR,
                corr_length=OI_CORR_LENGTH,
                max_stations=OI_MAX_STATIONS,
                max_scan=OI_MAX_SCAN_RADIUS,
            )
            field = analysis

        elif self.refiner is not None:
            refined = self._ml_refine(obs)
            # Map variable name to channel
            channel_map = {"temperature": "t2m", "humidity": "d2m",
                           "pressure": "msl", "precipitation": None}
            ch = channel_map.get(variable, "t2m")
            if ch is None:
                # precipitation: ML model doesn't handle it, use Barnes
                data = obs.to_dict(variable)
                field = barnes(data["lon"], data["lat"], data["values"],
                               self.lon_g, self.lat_g,
                               sigma=BARNES_SIGMA, passes=BARNES_PASSES,
                               gamma=BARNES_GAMMA, max_scan=OI_MAX_SCAN_RADIUS)
            elif ch == "d2m":
                # humidity: compute RH from refined t2m and d2m
                dewpoint_f = refined[ch]
                temp_f = refined["t2m"]
                # Inverse of Magnus: RH = 100 * e/es
                a, b = 17.62, 243.12
                es = 6.112 * np.exp(a * temp_f / (temp_f + b))
                e = 6.112 * np.exp(a * dewpoint_f / (dewpoint_f + b))
                rh = np.clip(100.0 * e / es, 0, 100)
                field = rh
            else:
                field = refined[ch]

            lons_s, lats_s, values = obs.lons, obs.lats, obs.temp
        else:
            data = obs.to_dict(variable)
            lons_s, lats_s, values = data["lon"], data["lat"], data["values"]
            field = self._fallback_interp(lons_s, lats_s, values)

        field = self._smooth(field)
        stations_dict = {"lon": obs.lons, "lat": obs.lats,
                          "values": self._station_display_values(variable, obs),
                          "names": obs.names}
        return plot_grid(self.lon_g, self.lat_g, field,
                         title=self._title(variable),
                         var_type=variable,
                         extent=self.extent,
                         stations=stations_dict,
                         out_path=out_name,
                         dpi=OUTPUT_DPI)

    def _scalar_with_bg(self, variable: str, obs: StationObs, bg_field) -> str:
        data = obs.to_dict(variable)
        lons_s, lats_s, values = data["lon"], data["lat"], data["values"]
        errors = data.get("errors", np.full_like(values, 1.0))
        out_name = f"weather_{variable}_oi_{int(time.time())}.png"

        analysis, _ = oi_analyze(
            lons_s, lats_s, values, errors,
            self.lon_g, self.lat_g, bg_field,
            bg_error_var=OI_BG_ERROR_VAR,
            corr_length=OI_CORR_LENGTH,
            max_stations=OI_MAX_STATIONS,
            max_scan=OI_MAX_SCAN_RADIUS,
        )
        return plot_comparison(self.lon_g, self.lat_g, analysis, bg_field,
                               title=f"{self._title(variable)} — OI Analysis",
                               var_type=variable, out_path=out_name)

    # ------------------------------------------------------------------
    # Wind
    # ------------------------------------------------------------------

    def _wind(self, obs: StationObs, bg_field=None) -> str:
        out_name = f"weather_wind_{int(time.time())}.png"

        if self.refiner is not None:
            refined = self._ml_refine(obs)
            u_field = refined["u10"]
            v_field = refined["v10"]
        else:
            u_field = barnes(obs.lons, obs.lats, obs.wind_u, self.lon_g, self.lat_g,
                             sigma=BARNES_SIGMA, passes=BARNES_PASSES,
                             gamma=BARNES_GAMMA, max_scan=OI_MAX_SCAN_RADIUS)
            v_field = barnes(obs.lons, obs.lats, obs.wind_v, self.lon_g, self.lat_g,
                             sigma=BARNES_SIGMA, passes=BARNES_PASSES,
                             gamma=BARNES_GAMMA, max_scan=OI_MAX_SCAN_RADIUS)

        u_field = self._smooth(u_field)
        v_field = self._smooth(v_field)
        speed_field = np.sqrt(u_field ** 2 + v_field ** 2)

        return plot_wind_barbs(self.lon_g, self.lat_g, speed_field, u_field, v_field,
                               title=self._title("wind"),
                               extent=self.extent,
                               out_path=out_name,
                               dpi=OUTPUT_DPI)

    # ------------------------------------------------------------------
    # Comprehensive multi-panel
    # ------------------------------------------------------------------

    def _comprehensive(self, obs: StationObs, bg_field=None) -> str:
        out_name = f"weather_comprehensive_{int(time.time())}.png"
        fields = {}

        if self.refiner is not None:
            refined = self._ml_refine(obs)
            fields["temperature"] = refined["t2m"]
            fields["pressure"] = refined["msl"]
            # Wind speed from refined u/v
            ws = np.sqrt(refined["u10"] ** 2 + refined["v10"] ** 2)
            fields["wind_speed"] = ws
            # Humidity from refined t2m + d2m
            a, b = 17.62, 243.12
            es = 6.112 * np.exp(a * refined["t2m"] / (refined["t2m"] + b))
            e = 6.112 * np.exp(a * refined["d2m"] / (refined["d2m"] + b))
            fields["humidity"] = np.clip(100.0 * e / es, 0, 100)
        else:
            # Barnes fallback
            for var in ["temperature", "humidity", "pressure"]:
                data = obs.to_dict(var)
                fields[var] = barnes(data["lon"], data["lat"], data["values"],
                                     self.lon_g, self.lat_g,
                                     sigma=BARNES_SIGMA, passes=BARNES_PASSES,
                                     gamma=BARNES_GAMMA, max_scan=OI_MAX_SCAN_RADIUS)
            u_field = barnes(obs.lons, obs.lats, obs.wind_u, self.lon_g, self.lat_g,
                             sigma=BARNES_SIGMA, passes=BARNES_PASSES,
                             gamma=BARNES_GAMMA, max_scan=OI_MAX_SCAN_RADIUS)
            v_field = barnes(obs.lons, obs.lats, obs.wind_v, self.lon_g, self.lat_g,
                             sigma=BARNES_SIGMA, passes=BARNES_PASSES,
                             gamma=BARNES_GAMMA, max_scan=OI_MAX_SCAN_RADIUS)
            fields["wind_speed"] = np.sqrt(u_field ** 2 + v_field ** 2)

        for key in fields:
            fields[key] = self._smooth(fields[key])

        return plot_multi_panel(self.lon_g, self.lat_g, fields,
                                title="长三角天气综合分析",
                                extent=self.extent,
                                out_path=out_name,
                                dpi=OUTPUT_DPI)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fallback_interp(self, lons_s, lats_s, values):
        if self.method == "cressman":
            return cressman(lons_s, lats_s, values, self.lon_g, self.lat_g,
                            radii=CRESSMAN_RADII)
        elif self.method == "kriging":
            return ordinary_kriging(lons_s, lats_s, values, self.lon_g, self.lat_g,
                                    max_scan=OI_MAX_SCAN_RADIUS)
        else:
            return barnes(lons_s, lats_s, values, self.lon_g, self.lat_g,
                         sigma=BARNES_SIGMA, passes=BARNES_PASSES,
                         gamma=BARNES_GAMMA, max_scan=OI_MAX_SCAN_RADIUS)

    @staticmethod
    def _station_display_values(variable: str, obs: StationObs) -> np.ndarray:
        """Return station values appropriate for display on the plot."""
        if variable == "humidity":
            return obs.humidity
        elif variable == "pressure":
            return obs.pressure
        elif variable == "precipitation":
            return obs.precip
        return obs.temp  # default: show temperature

    @staticmethod
    def _smooth(field: np.ndarray) -> np.ndarray:
        if POST_SMOOTH_SIGMA > 0:
            return gaussian_filter(field, sigma=POST_SMOOTH_SIGMA)
        return field

    def _title(self, var: str) -> str:
        titles = {
            "temperature":  "长三角 温度分析",
            "precipitation": "长三角 降水分析",
            "wind":         "长三角 风场分析",
            "pressure":     "长三角 海平面气压分析",
            "humidity":     "长三角 相对湿度分析",
        }
        return titles.get(var, f"长三角 {var} 分析")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline
