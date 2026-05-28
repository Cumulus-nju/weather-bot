"""Synoptic forecast text engine — NWP data → professional Chinese weather discussion.

Extracts features from ECMWF/GFS gridded fields at surface, 850, 500, 200 hPa
and generates a structured forecast text with national overview and local
(southern Jiangsu) 3-day prediction.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter

from config import CHINA_EXTENT, NWP_CACHE_DIR

logger = logging.getLogger("weather-bot.forecast")

# Southern Jiangsu target box (lon_min, lon_max, lat_min, lat_max)
SUNAN_BOX = [119.5, 121.5, 31.0, 32.5]

# Geographic reference points for feature description
GEO_REF = {
    "蒙古": (105, 47),
    "华北": (115, 40),
    "东北": (125, 45),
    "黄淮": (117, 34),
    "江淮": (119, 32),
    "江南": (118, 29),
    "华南": (113, 24),
    "青藏高原": (92, 33),
    "西北": (95, 40),
    "西南": (105, 28),
    "华东": (119, 31),
    "苏南": (120.5, 31.8),
    "日本海": (135, 40),
    "东海": (125, 29),
}


def _describe_location(lon: float, lat: float) -> str:
    """Convert a lon/lat to the nearest geographic region name."""
    best, best_d = "附近", float("inf")
    for name, (rlon, rlat) in GEO_REF.items():
        d = ((lon - rlon) * np.cos(np.radians((lat + rlat) / 2))) ** 2 + (lat - rlat) ** 2
        if d < best_d:
            best_d, best = d, name
    return best


def _wind_dir_name(u: float, v: float) -> str:
    """Return Chinese wind direction from u, v components."""
    deg = np.degrees(np.arctan2(-u, -v))  # meteorological convention
    if deg < 0:
        deg += 360
    dirs = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]
    return dirs[int((deg + 22.5) % 360 // 45)]


def _beaufort(ws_mps: float) -> str:
    """Convert wind speed (m/s) to Beaufort scale description."""
    limits = [0.3, 1.6, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 32.7]
    labels = ["静风", "1级", "2级", "3级", "4级", "5级", "6级", "7级", "8级", "9级", "10级", "11级", "12级"]
    for i, lim in enumerate(limits):
        if ws_mps <= lim:
            return labels[i]
    return labels[-1]


# ---------------------------------------------------------------------------
# NWP data loader
# ---------------------------------------------------------------------------


def _load_nc(cache_dir: Path, date_str: str, hour: int, step: int) -> "xr.Dataset | None":
    """Load a cached netCDF file, returning None if not found."""
    import xarray as xr

    p = cache_dir / f"{date_str}_{hour:02d}z_f{step:03d}.nc"
    if not p.exists():
        return None
    with __import__("threading").Lock():
        return xr.open_dataset(p)


def _extract(ds, short_names: list[str], level: int | None = None) -> np.ndarray:
    """Extract a 2D array from an xarray Dataset, optionally at a pressure level."""
    for vn in list(ds.data_vars):
        da = ds[vn]
        sn = da.attrs.get("GRIB_shortName", vn)
        if sn in short_names:
            if level is not None and "isobaricInhPa" in da.dims:
                try:
                    da = da.sel(isobaricInhPa=float(level), method="nearest")
                except Exception:
                    pass
            arr = np.squeeze(da.values).astype(np.float32)
            if arr.ndim != 2:
                continue
            # Flip to ascending lat if needed
            if da.latitude.values[0] > da.latitude.values[-1]:
                arr = arr[::-1, :]
            return arr
    # Fallback: try variable name
    for sn in short_names:
        if sn in ds.data_vars:
            da = ds[sn]
            if level is not None and "isobaricInhPa" in da.dims:
                try:
                    da = da.sel(isobaricInhPa=float(level), method="nearest")
                except Exception:
                    pass
            arr = np.squeeze(da.values).astype(np.float32)
            if arr.ndim != 2:
                continue
            if da.latitude.values[0] > da.latitude.values[-1]:
                arr = arr[::-1, :]
            return arr
    raise KeyError(f"Variable {short_names} not found")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _grid_coords(ds) -> tuple[np.ndarray, np.ndarray]:
    """Return (lon2d, lat2d) in ascending-lat order."""
    lon1d = ds.longitude.values.astype(np.float64)
    lat1d = ds.latitude.values.astype(np.float64)
    if lat1d[0] > lat1d[-1]:
        lat1d = lat1d[::-1]
    return np.meshgrid(lon1d, lat1d)


def _subset_box(field, lon2d, lat2d, box):
    """Extract field within a lon/lat box, return (mean, max, min)."""
    mask = (lon2d >= box[0]) & (lon2d <= box[1]) & (lat2d >= box[2]) & (lat2d <= box[3])
    sub = field[mask]
    if len(sub) == 0:
        return 0, 0, 0
    return float(np.nanmean(sub)), float(np.nanmax(sub)), float(np.nanmin(sub))


def _find_extrema(field, lon2d, lat2d, n=3, find_max=True):
    """Find top-n local maxima or minima with their geographic descriptions."""
    smooth = gaussian_filter(field, sigma=2.0)
    if find_max:
        local = maximum_filter(smooth, size=15) == smooth
    else:
        local = minimum_filter(smooth, size=15) == smooth

    ys, xs = np.where(local)
    vals = smooth[ys, xs]
    order = np.argsort(vals)
    if find_max:
        order = order[::-1]

    results = []
    for idx in order[:n * 3]:  # oversample to avoid duplicates
        y, x = ys[idx], xs[idx]
        val = field[y, x]
        lo, la = float(lon2d[y, x]), float(lat2d[y, x])
        # Skip if too close to an already selected point
        too_close = False
        for _, rlo, rla in results:
            if abs(lo - rlo) < 5 and abs(la - rla) < 5:
                too_close = True
                break
        if not too_close:
            loc = _describe_location(lo, la)
            results.append((val, lo, la, loc))
        if len(results) >= n:
            break
    return results


def _zonal_anomaly(field):
    """Subtract zonal mean to highlight troughs/ridges."""
    zmean = np.nanmean(field, axis=1, keepdims=True)
    return field - zmean


def _temp_advection(t_field, u_field, v_field, dx_deg, dy_deg):
    """Compute simplified temperature advection = -(u·∂T/∂x + v·∂T/∂y).

    Positive → warm advection, negative → cold advection.
    dx_deg, dy_deg are grid spacings in degrees.
    """
    dty, dtx = np.gradient(gaussian_filter(t_field, sigma=1.0), dy_deg, dx_deg)
    return -(u_field * dtx + v_field * dty)


def _jet_axis(u200, v200, lon2d, lat2d):
    """Find jet stream axis (latitude of max wind speed at each longitude)."""
    ws = np.sqrt(u200 ** 2 + v200 ** 2)
    ny, nx = ws.shape
    jet_lats = np.full(nx, np.nan)
    jet_speeds = np.full(nx, np.nan)

    for i in range(nx):
        col = ws[:, i]
        if np.all(np.isnan(col)):
            continue
        j = int(np.nanargmax(col))
        jet_lats[i] = lat2d[j, i]
        jet_speeds[i] = col[j]

    # Mean jet latitude and max speed (only over China domain)
    valid = ~np.isnan(jet_lats)
    if valid.sum() == 0:
        return 35.0, 0
    mean_lat = np.nanmean(jet_lats[valid])
    max_speed = np.nanmax(jet_speeds[valid])
    return float(mean_lat), float(max_speed)


def _flow_regime(h500_anom):
    """Determine if 500hPa flow is zonal or meridional based on anomaly amplitude."""
    amp = np.nanstd(h500_anom)
    if amp > 80:
        return "经向型", "槽脊振幅较大，有利于冷暖气团南北交换"
    elif amp > 40:
        return "纬向型为主，叠加短波扰动", "西风带较为平直，冷空气活动偏弱"
    else:
        return "纬向型", "西风带平直，无明显槽脊活动"


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------


class ForecastEngine:
    """Generate structured Chinese weather forecast from NWP data."""

    def __init__(self):
        self.ecmwf_dir = NWP_CACHE_DIR / "ecmwf"
        self.gfs_dir = NWP_CACHE_DIR / "gfs"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> str:
        """Generate forecast text for the latest cycle. Returns formatted string."""
        from .nwp import ECMWFSource, GFSSource

        ec_src = ECMWFSource()
        gfs_src = GFSSource()

        ec_date, ec_hour = ec_src.get_latest_cycle()
        gfs_date, gfs_hour = gfs_src.get_latest_cycle()

        date_str_ec = ec_date.strftime("%Y%m%d")
        date_str_gfs = gfs_date.strftime("%Y%m%d")

        # Load analysis (step=0) and 72h forecast
        ec_an = _load_nc(self.ecmwf_dir, date_str_ec, ec_hour, 0)
        gfs_an = _load_nc(self.gfs_dir, date_str_gfs, gfs_hour, 0)

        if ec_an is None and gfs_an is None:
            return (
                "当前没有可用的NWP数据缓存。\n"
                "请先发送 /数据更新 下载最新数值预报数据。"
            )

        # Use whichever source is available (prefer ECMWF)
        ds_main = ec_an if ec_an is not None else gfs_an
        ds_main_label = "ECMWF IFS" if ec_an is not None else "GFS"

        # Default steps for forecast
        main_date, main_hour = (ec_date, ec_hour) if ec_an is not None else (gfs_date, gfs_hour)
        main_date_str = main_date.strftime("%Y%m%d")

        ec_date_str = ec_date.strftime("%Y%m%d") if ec_date else ""
        gfs_date_str = gfs_date.strftime("%Y%m%d") if gfs_date else ""

        parts: list[str] = []
        self._build_header(parts, main_date, main_hour, ds_main_label)
        self._build_synoptic_overview(parts, ds_main)
        self._build_upper_features(parts, ds_main)
        self._build_surface(parts, ds_main)
        self._build_sunan_forecast(parts, ec_an, gfs_an, ec_date, ec_hour, gfs_date, gfs_hour)
        self._build_model_diff(parts, ec_an, gfs_an, ec_date_str, ec_hour, gfs_date_str, gfs_hour)
        self._build_disclaimer(parts)

        # Clean up
        for ds in [ec_an, gfs_an]:
            if ds is not None:
                ds.close()

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def _build_header(self, parts, date, hour, label):
        cn_date = f"{date.year}年{date.month}月{date.day}日"
        parts.append("═══════════════════════════════════")
        parts.append(f"  {cn_date} {hour:02d}z 天气形势分析")
        parts.append(f"  数据来源: {label} 0.25°")
        parts.append("═══════════════════════════════════")

    def _build_synoptic_overview(self, parts, ds):
        try:
            h500 = self._h500(ds)
        except Exception:
            parts.append("\n【环流形势】数据不可用，跳过。")
            return

        parts.append("")
        parts.append("【一、环流形势】")
        parts.append("")
        parts.append("500hPa位势高度场分析：")
        anom = _zonal_anomaly(h500)
        regime, desc = _flow_regime(anom)
        parts.append(f"  环流型: {regime}。{desc}。")

        # Troughs (negative anomalies)
        lon2d, lat2d = _grid_coords(ds)
        troughs = _find_extrema(-anom, lon2d, lat2d, n=2, find_max=True)
        for i, (val, lo, la, loc) in enumerate(troughs):
            parts.append(f"  主要槽区: {loc}({lo:.0f}°E, {la:.0f}°N)，位势高度距平约{-val:.0f}gpm")

        ridges = _find_extrema(anom, lon2d, lat2d, n=2, find_max=True)
        for val, lo, la, loc in ridges:
            parts.append(f"  主要脊区: {loc}({lo:.0f}°E, {la:.0f}°N)，位势高度距平约{val:.0f}gpm")

    def _build_upper_features(self, parts, ds):
        try:
            u200 = _extract(ds, ["u", "ugrd"], level=200)
            v200 = _extract(ds, ["v", "vgrd"], level=200)
            lon2d, lat2d = _grid_coords(ds)
            jet_lat, jet_max = _jet_axis(u200, v200, lon2d, lat2d)
        except Exception:
            parts.append("")
            parts.append("【二、高空急流】数据不可用，跳过。")
            return

        parts.append("")
        parts.append("【二、高空急流】")
        parts.append("")
        parts.append(f"  200hPa急流轴大致位于 {jet_lat:.0f}°N 附近，急流核最大风速约{jet_max:.0f}m/s。")

        # 850hPa features
        try:
            t850 = _extract(ds, ["t", "tmp"], level=850)
            u850 = _extract(ds, ["u", "ugrd"], level=850)
            v850 = _extract(ds, ["v", "vgrd"], level=850)
            # Convert to Celsius if needed
            if np.nanmax(t850) > 100:
                t850 = t850 - 273.15

            # Temperature advection over eastern China
            adv = _temp_advection(t850, u850, v850, 0.25, 0.25)
            lon1d = lon2d[0, :]
            lat1d = lat2d[:, 0]
            ei = np.searchsorted(lon1d, 110)
            ej = np.searchsorted(lon1d, 125)
            si = np.searchsorted(lat1d, 25)
            sj = np.searchsorted(lat1d, 40)
            east_adv = adv[si:sj, ei:ej]

            adv_mean = float(np.nanmean(east_adv))
            if adv_mean > 0.5:
                adv_str = "以暖平流为主，有利于升温"
            elif adv_mean < -0.5:
                adv_str = "以冷平流为主，冷空气正在南下渗透"
            else:
                adv_str = "温度平流较弱，无明显冷暖空气交汇"

            # Sunan 850hPa wind
            u_sn = float(np.nanmean(u850[si:sj, ei:ej]))
            v_sn = float(np.nanmean(v850[si:sj, ei:ej]))
            ws_850 = np.sqrt(u_sn ** 2 + v_sn ** 2)
            dir_850 = _wind_dir_name(u_sn, v_sn)

            parts.append("")
            parts.append("【三、850hPa低层特征】")
            parts.append("")
            parts.append(f"  华东区域温度平流: {adv_str}。")
            parts.append(f"  苏南低空为{dir_850}风，风速约{ws_850:.0f}m/s。")
        except Exception:
            parts.append("")
            parts.append("【三、850hPa低层特征】数据不可用，跳过。")

    def _build_surface(self, parts, ds):
        try:
            msl = _extract(ds, ["msl", "prmsl", "mslma"])
            if np.nanmax(msl) > 5000:
                msl = msl / 100.0
            lon2d, lat2d = _grid_coords(ds)

            highs = _find_extrema(msl, lon2d, lat2d, n=2, find_max=True)
            lows = _find_extrema(-msl, lon2d, lat2d, n=2, find_max=True)

            parts.append("")
            parts.append("【四、地面气压形势】")
            parts.append("")
            for val, lo, la, loc in highs:
                parts.append(f"  高压中心: {loc}({lo:.0f}°E, {la:.0f}°N)，中心气压约{val:.0f}hPa")
            for val, lo, la, loc in lows:
                parts.append(f"  低压中心: {loc}({lo:.0f}°E, {la:.0f}°N)，中心气压约{-val:.0f}hPa")

            # Precipitation
            tp = _extract(ds, ["tp", "apcp"])
            if np.nanmax(tp) < 1.0:
                tp = tp * 1000.0
            tp_max = float(np.nanmax(tp))
            if tp_max < 0.1:
                parts.append("  全国无明显降水区。")
            else:
                yx = np.unravel_index(int(np.nanargmax(tp)), tp.shape)
                lo, la = float(lon2d[yx[0], yx[1]]), float(lat2d[yx[0], yx[1]])
                loc = _describe_location(lo, la)
                parts.append(f"  最大降水区: {loc}，累计降水量约{tp_max:.0f}mm。")
        except Exception:
            parts.append("")
            parts.append("【四、地面气压形势】数据不可用，跳过。")

    def _build_sunan_forecast(self, parts, ec_ds, gfs_ds, ec_date, ec_hour, gfs_date, gfs_hour):
        parts.append("")
        parts.append("【五、苏南地区三天预报】")
        parts.append("")

        today = datetime(ec_date.year, ec_date.month, ec_date.day, ec_hour, tzinfo=timezone.utc)

        for day_offset, label in [(0, "今日"), (1, "明日"), (2, "后天")]:
            target = today + timedelta(days=day_offset)
            step = day_offset * 24

            # Try ECMWF first, fall back to GFS
            ds = None
            for d, h, cache_dir, lbl in [
                (ec_date, ec_hour, self.ecmwf_dir, "EC"),
                (gfs_date, gfs_hour, self.gfs_dir, "GFS"),
            ]:
                ds = _load_nc(cache_dir, d.strftime("%Y%m%d"), h, step)
                if ds is not None:
                    break

            date_label = f"{target.month}月{target.day}日"
            if ds is None:
                parts.append(f"  {label}({date_label}): 数据暂缺")
                continue

            try:
                t2m = _extract(ds, ["2t", "t2m"])
                u10 = _extract(ds, ["10u", "u10"])
                v10 = _extract(ds, ["10v", "v10"])
                tp = _extract(ds, ["tp", "apcp"])

                if np.nanmax(t2m) > 100:
                    t2m = t2m - 273.15
                if np.nanmax(tp) < 1.0:
                    tp = tp * 1000.0

                lon2d, lat2d = _grid_coords(ds)
                t_mean, t_max, t_min = _subset_box(t2m, lon2d, lat2d, SUNAN_BOX)
                u_mean = float(np.nanmean(u10[(lat2d >= 31) & (lat2d <= 32.5) & (lon2d >= 119.5) & (lon2d <= 121.5)]))
                v_mean = float(np.nanmean(v10[(lat2d >= 31) & (lat2d <= 32.5) & (lon2d >= 119.5) & (lon2d <= 121.5)]))
                ws = np.sqrt(u_mean ** 2 + v_mean ** 2)
                wd = _wind_dir_name(u_mean, v_mean)
                bf = _beaufort(ws)
                tp_box = _subset_box(tp, lon2d, lat2d, SUNAN_BOX)[1]  # max

                # Weather description
                weather = self._describe_weather(t_mean, t_max, t_min, ws, tp_box)

                parts.append(
                    f"  {label}({date_label}): {weather}。"
                    f"气温{t_min:.0f}~{t_max:.0f}°C，"
                    f"{wd}风{bf}。"
                    f"降水量{tp_box:.1f}mm。"
                )
            except Exception as e:
                parts.append(f"  {label}({date_label}): 数据提取失败 ({e})")
            finally:
                ds.close()

            # After step=0 (analysis), use forecast steps
            if step == 0:
                step = 24

    def _build_model_diff(self, parts, ec_ds, gfs_ds, ec_date_str, ec_hour, gfs_date_str, gfs_hour):
        if ec_ds is None or gfs_ds is None:
            parts.append("")
            parts.append("【六、EC/GFS模式对比】数据不足，跳过。")
            return

        parts.append("")
        parts.append("【六、EC/GFS模式对比】")
        parts.append("")

        try:
            ec_msl = _extract(ec_ds, ["msl"])
            gfs_msl = _extract(gfs_ds, ["prmsl", "mslma", "msl"])
            if np.nanmax(ec_msl) > 5000:
                ec_msl /= 100
            if np.nanmax(gfs_msl) > 5000:
                gfs_msl /= 100

            # RMSE over China domain
            rmse = float(np.sqrt(np.nanmean((ec_msl - gfs_msl) ** 2)))
            if rmse < 2:
                parts.append("  两模式对地面气压场的分析基本一致（RMSE < 2hPa），大尺度环流形势无明显差异。")
            elif rmse < 5:
                parts.append(f"  两模式地面气压场存在轻微差异（RMSE≈{rmse:.0f}hPa），但对大尺度形势判断影响不大。")
            else:
                parts.append(f"  两模式地面气压场存在较明显差异（RMSE≈{rmse:.0f}hPa），预报需综合参考。")

            # Check 500hPa pattern difference
            try:
                ec_h500 = self._h500(ec_ds)
                gfs_h500 = self._h500(gfs_ds)
                h500_rmse = float(np.sqrt(np.nanmean((ec_h500 - gfs_h500) ** 2)))
                if h500_rmse < 15:
                    parts.append(f"  500hPa位势高度场一致性好（RMSE≈{h500_rmse:.0f}gpm）。")
                else:
                    parts.append(f"  500hPa位势高度场存在差异（RMSE≈{h500_rmse:.0f}gpm），槽脊位置和强度需关注更新预报。")
            except Exception:
                pass
        except Exception:
            parts.append("  模式对比暂时不可用。")

    def _build_disclaimer(self, parts):
        parts.append("")
        parts.append("───────────────────────────────────")
        parts.append("免责声明: 本预报由 AI 基于 ECMWF IFS")
        parts.append("及 GFS 全球数值预报产品自动生成，仅供")
        parts.append("学习参考。详细预报请关注中央气象台及")
        parts.append("江苏省气象台发布的最新预报。")
        parts.append("───────────────────────────────────")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _h500(self, ds) -> np.ndarray:
        h = _extract(ds, ["z", "gh", "hgt"], level=500)
        if np.nanmax(h) > 10000:
            h = h / 9.80665  # m²/s² → gpm
        return h

    @staticmethod
    def _describe_weather(t_mean, t_max, t_min, ws, precip) -> str:
        parts = []
        # Temperature description
        if t_min < 0:
            parts.append("有冰冻")
        elif t_min < 5:
            parts.append("偏冷")
        elif t_max > 35:
            parts.append("高温炎热")
        elif t_max > 30:
            parts.append("较热")
        elif t_mean > 20:
            parts.append("温暖舒适")
        elif t_mean > 10:
            parts.append("凉爽")
        else:
            parts.append("偏凉")

        # Wind
        if ws > 10.8:
            parts.append("风较大")
        elif ws > 5.5:
            parts.append("有风")
        else:
            parts.append("风力较小")

        # Precipitation
        if precip > 10:
            parts.append(f"有明显降水")
        elif precip > 1:
            parts.append("有小雨")
        elif precip > 0.1:
            parts.append("可能零星阵雨")
        else:
            parts.append("无降水")

        return "，".join(parts)
