"""Cartopy-based plotting for gridded meteorological fields.

Supports: scalar contour-fill, wind barbs, multi-panel composites.
"""

from datetime import datetime

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Chinese font support
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from pathlib import Path
from config import OUTPUT_DIR

# ---------------------------------------------------------------------------
# Colormaps
# ---------------------------------------------------------------------------

def _temp_colors():
    from matplotlib.colors import LinearSegmentedColormap
    colors = [
        (0.0, "#000080"), (0.15, "#0066cc"), (0.3, "#00ccff"),
        (0.4, "#66ff66"), (0.5, "#ffff00"), (0.6, "#ffcc00"),
        (0.7, "#ff6600"), (0.85, "#ff0000"), (1.0, "#990000"),
    ]
    return LinearSegmentedColormap.from_list("temp", colors)


def _precip_colors():
    from matplotlib.colors import LinearSegmentedColormap
    colors = [
        (0.0, "#ffffff"), (0.1, "#c0e8ff"), (0.2, "#90d0ff"),
        (0.3, "#60b8ff"), (0.4, "#30a0ff"), (0.5, "#00ff00"),
        (0.6, "#ffff00"), (0.7, "#ff9900"), (0.8, "#ff0000"),
        (0.9, "#cc00cc"), (1.0, "#660066"),
    ]
    return LinearSegmentedColormap.from_list("precip", colors)


def _humidity_colors():
    from matplotlib.colors import LinearSegmentedColormap
    colors = [
        (0.0, "#8B4513"), (0.2, "#D2B48C"), (0.35, "#F5F5DC"),
        (0.5, "#87CEEB"), (0.65, "#4169E1"), (0.85, "#0000CD"),
        (1.0, "#000080"),
    ]
    return LinearSegmentedColormap.from_list("humidity", colors)


CMAPS = {
    "temperature":   _temp_colors(),
    "precipitation": _precip_colors(),
    "humidity":      _humidity_colors(),
    "wind_speed":    plt.cm.viridis_r,
    "wind":          plt.cm.viridis_r,
    "pressure":      plt.cm.Spectral_r,
    "default":       plt.cm.coolwarm,
}

VAR_LABELS = {
    "temperature":   "温度 (°C)",
    "precipitation": "降水量 (mm)",
    "wind_speed":    "风速 (m/s)",
    "wind":          "风速 (m/s)",
    "pressure":      "气压 (hPa)",
    "humidity":      "相对湿度 (%)",
}

# Fixed color ranges so that same color = same value across all times
FIXED_RANGES = {
    "temperature":   (-10, 40),
    "precipitation": (0, 25),
    "wind_speed":    (0, 15),
    "wind":          (0, 15),
    "pressure":      (990, 1040),
    "humidity":      (10, 100),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _map_features(ax):
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#333333")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="--", edgecolor="#666666")
    ax.add_feature(cfeature.OCEAN, facecolor="#e8f4f8", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#f5f5f0", zorder=0)
    ax.add_feature(cfeature.LAKES, facecolor="#d4e8f0", linewidth=0.2, zorder=1)


def _gridlines(ax):
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#999999", linestyle=":")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.MaxNLocator(nbins=6)
    gl.ylocator = mticker.MaxNLocator(nbins=6)


def _nice_ticks(vmin, vmax, target_n=12):
    """Return float ticks at nice intervals (0.5, 1, 2, 5, 10, ...)."""
    span = vmax - vmin
    if span <= 0:
        return np.array([vmin, vmax])
    # Pick step size from [0.5, 1, 2, 5, 10, 20, 50, ...]
    raw_step = span / target_n
    magnitude = 10 ** np.floor(np.log10(raw_step))
    for s in [0.5, 1, 2, 5, 10]:
        candidate = s * magnitude
        if candidate >= raw_step:
            break
    else:
        candidate = 10 * magnitude
    lo = np.floor(vmin / candidate) * candidate
    hi = np.ceil(vmax / candidate) * candidate
    return np.arange(lo, hi + candidate * 0.5, candidate)


def _tick_format(vmin, vmax):
    """Return format string with appropriate decimal precision."""
    span = vmax - vmin
    if span <= 2:
        return "%.1f"
    elif span <= 10:
        return "%.1f"
    elif span <= 50:
        return "%.0f"
    else:
        return "%.0f"


def _colorbar(fig, ax, cf, label, vmin, vmax):
    ticks = _nice_ticks(vmin, vmax)
    fmt = _tick_format(vmin, vmax)
    cbar = fig.colorbar(cf, ax=ax, orientation="horizontal",
                        pad=0.05, shrink=0.8, aspect=30, ticks=ticks)
    cbar.set_label(label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_xticklabels([fmt % t for t in ticks])


def _timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _station_labels(ax, stations, vmin, vmax):
    """Add small temperature labels next to each station dot."""
    if stations is None:
        return
    lons = stations.get("lon", [])
    lats = stations.get("lat", [])
    values = stations.get("values", [])
    names = stations.get("names", [])

    for i, (x, y, v) in enumerate(zip(lons, lats, values)):
        # offset text slightly to the right of the dot
        name = names[i] if i < len(names) else ""
        # show name:value
        label = f"{name} {v:.0f}"
        ax.annotate(label, (x, y), xytext=(6, 2), textcoords="offset points",
                    fontsize=5, color="#222222",
                    transform=ccrs.PlateCarree(), zorder=6,
                    bbox=dict(facecolor="white", alpha=0.65, pad=1, lw=0))


def _extremes_box(ax, stations, var_label):
    """Add min/max annotation box at top-left of the axes area."""
    if stations is None:
        return
    values = stations.get("values", [])
    names = stations.get("names", [])
    if len(values) == 0:
        return

    max_idx = int(np.argmax(values))
    min_idx = int(np.argmin(values))
    max_name = names[max_idx] if max_idx < len(names) else "?"
    min_name = names[min_idx] if min_idx < len(names) else "?"

    text = (
        f"最高: {values[max_idx]:.1f}  ({max_name})\n"
        f"最低: {values[min_idx]:.1f}  ({min_name})"
    )
    ax.text(0.02, 1.02, text, transform=ax.transAxes,
            fontsize=9, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#999999", alpha=0.9),
            zorder=10)


# ---------------------------------------------------------------------------
# Scalar contour plot
# ---------------------------------------------------------------------------

def plot_grid(lon_g, lat_g, field, title="", var_type="default",
              extent=None, stations=None, bg_field=None, bg_label="",
              out_path=None, levels=None, cmap=None, dpi=150):
    if isinstance(levels, (list, np.ndarray)):
        vmin, vmax = levels[0], levels[-1]
    elif var_type in FIXED_RANGES:
        vmin, vmax = FIXED_RANGES[var_type]
    else:
        vmin = np.nanpercentile(field, 2)
        vmax = np.nanpercentile(field, 98)
        if vmax - vmin < 0.1:
            vmax = vmin + 5

    nlev = 80
    levels_arr = np.linspace(vmin, vmax, nlev)

    cmap = cmap or CMAPS.get(var_type, CMAPS["default"])
    cbar_label = VAR_LABELS.get(var_type, var_type)

    ncols = 2 if bg_field is not None else 1
    fig = plt.figure(figsize=(8 * ncols + 1, 9))
    proj = ccrs.PlateCarree()

    for idx, (data, sub_title) in enumerate(
        [(field, title),
         (bg_field, bg_label or "Background")], start=1):
        if data is None:
            continue

        ax = fig.add_subplot(1, ncols, idx, projection=proj)
        ax.set_extent(extent or [lon_g.min(), lon_g.max(), lat_g.min(), lat_g.max()], crs=proj)
        _map_features(ax)

        cf = ax.contourf(lon_g, lat_g, data, levels=levels_arr, cmap=cmap,
                         transform=proj, extend="both", antialiased=True)

        ticks = _nice_ticks(vmin, vmax)
        fmt = _tick_format(vmin, vmax)
        cs = ax.contour(lon_g, lat_g, data, levels=ticks, colors="#222222",
                        linewidths=0.5, transform=proj)
        ax.clabel(cs, fontsize=7, fmt=fmt)

        # Station overlay with labels
        if stations and idx == 1:
            sv = stations.get("values", [0] * len(stations["lon"]))
            ax.scatter(stations["lon"], stations["lat"], c=sv,
                       s=35, cmap=cmap, edgecolors="black",
                       linewidths=0.5, transform=proj, zorder=5,
                       vmin=vmin, vmax=vmax)
            _station_labels(ax, stations, vmin, vmax)
            _extremes_box(ax, stations, cbar_label)

        _gridlines(ax)

        ts = _timestamp()
        ax.set_title(f"{sub_title}\n数据更新时间: {ts}", fontsize=11, weight="bold",
                     linespacing=1.4)

        _colorbar(fig, ax, cf, cbar_label, vmin, vmax)

    fig.tight_layout()

    if out_path:
        out_dir = Path(OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        full = out_dir / out_path
        fig.savefig(full, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return str(full)
    else:
        plt.show()
        return None


# ---------------------------------------------------------------------------
# Side-by-side comparison
# ---------------------------------------------------------------------------

def plot_comparison(lon_g, lat_g, analysis, background, title="",
                    var_type="default", out_path=None, levels=None, dpi=150):
    return plot_grid(lon_g, lat_g, analysis, title=title,
                     var_type=var_type, bg_field=background,
                     bg_label="ERA5 背景场", out_path=out_path,
                     levels=levels, dpi=dpi)


# ---------------------------------------------------------------------------
# Wind barbs
# ---------------------------------------------------------------------------

def plot_wind_barbs(lon_g, lat_g, speed, u, v, title="", extent=None,
                    out_path=None, dpi=150, skip=6):
    vr = FIXED_RANGES.get("wind_speed", (None, None))
    vmin, vmax = vr[0] if vr[0] is not None else np.nanpercentile(speed, 2), \
                 vr[1] if vr[1] is not None else np.nanpercentile(speed, 98)
    nlev = 80
    levels_arr = np.linspace(vmin, vmax, nlev)

    fig = plt.figure(figsize=(10, 10))
    proj = ccrs.PlateCarree()
    ax = fig.add_subplot(111, projection=proj)
    ax.set_extent(extent or [lon_g.min(), lon_g.max(), lat_g.min(), lat_g.max()], crs=proj)
    _map_features(ax)

    cmap = CMAPS["wind_speed"]
    cf = ax.contourf(lon_g, lat_g, speed, levels=levels_arr, cmap=cmap,
                     transform=proj, extend="max", antialiased=True)

    ticks = _nice_ticks(vmin, vmax)
    fmt = _tick_format(vmin, vmax)
    cs = ax.contour(lon_g, lat_g, speed, levels=ticks, colors="#222222",
                    linewidths=0.4, transform=proj)
    ax.clabel(cs, fontsize=7, fmt=fmt)

    ny, nx = lon_g.shape
    sl = slice(skip // 2, ny, skip), slice(skip // 2, nx, skip)
    ax.barbs(lon_g[sl], lat_g[sl], u[sl], v[sl],
             transform=proj, length=5, linewidth=0.5, color="#222222", zorder=4)

    ts = _timestamp()
    ax.set_title(f"{title}\n数据更新时间: {ts}", fontsize=12, weight="bold",
                 linespacing=1.4)

    _gridlines(ax)
    _colorbar(fig, ax, cf, VAR_LABELS["wind_speed"], vmin, vmax)
    fig.tight_layout()

    if out_path:
        out_dir = Path(OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        full = out_dir / out_path
        fig.savefig(full, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return str(full)
    else:
        plt.show()
        return None


# ---------------------------------------------------------------------------
# Multi-panel composite (2x2)
# ---------------------------------------------------------------------------

def plot_multi_panel(lon_g, lat_g, fields, title="长三角天气综合分析",
                     extent=None, out_path=None, dpi=150):
    panels = [
        ("temperature",   "温度 (°C)",      CMAPS["temperature"],  21),
        ("humidity",      "相对湿度 (%)",   CMAPS["humidity"],     21),
        ("pressure",      "气压 (hPa)",     CMAPS["pressure"],     21),
        ("wind_speed",    "风速 (m/s)",     CMAPS["wind_speed"],   21),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14),
                             subplot_kw={"projection": ccrs.PlateCarree()})
    axes = axes.flatten()
    proj = ccrs.PlateCarree()

    for ax, (key, panel_label, cmap, _) in zip(axes, panels):
        data = fields.get(key)
        if data is None:
            ax.set_visible(False)
            continue

        ax.set_extent(extent or [lon_g.min(), lon_g.max(), lat_g.min(), lat_g.max()], crs=proj)
        _map_features(ax)

        vr = FIXED_RANGES.get(key, (None, None))
        vmin = vr[0] if vr[0] is not None else np.nanpercentile(data, 2)
        vmax = vr[1] if vr[1] is not None else np.nanpercentile(data, 98)
        if vmax - vmin < 0.1:
            vmax = vmin + 5
        levels_arr = np.linspace(vmin, vmax, 60)

        cf = ax.contourf(lon_g, lat_g, data, levels=levels_arr, cmap=cmap,
                         transform=proj, extend="both", antialiased=True)
        ticks = _nice_ticks(vmin, vmax)
        fmt = _tick_format(vmin, vmax)
        cs = ax.contour(lon_g, lat_g, data, levels=ticks, colors="#222222",
                        linewidths=0.3, transform=proj)
        ax.clabel(cs, fontsize=6, fmt=fmt)
        _gridlines(ax)
        ax.set_title(panel_label, fontsize=11, weight="bold")
        _colorbar(fig, ax, cf, panel_label, vmin, vmax)

    ts = _timestamp()
    fig.suptitle(f"{title}\n数据更新时间: {ts}", fontsize=14, weight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if out_path:
        out_dir = Path(OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        full = out_dir / out_path
        fig.savefig(full, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return str(full)
    else:
        plt.show()
        return None
