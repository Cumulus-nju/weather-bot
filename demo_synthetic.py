"""
End-to-end demo with synthetic data.

Generates a realistic-looking temperature field across eastern China,
places ~80 virtual "stations" on it, adds noise, then shows:
  1. Pure interpolation (Barnes) from stations alone
  2. OI assimilation blending station obs with an ERA5-like background
  3. Side-by-side comparison of background vs analysis

No external API or data source required — everything is synthetic.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
from config import EAST_CHINA_EXTENT

from src.interpolation import make_grid, barnes
from src.assimilation import oi_analyze
from src.plotter import plot_grid, plot_comparison


# ---------------------------------------------------------------------------
# 1. Create a synthetic "true" temperature field
# ---------------------------------------------------------------------------

def make_true_field(lon_g, lat_g):
    """Generate a plausible temperature field with terrain and gradient effects."""
    # Meridional gradient: colder north
    lat_effect = -0.4 * (lat_g - 30)  # ~ -2°C at 35°N, -10°C at 50°N
    # Zonal gradient: colder inland (west)
    lon_effect = -0.15 * (lon_g - 118)  # cooler toward west

    # Simulated terrain effect: Gaussian "mountains"
    terrain = (
        8 * np.exp(-((lon_g - 112) ** 2 + (lat_g - 38) ** 2) / 8)  # Taihang/Shanxi
        + 5 * np.exp(-((lon_g - 102) ** 2 + (lat_g - 30) ** 2) / 10)  # Sichuan basin edge
        + 4 * np.exp(-((lon_g - 118) ** 2 + (lat_g - 26) ** 2) / 6)  # Wuyi mountains
    )
    temp = 20 + lat_effect + lon_effect - 0.6 * terrain

    # Add medium-scale waves
    wave = (3 * np.sin(lon_g / 4) * np.cos(lat_g / 5)
            + 2 * np.sin(lat_g / 8 + lon_g / 6)
            + 1.5 * np.cos(lon_g / 3 + lat_g / 3))
    temp += wave
    return temp


# ---------------------------------------------------------------------------
# 2. Place virtual stations and sample the true field
# ---------------------------------------------------------------------------

def place_stations(extent, n_stations=80, seed=42):
    """Place stations semi-randomly with some clustering near cities."""
    rng = np.random.default_rng(seed)
    lon_min, lon_max, lat_min, lat_max = extent

    # 60% random uniform, 40% clustered around "cities"
    n_random = int(n_stations * 0.6)
    n_cluster = n_stations - n_random

    lons = list(rng.uniform(lon_min, lon_max, n_random))
    lats = list(rng.uniform(lat_min, lat_max, n_random))

    # Cities to cluster around
    cities = [
        (116.4, 39.9),   # Beijing
        (121.5, 31.2),   # Shanghai
        (113.3, 23.1),   # Guangzhou
        (104.1, 30.6),   # Chengdu
        (114.3, 30.6),   # Wuhan
        (108.9, 34.3),   # Xi'an
        (120.2, 30.3),   # Hangzhou
        (118.8, 32.0),   # Nanjing
        (117.0, 36.7),   # Jinan
        (112.9, 28.2),   # Changsha
    ]
    cluster_std = 0.5  # degrees

    for _ in range(n_cluster):
        cx, cy = cities[rng.integers(0, len(cities))]
        lons.append(rng.normal(cx, cluster_std))
        lats.append(rng.normal(cy, cluster_std))

    return np.clip(lons, lon_min, lon_max), np.clip(lats, lat_min, lat_max)


def sample_stations(lon_g, lat_g, true_field, lons_s, lats_s, obs_error_std=0.8, seed=43):
    """Sample the true field at station locations, add observation noise."""
    from scipy.interpolate import RegularGridInterpolator
    rng = np.random.default_rng(seed)

    lon1d = lon_g[0, :]
    lat1d = lat_g[:, 0]
    # True field is on 2D grid — interpolate to station points
    # Note: lat1d is increasing (south→north), so we use it directly
    interp = RegularGridInterpolator(
        (lat1d, lon1d), true_field,
        bounds_error=False, fill_value=np.nan
    )

    obs_values = np.array([
        interp([lat, lon])[0] for lon, lat in zip(lons_s, lats_s)
    ])
    # Add noise
    obs_values += rng.normal(0, obs_error_std, len(obs_values))
    obs_errors = np.full(len(lons_s), obs_error_std)
    return obs_values, obs_errors


# ---------------------------------------------------------------------------
# 3. Generate a noisy background field (coarse ERA5-like)
# ---------------------------------------------------------------------------

def make_background(lon_g, lat_g, true_field, resolution_deg=0.25):
    """Simulate a background field: true field downsampled + smoothed + shifted."""
    from scipy.ndimage import gaussian_filter

    # Downsample
    step = int(resolution_deg / (lon_g[0, 1] - lon_g[0, 0]))
    if step < 1:
        step = 1
    coarse = true_field[::step, ::step]

    # Upsample back with cubic
    from scipy.interpolate import RegularGridInterpolator
    lon1d_fine = lon_g[0, :]
    lat1d_fine = lat_g[:, 0]
    lon1d_coarse = lon_g[0, ::step]
    lat1d_coarse = lat_g[::step, 0]

    interp = RegularGridInterpolator(
        (lat1d_coarse, lon1d_coarse), coarse,
        bounds_error=False, fill_value=np.nan
    )
    points = np.column_stack([lat_g.ravel(), lon_g.ravel()])
    bg = interp(points).reshape(lon_g.shape)

    # Smooth to make it look like a coarse analysis
    bg = gaussian_filter(bg, sigma=step * 1.5)

    # Add systematic bias and large-scale errors
    bg += 1.5 * np.sin(lon_g / 8) + 1.0 * np.cos(lat_g / 6)
    return bg


# ---------------------------------------------------------------------------
# 4. Run everything
# ---------------------------------------------------------------------------

def main():
    extent = EAST_CHINA_EXTENT
    grid_res = 0.05

    print("Creating grid ...")
    lon_g, lat_g = make_grid(extent, grid_res)
    print(f"  Grid shape: {lon_g.shape}")

    print("Generating synthetic true temperature field ...")
    true_temp = make_true_field(lon_g, lat_g)

    print("Placing virtual stations ...")
    lons_s, lats_s = place_stations(extent, n_stations=80)
    obs_values, obs_errors = sample_stations(lon_g, lat_g, true_temp, lons_s, lats_s)
    print(f"  {len(lons_s)} stations, obs error std = {obs_errors[0]:.1f}°C")

    print("Generating synthetic background (coarse, biased ERA5-like) ...")
    bg_field = make_background(lon_g, lat_g, true_temp)

    # ---- Barnes interpolation (pure station analysis) ----
    print("Running Barnes interpolation (station-only) ...")
    barnes_field = barnes(lons_s, lats_s, obs_values, lon_g, lat_g,
                          sigma=0.5, passes=2, max_scan=5.0)

    # ---- Optimal Interpolation ----
    print("Running OI assimilation (station + background) ...")
    analysis, increments = oi_analyze(
        lons_s, lats_s, obs_values, obs_errors,
        lon_g, lat_g, bg_field,
        bg_error_var=4.0, corr_length=2.0, max_scan=10.0
    )

    # ---- Plot results ----
    stations_dict = {"lon": lons_s, "lat": lats_s, "values": obs_values}

    print("\nPlotting: Barnes interpolation ...")
    plot_grid(lon_g, lat_g, barnes_field,
              title="Barnes Interpolation (station-only)",
              var_type="temperature", extent=extent,
              stations=stations_dict, levels=np.arange(-5, 36, 2),
              out_path="demo_barnes.png")

    print("Plotting: OI analysis vs background ...")
    plot_comparison(lon_g, lat_g, analysis, bg_field,
                    title="OI Analysis (station + ERA5 background)",
                    var_type="temperature",
                    out_path="demo_oi_comparison.png",
                    levels=np.arange(-5, 36, 2))

    # OI increments
    print("Plotting: OI increments (analysis - background) ...")
    plot_grid(lon_g, lat_g, increments,
              title="OI Increments (Analysis − Background)",
              var_type="default",
              extent=extent,
              out_path="demo_oi_increments.png",
              levels=21, cmap=None)

    print("\nDone. Figures saved to output/")
    print("  - demo_barnes.png")
    print("  - demo_oi_comparison.png")
    print("  - demo_oi_increments.png")


if __name__ == "__main__":
    main()
