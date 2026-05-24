"""Core interpolation methods for station-to-grid mapping."""

import numpy as np
from scipy.spatial import KDTree


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def _haversine_deg(lon1, lat1, lon2, lat2):
    """Haversine distance on a sphere (lat/lon in degrees, result in km)."""
    R = 6371.0
    dlon = np.radians(lon2 - lon1)
    dlat = np.radians(lat2 - lat1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(np.maximum(a, 0)))


def _euclidean_deg(lon1, lat1, lon2, lat2):
    """Approx. Euclidean distance in degrees (fast, ok for regional grids)."""
    dx = (lon2 - lon1) * np.cos(np.radians((lat1 + lat2) / 2))
    dy = lat2 - lat1
    return np.sqrt(dx ** 2 + dy ** 2)


def _lonlat_to_km(lon, lat):
    """Return (kx, ky) scaling factors: km per degree lon/lat at given lat."""
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(np.radians(lat))
    return km_per_deg_lon, km_per_deg_lat


# ---------------------------------------------------------------------------
# IDW  (Inverse Distance Weighting)
# ---------------------------------------------------------------------------

def idw(lons_s, lats_s, values, lon_g, lat_g, power=2.0, min_neighbors=3,
        max_radius=None):
    """
    Inverse-distance-weighted interpolation.

    Parameters
    ----------
    lons_s, lats_s : 1D arrays of station coordinates (degrees).
    values : 1D array of station values.
    lon_g, lat_g : 2D arrays of grid coordinates.
    power : distance exponent (default 2).
    min_neighbors : grid points with fewer stations get NaN.
    max_radius : ignore stations beyond this distance (degrees).

    Returns
    -------
    grid : 2D array of interpolated values.
    """
    cos_mean = np.cos(np.radians(np.mean(lats_s)))
    tree = KDTree(np.column_stack([lons_s * cos_mean, lats_s]))
    ny, nx = lon_g.shape
    grid = np.full((ny, nx), np.nan, dtype=np.float64)

    for j in range(ny):
        for i in range(nx):
            # Find nearby stations
            if max_radius is not None:
                idxs = tree.query_ball_point([lon_g[j, i] * cos_mean, lat_g[j, i]], max_radius)
            else:
                # Take all stations (slow for large grids)
                idxs = list(range(len(lons_s)))
            if len(idxs) < min_neighbors:
                continue
            dists = _euclidean_deg(lons_s[idxs], lats_s[idxs], lon_g[j, i], lat_g[j, i])
            dists = np.maximum(dists, 1e-6)
            w = dists ** (-power)
            grid[j, i] = np.average(values[idxs], weights=w)
    return grid


# ---------------------------------------------------------------------------
# Cressman  successive-correction
# ---------------------------------------------------------------------------

def cressman(lons_s, lats_s, values, lon_g, lat_g, radii=None, min_neighbors=2):
    """
    Cressman multi-pass successive-correction scheme.

    Parameters
    ----------
    radii : list of influence radii in degrees (default [5, 3, 1.5]).
    """
    if radii is None:
        radii = [5.0, 3.0, 1.5]

    ny, nx = lon_g.shape
    cos_mean = np.cos(np.radians(np.mean(lats_s)))
    tree = KDTree(np.column_stack([lons_s * cos_mean, lats_s]))
    # First guess: simple mean of all stations
    bg = np.full((ny, nx), np.nanmean(values), dtype=np.float64)
    analysis = bg.copy()

    for R in radii:
        for j in range(ny):
            for i in range(nx):
                idxs = tree.query_ball_point([lon_g[j, i] * cos_mean, lat_g[j, i]], R)
                if len(idxs) < min_neighbors:
                    continue
                dists = _euclidean_deg(lons_s[idxs], lats_s[idxs], lon_g[j, i], lat_g[j, i])
                innovations = values[idxs] - analysis[j, i]
                # Cressman weight
                w = (R ** 2 - dists ** 2) / (R ** 2 + dists ** 2)
                sum_w = np.sum(w)
                if sum_w > 0:
                    correction = np.sum(w * innovations) / sum_w
                    analysis[j, i] += correction
    return analysis


# ---------------------------------------------------------------------------
# Barnes  two-pass
# ---------------------------------------------------------------------------

def _barnes_weight(dist_sq, sigma):
    return np.exp(-dist_sq / (2 * sigma ** 2))


def barnes(lons_s, lats_s, values, lon_g, lat_g, sigma=0.5, passes=2,
           gamma=0.3, max_scan=5.0):
    """
    Two-pass Barnes objective analysis.

    Parameters
    ----------
    sigma : Gaussian smoothing parameter (degrees).
    passes : number of passes (usually 2).
    gamma : convergence parameter for pass 2+ (0 < gamma < 1).
    max_scan : max distance to search for stations (degrees).

    Returns
    -------
    grid : 2D array.
    """
    cos_mean = np.cos(np.radians(np.mean(lats_s)))
    tree = KDTree(np.column_stack([lons_s * cos_mean, lats_s]))
    ny, nx = lon_g.shape
    bg = np.nanmean(values)
    field = np.full((ny, nx), bg, dtype=np.float64)

    for p in range(passes):
        field_new = np.full_like(field, bg)
        for j in range(ny):
            for i in range(nx):
                idxs = tree.query_ball_point([lon_g[j, i] * cos_mean, lat_g[j, i]], max_scan)
                if len(idxs) == 0:
                    field_new[j, i] = field[j, i]
                    continue
                # Euclidean distance squared in degrees
                dx = (lons_s[idxs] - lon_g[j, i]) * np.cos(np.radians((lats_s[idxs] + lat_g[j, i]) / 2))
                dy = lats_s[idxs] - lat_g[j, i]
                d2 = dx ** 2 + dy ** 2

                innovations = values[idxs] - field[j, i]
                w = _barnes_weight(d2, sigma)
                sum_w = np.sum(w)
                if sum_w > 1e-12:
                    correction = np.sum(w * innovations) / sum_w
                    field_new[j, i] = field[j, i] + correction
                else:
                    field_new[j, i] = field[j, i]
        field = field_new
        # Narrow sigma for subsequent passes
        sigma *= gamma
    return field


# ---------------------------------------------------------------------------
# Ordinary Kriging  (simple exponential variogram)
# ---------------------------------------------------------------------------

def _exp_variogram(h, sill=1.0, range_=1.0, nugget=0.0):
    """Exponential variogram model."""
    return nugget + sill * (1 - np.exp(-h / range_))


def ordinary_kriging(lons_s, lats_s, values, lon_g, lat_g,
                     variogram_range=2.0, sill=1.0, nugget=0.1,
                     max_stations=50, max_scan=8.0):
    """
    Ordinary kriging with exponential variogram.

    Parameters
    ----------
    variogram_range : range parameter in degrees.
    sill : partial sill.
    nugget : nugget variance.
    max_stations : max nearest stations to use per grid point.
    max_scan : max search radius (degrees).
    """
    cos_mean = np.cos(np.radians(np.mean(lats_s)))
    tree = KDTree(np.column_stack([lons_s * cos_mean, lats_s]))
    ny, nx = lon_g.shape
    grid = np.full((ny, nx), np.nan, dtype=np.float64)

    for j in range(ny):
        for i in range(nx):
            dists, idxs = tree.query([lon_g[j, i] * cos_mean, lat_g[j, i]], k=min(max_stations, len(lons_s)))
            # Filter by max_scan (but tree.query returns sorted by dist)
            if max_scan is not None:
                mask = dists <= max_scan
                dists = dists[mask]
                idxs = idxs[mask]
            if len(idxs) < 3:
                continue

            # Build kriging system
            n = len(idxs)
            # Covariance matrix among stations
            C = np.zeros((n, n))
            for a in range(n):
                for b in range(n):
                    h = _euclidean_deg(lons_s[idxs[a]], lats_s[idxs[a]],
                                       lons_s[idxs[b]], lats_s[idxs[b]])
                    C[a, b] = sill + nugget - _exp_variogram(h, sill, variogram_range, nugget)
            C += 1e-8 * np.eye(n)  # stabilise

            # RHS: cov between grid point and each station
            c0 = np.zeros(n)
            for a in range(n):
                h = _euclidean_deg(lon_g[j, i], lat_g[j, i],
                                   lons_s[idxs[a]], lats_s[idxs[a]])
                c0[a] = sill + nugget - _exp_variogram(h, sill, variogram_range, nugget)

            # Solve with Lagrange multiplier
            A = np.zeros((n + 1, n + 1))
            A[:n, :n] = C
            A[:n, n] = 1
            A[n, :n] = 1
            rhs = np.zeros(n + 1)
            rhs[:n] = c0
            rhs[n] = 1

            try:
                w = np.linalg.solve(A, rhs)[:n]
                grid[j, i] = np.dot(w, values[idxs])
            except np.linalg.LinAlgError:
                grid[j, i] = np.nan
    return grid


# ---------------------------------------------------------------------------
# Utility: generate regular lat/lon grid
# ---------------------------------------------------------------------------

def make_grid(extent, res):
    """Return (lon2d, lat2d) for the given extent and resolution in degrees."""
    lon_min, lon_max, lat_min, lat_max = extent
    lons_1d = np.arange(lon_min, lon_max + res, res)
    lats_1d = np.arange(lat_min, lat_max + res, res)
    lon2d, lat2d = np.meshgrid(lons_1d, lats_1d)
    return lon2d, lat2d
