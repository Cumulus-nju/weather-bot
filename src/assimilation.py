"""Optimal Interpolation — fuse station obs with a background field."""

import numpy as np
from scipy.spatial import KDTree
from src.interpolation import _euclidean_deg


def gaussian_correlation(dist_deg, length_scale):
    """Gaussian background-error correlation model."""
    return np.exp(-0.5 * (dist_deg / length_scale) ** 2)


def oi_analyze(lons_s, lats_s, obs_values, obs_errors,
               lon_g, lat_g, bg_field,
               bg_error_var=4.0, corr_length=2.0,
               max_stations=60, max_scan=10.0):
    """
    Optimal Interpolation on a regular grid.

    Parameters
    ----------
    lons_s, lats_s : 1D arrays, station coordinates (degrees).
    obs_values : 1D array of observed values.
    obs_errors : 1D array of observation error std (same units as obs).
    lon_g, lat_g : 2D arrays of grid coordinates.
    bg_field : 2D array of background values (must match grid shape).
    bg_error_var : background error variance.
    corr_length : background error correlation length scale (degrees).
    max_stations : max nearby stations to use.
    max_scan : max search radius (degrees).

    Returns
    -------
    analysis : 2D array.
    increments : 2D array (analysis - bg).
    """
    cos_mean = np.cos(np.radians(np.mean(lats_s)))
    tree = KDTree(np.column_stack([lons_s * cos_mean, lats_s]))
    ny, nx = lon_g.shape
    # Convert obs error std to variance
    obs_var = obs_errors ** 2
    analysis = bg_field.astype(np.float64).copy()
    increments = np.zeros_like(analysis)

    for j in range(ny):
        for i in range(nx):
            # Find nearby stations
            idxs = tree.query_ball_point([lon_g[j, i] * cos_mean, lat_g[j, i]], max_scan)
            if len(idxs) > max_stations:
                # Keep the closest max_stations
                dists_all = _euclidean_deg(lons_s[idxs], lats_s[idxs],
                                           lon_g[j, i], lat_g[j, i])
                order = np.argsort(dists_all)
                idxs = np.array(idxs)[order[:max_stations]]
            if len(idxs) == 0:
                continue

            n = len(idxs)
            # Build (B + R) matrix: B_ij = bg_err_var * correlation, R_ii = obs_err_var
            B_plus_R = np.zeros((n, n))
            for a, ia in enumerate(idxs):
                # Observation error (diagonal)
                B_plus_R[a, a] += obs_var[ia]
                for b in range(a, n):
                    ib = idxs[b]
                    h = _euclidean_deg(lons_s[ia], lats_s[ia],
                                       lons_s[ib], lats_s[ib])
                    cov = bg_error_var * gaussian_correlation(h, corr_length)
                    B_plus_R[a, b] += cov
                    if a != b:
                        B_plus_R[b, a] += cov

            # Background-to-observation covariances
            b_vec = np.zeros(n)
            for a, ia in enumerate(idxs):
                h = _euclidean_deg(lon_g[j, i], lat_g[j, i],
                                   lons_s[ia], lats_s[ia])
                b_vec[a] = bg_error_var * gaussian_correlation(h, corr_length)

            # Innovations: obs - H(bg)
            innovations = obs_values[idxs] - bg_field[j, i]

            # Solve: (B+R) * w = b_vec  →  weights
            try:
                w = np.linalg.solve(B_plus_R + 1e-10 * np.eye(n), b_vec)
                inc = np.dot(w, innovations)
                analysis[j, i] = bg_field[j, i] + inc
                increments[j, i] = inc
            except np.linalg.LinAlgError:
                continue

    return analysis, increments
