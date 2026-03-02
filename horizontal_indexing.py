# horizontal_indexing.py
import numpy as np
import pandas as pd


def nearest_grid_index(st_lat, st_lon, lats, lons):
    """
    Return nearest grid indices (i, j).

    Supports:
      - 1D lats and 1D lons
      - 2D lats and 2D lons (same shape)

    Raises:
      ValueError if shapes are not supported.
    """
    lats = np.asarray(lats)
    lons = np.asarray(lons)

    if lats.ndim == 1 and lons.ndim == 1:
        i = int(np.abs(lats - st_lat).argmin())
        j = int(np.abs(lons - st_lon).argmin())
        return i, j

    if lats.ndim == 2 and lons.ndim == 2 and lats.shape == lons.shape:
        dlat = lats - st_lat
        dlon = lons - st_lon
        dist2 = dlat**2 + dlon**2
        idx = int(np.argmin(dist2))
        i, j = np.unravel_index(idx, lats.shape)
        return int(i), int(j)

    raise ValueError("Unsupported lat/lon shapes for nearest_grid_index.")


def make_small_box_indices(i, j, Ny, Nx, cell_nums):
    """
    Compute the small box bounds and the local indices of the station inside that box.

    Returns:
      i1_s, i2_s, j1_s, j2_s: bounds in the FULL grid
      ii, jj: station indices inside the small box (0..Ny_s-1, 0..Nx_s-1)
    """
    i1_s = max(0, i - cell_nums)
    i2_s = min(Ny - 1, i + cell_nums)
    j1_s = max(0, j - cell_nums)
    j2_s = min(Nx - 1, j + cell_nums)
    ii = i - i1_s
    jj = j - j1_s
    return i1_s, i2_s, j1_s, j2_s, ii, jj


def add_distance_bins(df, radii_km):
    """
    Add non-cumulative distance-bin labels to a distance DataFrame.

    radii_km: e.g. [10, 20, 30]
    bins produced: (0–10], (10–20], (20–30]
    Adds:
      - bin_label: string label "0–10", "10–20", ...
      - dmax_km: the upper edge of the bin for convenience
    """
    radii_km = np.asarray(radii_km, dtype=float)
    edges = np.concatenate(([0.0], radii_km))

    labels = [f"{int(edges[i])}–{int(edges[i+1])}" for i in range(len(edges) - 1)]

    out = df.copy()
    out["bin_label"] = pd.cut(
        out["distance_km"],
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    out["dmax_km"] = out["bin_label"].map({lab: radii_km[i] for i, lab in enumerate(labels)})
    return out
