import numpy as np
import pandas as pd
import xarray as xr


def _parse_datetime_cols(df, date_col="date", time_col="time"):
    """
    Robust datetime parser for your df_30min:
      date like 20050520 (int or str)
      time like 0, 30, 100, 1330 or "0030" etc.
    Returns pandas datetime64[ns] Series.
    """
    d = df[date_col].astype(str).str.strip()

    # time can be int minutes like 0/30, or HHMM like 100, or already "0030"
    t_raw = df[time_col]

    def _to_hhmm(x):
        # numeric
        if pd.isna(x):
            return None
        if isinstance(x, (int, np.integer)):
            return f"{int(x):04d}"
        if isinstance(x, float) and np.isfinite(x):
            return f"{int(x):04d}"
        # string
        s = str(x).strip()
        if s == "":
            return None
        # if it's "30" treat as 0030, if it's "100" treat as 0100, if it's "0030" keep
        if s.isdigit():
            return f"{int(s):04d}"
        return s  # last resort

    t = t_raw.map(_to_hhmm)
    dt = pd.to_datetime(d + t, format="%Y%m%d%H%M", errors="coerce")
    return dt


def df30min_to_netcdf_station_species(
    df_30min,
    station_dict,
    model_lat,
    model_lon,
    species,
    out_nc_path,
    mode=None,          # e.g. "A" or "HEIGHT" or None=keep all
    weighted=True,      # True -> use mean_w/std_w/cv_w ; False -> mean/std/cv
    sector_type_cum="CUM",
    sector_type_dist="DISTCUM",
):
    """
    Create one NetCDF for one station + one species:
      - coords: time, sector, distance_km
      - vars (sector): mean/std/cv/n (time, sector)
      - vars (distance): mean/std/cv/n (time, distance_km)
      - plus station/model metadata as attributes and scalar vars

    model_lat/model_lon should be the model grid cell lat/lon for the station.
    """

    df = df_30min.copy()

    # --- filter station ---
    st_name = station_dict.get("Station_Name", None)
    if st_name is None:
        raise ValueError("station_dict must include 'Station_Name' to filter df_30min.")
    df = df[df["station"] == st_name].copy()

    # --- filter mode if requested ---
    if mode is not None:
        df = df[df["mode"].astype(str).str.upper() == str(mode).upper()].copy()

    if df.empty:
        raise ValueError("No rows left after filtering by station/mode.")

    # --- time coordinate ---
    df["time_dt"] = _parse_datetime_cols(df, "date", "time")
    df = df[df["time_dt"].notna()].copy()
    if df.empty:
        raise ValueError("All timestamps failed to parse (check date/time columns).")

    # --- choose stat column names ---
    if weighted:
        col_mean, col_std, col_cv = "mean_w", "std_w", "cv_w"
        col_q1, col_q3, col_med, col_iqr = "q1_w", "q3_w", "median_w", "iqr_w"
    else:
        col_mean, col_std, col_cv = "mean", "std", "cv"
        col_q1, col_q3, col_med, col_iqr = "q25", "q75", "median", "iqr"

    # ensure numeric
    for c in ["radius", "n", col_mean, col_std, col_cv, "center_ppb"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # -------------------------
    # 1) CUM SECTORS -> (time, sector)
    # -------------------------
    df_c = df[df["sector_type"] == sector_type_cum].copy()
    if df_c.empty:
        raise ValueError(f"No rows with sector_type == {sector_type_cum}")

    # sector coordinate (strings like C1..Ck)
    # sort by radius (1..cell_nums) to keep consistent ordering
    df_c = df_c.sort_values(["time_dt", "radius"])

    sectors = (
        df_c[["sector", "radius"]]
        .drop_duplicates()
        .sort_values("radius")["sector"]
        .astype(str)
        .tolist()
    )

    # pivot helpers
    def _pivot(dfsub, value_col):
        wide = dfsub.pivot_table(
            index="time_dt",
            columns="sector",
            values=value_col,
            aggfunc="mean"
        )
        # enforce consistent sector order
        wide = wide.reindex(columns=sectors)
        return wide.sort_index()

    wide_mean_c = _pivot(df_c, col_mean)
    wide_std_c  = _pivot(df_c, col_std)
    wide_cv_c   = _pivot(df_c, col_cv)
    wide_n_c    = _pivot(df_c, "n")

    # optional quantiles
    wide_med_c = _pivot(df_c, col_med) if col_med in df_c.columns else None
    wide_q1_c  = _pivot(df_c, col_q1)  if col_q1  in df_c.columns else None
    wide_q3_c  = _pivot(df_c, col_q3)  if col_q3  in df_c.columns else None
    wide_iqr_c = _pivot(df_c, col_iqr) if col_iqr in df_c.columns else None

    # -------------------------
    # 2) DISTCUM -> (time, distance_km)
    # -------------------------
    df_d = df[df["sector_type"] == sector_type_dist].copy()
    if df_d.empty:
        raise ValueError(f"No rows with sector_type == {sector_type_dist}")

    df_d = df_d.sort_values(["time_dt", "radius"])
    # distance coordinate: numeric km, from radius
    distances = (
        df_d[["radius"]]
        .drop_duplicates()
        .sort_values("radius")["radius"]
        .astype(float)
        .tolist()
    )

    def _pivot_dist(dfsub, value_col):
        wide = dfsub.pivot_table(
            index="time_dt",
            columns="radius",
            values=value_col,
            aggfunc="mean"
        )
        wide = wide.reindex(columns=distances)
        return wide.sort_index()

    wide_mean_d = _pivot_dist(df_d, col_mean)
    wide_std_d  = _pivot_dist(df_d, col_std)
    wide_cv_d   = _pivot_dist(df_d, col_cv)
    wide_n_d    = _pivot_dist(df_d, "n")

    wide_med_d = _pivot_dist(df_d, col_med) if col_med in df_d.columns else None
    wide_q1_d  = _pivot_dist(df_d, col_q1)  if col_q1  in df_d.columns else None
    wide_q3_d  = _pivot_dist(df_d, col_q3)  if col_q3  in df_d.columns else None
    wide_iqr_d = _pivot_dist(df_d, col_iqr) if col_iqr in df_d.columns else None

    # -------------------------
    # Build Dataset
    # -------------------------
    time_index = wide_mean_c.index.union(wide_mean_d.index).sort_values()

    # align all wide tables to same time index
    def _align(wide):
        return wide.reindex(time_index)

    wide_mean_c, wide_std_c, wide_cv_c, wide_n_c = map(_align, [wide_mean_c, wide_std_c, wide_cv_c, wide_n_c])
    wide_mean_d, wide_std_d, wide_cv_d, wide_n_d = map(_align, [wide_mean_d, wide_std_d, wide_cv_d, wide_n_d])

    ds = xr.Dataset(
        coords={
            "time": ("time", time_index.to_numpy(dtype="datetime64[ns]")),
            "sector": ("sector", np.array(sectors, dtype=object)),
            "distance_km": ("distance_km", np.array(distances, dtype=float)),
        },
        data_vars={
            # station / model scalar vars
            "station_lat": ((), float(station_dict["Latitude"])),
            "station_lon": ((), float(station_dict["Longitude"])),
            "station_alt_m": ((), float(station_dict.get("Altitude", np.nan))),
            "model_lat": ((), float(model_lat)),
            "model_lon": ((), float(model_lon)),

            # sector stats: (time, sector)
            f"{species}_sector_mean": (("time", "sector"), wide_mean_c.to_numpy()),
            f"{species}_sector_std":  (("time", "sector"), wide_std_c.to_numpy()),
            f"{species}_sector_cv":   (("time", "sector"), wide_cv_c.to_numpy()),
            "sector_n":               (("time", "sector"), wide_n_c.to_numpy()),

            # distance stats: (time, distance_km)
            f"{species}_dist_mean": (("time", "distance_km"), wide_mean_d.to_numpy()),
            f"{species}_dist_std":  (("time", "distance_km"), wide_std_d.to_numpy()),
            f"{species}_dist_cv":   (("time", "distance_km"), wide_cv_d.to_numpy()),
            "dist_n":               (("time", "distance_km"), wide_n_d.to_numpy()),
        },
        attrs={
            "station_name": str(st_name),
            "species": str(species),
            "mode": str(mode) if mode is not None else "ALL",
            "weighted": int(bool(weighted)),
            "note": "sector= cumulative square rings C1..; distance_km = cumulative thresholds D<=distance_km",
        }
    )

    # add optional quantiles if available
    if wide_med_c is not None:
        ds[f"{species}_sector_median"] = (("time", "sector"), _align(wide_med_c).to_numpy())
    if wide_q1_c is not None:
        ds[f"{species}_sector_q1"] = (("time", "sector"), _align(wide_q1_c).to_numpy())
    if wide_q3_c is not None:
        ds[f"{species}_sector_q3"] = (("time", "sector"), _align(wide_q3_c).to_numpy())
    if wide_iqr_c is not None:
        ds[f"{species}_sector_iqr"] = (("time", "sector"), _align(wide_iqr_c).to_numpy())

    if wide_med_d is not None:
        ds[f"{species}_dist_median"] = (("time", "distance_km"), _align(wide_med_d).to_numpy())
    if wide_q1_d is not None:
        ds[f"{species}_dist_q1"] = (("time", "distance_km"), _align(wide_q1_d).to_numpy())
    if wide_q3_d is not None:
        ds[f"{species}_dist_q3"] = (("time", "distance_km"), _align(wide_q3_d).to_numpy())
    if wide_iqr_d is not None:
        ds[f"{species}_dist_iqr"] = (("time", "distance_km"), _align(wide_iqr_d).to_numpy())

    # add year/month/day/hour/minute as requested
    t = pd.to_datetime(ds["time"].values)
    ds["year"] = ("time", t.year.astype(np.int16))
    ds["month"] = ("time", t.month.astype(np.int8))
    ds["day"] = ("time", t.day.astype(np.int8))
    ds["hour"] = ("time", t.hour.astype(np.int8))
    ds["minute"] = ("time", t.minute.astype(np.int8))

    # compression (optional but recommended)
    comp = dict(zlib=True, complevel=4)
    encoding = {v: comp for v in ds.data_vars if ds[v].ndim > 0}

    ds.to_netcdf(out_nc_path, encoding=encoding)
    return ds
