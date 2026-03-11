#%%
import os
from pathlib import Path
import calendar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
#%%
# ============================================================
# 1. STATIONS LOADER
# ============================================================
def load_stations_file(stations_path: str) -> pd.DataFrame:
    df = pd.read_csv(stations_path, sep="\t")
    df = df.reset_index().rename(columns={"index": "idx"})

    rename_map = {}
    for col in df.columns:
        c = col.lower()
        if c.startswith("station"):
            rename_map[col] = "Station_Name"
        elif c.startswith("lat"):
            rename_map[col] = "Latitude"
        elif c.startswith("lon"):
            rename_map[col] = "Longitude"
        elif c.startswith("alt"):
            rename_map[col] = "Altitude"

    df = df.rename(columns=rename_map)

    expected = ["idx", "Station_Name", "Latitude", "Longitude", "Altitude"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Stations file missing expected columns: {missing}")

    for col in ["Latitude", "Longitude", "Altitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["is_valid"] = df[["Latitude", "Longitude", "Altitude"]].notna().all(axis=1)

    return df[expected + ["is_valid"]]


# ============================================================
# 2. DATA LOADER
# ============================================================
def load_station_timeseries_csv(csv_path: str, station_name: str = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")

        if ts.isna().all() and {"date", "time"}.issubset(df.columns):
            date_str = df["date"].astype(str).str.zfill(8)
            time_str = df["time"].astype(str).str.zfill(4)
            ts = pd.to_datetime(date_str + time_str, format="%Y%m%d%H%M", errors="coerce")

        df["timestamp"] = ts

    elif {"date", "time"}.issubset(df.columns):
        date_str = df["date"].astype(str).str.zfill(8)
        time_str = df["time"].astype(str).str.zfill(4)
        df["timestamp"] = pd.to_datetime(date_str + time_str, format="%Y%m%d%H%M", errors="coerce")
    else:
        raise ValueError("Need either 'timestamp' or both 'date' and 'time' columns.")

    if df["timestamp"].isna().all():
        raise ValueError(f"All timestamps failed to parse for file: {csv_path}")

    if "station" not in df.columns:
        if station_name is None:
            stem = Path(csv_path).stem
            station_name = stem.split("_")[0]
        df["station"] = station_name

    return df


def load_many_station_csvs(
    input_dir: str,
    species: str,
    mode: str,
    station_names,
    pattern_template: str = "{station}_{species}_{mode}_30min.csv"
) -> pd.DataFrame:
    input_dir = Path(input_dir)
    all_dfs = []

    for st in station_names:
        filename = pattern_template.format(station=st, species=species, mode=mode)
        csv_path = input_dir / filename

        if not csv_path.exists():
            print(f"Skipping missing file: {csv_path}")
            continue

        tmp = load_station_timeseries_csv(str(csv_path), station_name=st)
        all_dfs.append(tmp)

    if not all_dfs:
        raise ValueError("No station CSVs were loaded.")

    return pd.concat(all_dfs, ignore_index=True)


# ============================================================
# 3. TIME FEATURES
# ============================================================
def month_to_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Autumn"
    return np.nan


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"]).copy()

    out["year"] = out["timestamp"].dt.year
    out["month"] = out["timestamp"].dt.month
    out["month_name"] = out["month"].map(lambda x: calendar.month_abbr[x])
    out["season"] = out["month"].map(month_to_season)
    out["day"] = out["timestamp"].dt.day
    out["hour"] = out["timestamp"].dt.hour
    out["minute"] = out["timestamp"].dt.minute
    return out


# ============================================================
# 4. MERGE STATION METADATA
# ============================================================
def attach_station_metadata(df: pd.DataFrame, stations_df: pd.DataFrame) -> pd.DataFrame:
    meta = stations_df.rename(columns={"Station_Name": "station"})
    return df.merge(
        meta[["station", "Latitude", "Longitude", "Altitude", "is_valid"]],
        on="station",
        how="left"
    )


# ============================================================
# 5. HELPERS
# ============================================================
def ensure_dir(path: str):
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)


def compute_basic_stats(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"count": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "median": np.nan, "max": np.nan}
    return {
        "count": int(s.count()),
        "mean": s.mean(),
        "std": s.std(),
        "min": s.min(),
        "median": s.median(),
        "max": s.max(),
    }


def format_stats_text(stats: dict) -> str:
    return (
        f"n={stats['count']}\n"
        f"mean={stats['mean']:.3f}\n"
        f"std={stats['std']:.3f}\n"
        f"min={stats['min']:.3f}\n"
        f"median={stats['median']:.3f}\n"
        f"max={stats['max']:.3f}"
    )


def safe_corr(x: pd.Series, y: pd.Series, method: str = "pearson") -> float:
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(tmp) < 2:
        return np.nan
    return tmp["x"].corr(tmp["y"], method=method)


def get_sector_order(sectors):
    def sector_key(x):
        s = str(x)
        digits = "".join(ch for ch in s if ch.isdigit())
        return int(digits) if digits else 9999
    return sorted(sectors, key=sector_key)


def make_red_pink_gradient(n=10):
    colors = []
    red_vals = np.linspace(0.85, 1.0, n)
    gb_vals = np.linspace(0.25, 0.85, n)
    for r, gb in zip(red_vals, gb_vals):
        colors.append((r, gb, gb))
    return [to_hex(c) for c in colors]


# ============================================================
# 6. DATA PREPARATION
# ============================================================
def prepare_center_timeseries(df: pd.DataFrame, stations=None, center_col="center_ppb", mode=None) -> pd.DataFrame:
    out = df.copy()

    if stations is not None:
        if isinstance(stations, str):
            stations = [stations]
        out = out[out["station"].isin(stations)]

    if mode is not None and "mode" in out.columns:
        out = out[out["mode"] == mode]

    out = out.dropna(subset=["timestamp", center_col]).copy()

    # one center value per station/timestamp
    out = (
        out.groupby(["station", "timestamp"], as_index=False)[center_col]
        .mean()
        .sort_values(["station", "timestamp"])
    )
    out = add_time_features(out)
    return out


def prepare_cv_by_sector(df: pd.DataFrame, station: str, cv_col="cv_w", sector_col="sector", sector_type=None, mode=None):
    out = df.copy()
    out = out[out["station"] == station]

    if mode is not None and "mode" in out.columns:
        out = out[out["mode"] == mode]

    if sector_type is not None and "sector_typ" in out.columns:
        out = out[out["sector_typ"] == sector_type]

    out = out.dropna(subset=["timestamp", cv_col, sector_col]).copy()
    out = add_time_features(out)
    return out.sort_values(["season", "timestamp", sector_col])


def prepare_center_cv_pairs(
    df: pd.DataFrame,
    stations=None,
    center_col="center_ppb",
    cv_col="cv_w",
    mode=None,
    sector_type=None,
    aggregate_cv_over_sectors=True
) -> pd.DataFrame:
    out = df.copy()

    if stations is not None:
        if isinstance(stations, str):
            stations = [stations]
        out = out[out["station"].isin(stations)]

    if mode is not None and "mode" in out.columns:
        out = out[out["mode"] == mode]

    if sector_type is not None and "sector_typ" in out.columns:
        out = out[out["sector_typ"] == sector_type]

    out = out.dropna(subset=["timestamp", center_col, cv_col]).copy()

    if aggregate_cv_over_sectors:
        # one row per station/timestamp
        out = (
            out.groupby(["station", "timestamp"], as_index=False)
            .agg(
                center_ppb=(center_col, "mean"),
                cv_w=(cv_col, "mean")
            )
        )
    else:
        # keep one row per sector; center_ppb repeats across sectors
        out = out.rename(columns={center_col: "center_ppb", cv_col: "cv_w"})

    out = add_time_features(out)
    return out


# ============================================================
# 7. STATISTICS TABLES
# ============================================================
def monthly_stats_table(df: pd.DataFrame, value_cols, station_col="station") -> pd.DataFrame:
    tmp = add_time_features(df)

    agg = {col: ["count", "mean", "std", "min", "median", "max"] for col in value_cols if col in tmp.columns}
    out = tmp.groupby([station_col, "month", "month_name"]).agg(agg).reset_index()

    flat_cols = []
    for c in out.columns:
        if isinstance(c, tuple):
            if c[1] == "":
                flat_cols.append(c[0])
            else:
                flat_cols.append(f"{c[0]}_{c[1]}")
        else:
            flat_cols.append(c)
    out.columns = flat_cols

    return out.sort_values([station_col, "month"]).reset_index(drop=True)


def seasonal_stats_table(df: pd.DataFrame, value_cols, station_col="station") -> pd.DataFrame:
    tmp = add_time_features(df)

    agg = {col: ["count", "mean", "std", "min", "median", "max"] for col in value_cols if col in tmp.columns}
    out = tmp.groupby([station_col, "season"]).agg(agg).reset_index()

    flat_cols = []
    for c in out.columns:
        if isinstance(c, tuple):
            if c[1] == "":
                flat_cols.append(c[0])
            else:
                flat_cols.append(f"{c[0]}_{c[1]}")
        else:
            flat_cols.append(c)
    out.columns = flat_cols

    season_order = pd.CategoricalDtype(["Winter", "Spring", "Summer", "Autumn"], ordered=True)
    out["season"] = out["season"].astype(season_order)
    out = out.sort_values([station_col, "season"]).reset_index(drop=True)
    return out


def correlation_by_month(df_pairs: pd.DataFrame, x_col="center_ppb", y_col="cv_w", station_col="station", method="pearson"):
    out = (
        df_pairs.groupby([station_col, "month", "month_name"])
        .apply(lambda g: pd.Series({
            "n": len(g[[x_col, y_col]].dropna()),
            "corr": safe_corr(g[x_col], g[y_col], method=method)
        }))
        .reset_index()
        .sort_values([station_col, "month"])
        .reset_index(drop=True)
    )
    return out


def correlation_by_season(df_pairs: pd.DataFrame, x_col="center_ppb", y_col="cv_w", station_col="station", method="pearson"):
    out = (
        df_pairs.groupby([station_col, "season"])
        .apply(lambda g: pd.Series({
            "n": len(g[[x_col, y_col]].dropna()),
            "corr": safe_corr(g[x_col], g[y_col], method=method)
        }))
        .reset_index()
    )
    season_order = pd.CategoricalDtype(["Winter", "Spring", "Summer", "Autumn"], ordered=True)
    out["season"] = out["season"].astype(season_order)
    out = out.sort_values([station_col, "season"]).reset_index(drop=True)
    return out


def altitude_relation_table(
    df: pd.DataFrame,
    stations=None,
    center_col="center_ppb",
    cv_col="cv_w",
    mode=None,
    sector_type=None
) -> pd.DataFrame:
    out = df.copy()

    if stations is not None:
        if isinstance(stations, str):
            stations = [stations]
        out = out[out["station"].isin(stations)]

    if mode is not None and "mode" in out.columns:
        out = out[out["mode"] == mode]

    if sector_type is not None and "sector_typ" in out.columns:
        out = out[out["sector_typ"] == sector_type]

    grouped = (
        out.groupby("station", as_index=False)
        .agg(
            Altitude=("Altitude", "first"),
            center_mean=(center_col, "mean"),
            center_std=(center_col, "std"),
            center_cv=(center_col, lambda s: np.nan if pd.to_numeric(s, errors="coerce").dropna().mean() == 0 else pd.to_numeric(s, errors="coerce").dropna().std() / pd.to_numeric(s, errors="coerce").dropna().mean()),
            cv_w_mean=(cv_col, "mean"),
            cv_w_std=(cv_col, "std"),
        )
    )
    return grouped


# ============================================================
# 8. PLOTS: SEASONAL TIMESERIES
# ============================================================
def plot_one_station_seasonal_center(
    df, station, stations_df=None, center_col="center_ppb", mode=None,
    species="O3", units="ppb", out_path=None, figsize=(16, 10)
):
    data = prepare_center_timeseries(df, stations=station, center_col=center_col, mode=mode)

    altitude = np.nan
    if stations_df is not None:
        row = stations_df.loc[stations_df["Station_Name"] == station]
        if not row.empty:
            altitude = row["Altitude"].iloc[0]

    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharey=True)
    axes = axes.ravel()

    for ax, season in zip(axes, seasons):
        sub = data[data["season"] == season].sort_values("timestamp")

        if sub.empty:
            ax.set_title(f"{season} (no data)")
            ax.grid(True, alpha=0.3)
            continue

        ax.plot(sub["timestamp"], sub[center_col], label=station)
        stats = compute_basic_stats(sub[center_col])

        ax.text(
            0.02, 0.98, format_stats_text(stats),
            transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

        title = f"{season} | {station}"
        if pd.notna(altitude):
            title += f" ({altitude:.0f} m)"

        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel(f"{species} ({units})")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{species} center pixel seasonal timeseries - {station}", y=1.02)
    fig.tight_layout()
    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig, axes


def plot_multi_station_seasonal_center(
    df, stations, stations_df=None, center_col="center_ppb", mode=None,
    species="O3", units="ppb", out_path=None, figsize=(16, 10)
):
    data = prepare_center_timeseries(df, stations=stations, center_col=center_col, mode=mode)

    alt_map = {}
    if stations_df is not None:
        alt_map = dict(zip(stations_df["Station_Name"], stations_df["Altitude"]))

    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharey=True)
    axes = axes.ravel()

    for ax, season in zip(axes, seasons):
        sub = data[data["season"] == season].copy()

        if sub.empty:
            ax.set_title(f"{season} (no data)")
            ax.grid(True, alpha=0.3)
            continue

        for st, g in sub.groupby("station"):
            label = f"{st}"
            alt = alt_map.get(st, np.nan)
            if pd.notna(alt):
                label += f" ({alt:.0f} m)"
            ax.plot(g["timestamp"], g[center_col], label=label)

        stats = compute_basic_stats(sub[center_col])
        ax.text(
            0.02, 0.98, format_stats_text(stats),
            transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

        ax.set_title(season)
        ax.set_xlabel("Time")
        ax.set_ylabel(f"{species} ({units})")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"{species} center pixel seasonal timeseries - multiple stations", y=1.02)
    fig.tight_layout()
    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig, axes


# ============================================================
# 9. PLOT: CV_W BY SECTOR PER SEASON
# ============================================================
def plot_one_station_seasonal_cv_by_sector(
    df, station, stations_df=None, cv_col="cv_w", sector_col="sector",
    sector_type="CUM", mode=None, species="O3", out_path=None, figsize=(16, 10)
):
    data = prepare_cv_by_sector(df, station=station, cv_col=cv_col, sector_col=sector_col, sector_type=sector_type, mode=mode)

    altitude = np.nan
    if stations_df is not None:
        row = stations_df.loc[stations_df["Station_Name"] == station]
        if not row.empty:
            altitude = row["Altitude"].iloc[0]

    all_sectors = get_sector_order(data[sector_col].dropna().unique())
    colors = make_red_pink_gradient(len(all_sectors))
    color_map = dict(zip(all_sectors, colors))

    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharey=True)
    axes = axes.ravel()

    for ax, season in zip(axes, seasons):
        sub = data[data["season"] == season].copy()

        if sub.empty:
            ax.set_title(f"{season} (no data)")
            ax.grid(True, alpha=0.3)
            continue

        for sec in all_sectors:
            ss = sub[sub[sector_col] == sec].sort_values("timestamp")
            if ss.empty:
                continue
            ax.plot(ss["timestamp"], ss[cv_col], label=str(sec), color=color_map[sec])

        title = f"{season} | {station}"
        if pd.notna(altitude):
            title += f" ({altitude:.0f} m)"

        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel(cv_col)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Sector", fontsize=8, ncol=2)

    fig.suptitle(f"{species} seasonal {cv_col} by sector - {station} - sector_type={sector_type}", y=1.02)
    fig.tight_layout()
    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig, axes


# ============================================================
# 10. BOXPLOTS: MONTH / SEASON
# ============================================================
def plot_monthly_boxplot(df, value_col, stations=None, mode=None, sector_type=None, title=None, ylabel=None, out_path=None, figsize=(12, 6)):
    out = df.copy()

    if stations is not None:
        if isinstance(stations, str):
            stations = [stations]
        out = out[out["station"].isin(stations)]

    if mode is not None and "mode" in out.columns:
        out = out[out["mode"] == mode]

    if sector_type is not None and "sector_typ" in out.columns:
        out = out[out["sector_typ"] == sector_type]

    out = add_time_features(out)
    out = out.dropna(subset=[value_col])

    months = list(range(1, 13))
    data = [out.loc[out["month"] == m, value_col].dropna().values for m in months]

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(data, tick_labels=[calendar.month_abbr[m] for m in months])
    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel or value_col)
    ax.set_title(title or f"Monthly boxplot of {value_col}")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig, ax


def plot_seasonal_boxplot(df, value_col, stations=None, mode=None, sector_type=None, title=None, ylabel=None, out_path=None, figsize=(10, 6)):
    out = df.copy()

    if stations is not None:
        if isinstance(stations, str):
            stations = [stations]
        out = out[out["station"].isin(stations)]

    if mode is not None and "mode" in out.columns:
        out = out[out["mode"] == mode]

    if sector_type is not None and "sector_typ" in out.columns:
        out = out[out["sector_typ"] == sector_type]

    out = add_time_features(out)
    out = out.dropna(subset=[value_col])

    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    data = [out.loc[out["season"] == s, value_col].dropna().values for s in seasons]

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(data, tick_labels=seasons)
    ax.set_xlabel("Season")
    ax.set_ylabel(ylabel or value_col)
    ax.set_title(title or f"Seasonal boxplot of {value_col}")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return fig, ax


# ============================================================
# 11. SCATTER: CENTER_PPB VS CV_W
# ============================================================
def plot_center_vs_cv_scatter(
    df, stations=None, mode=None, sector_type=None,
    aggregate_cv_over_sectors=True,
    out_path=None, figsize=(8, 6)
):
    pairs = prepare_center_cv_pairs(
        df=df,
        stations=stations,
        mode=mode,
        sector_type=sector_type,
        aggregate_cv_over_sectors=aggregate_cv_over_sectors
    )

    corr = safe_corr(pairs["center_ppb"], pairs["cv_w"], method="pearson")

    fig, ax = plt.subplots(figsize=figsize)
    if "station" in pairs.columns:
        for st, g in pairs.groupby("station"):
            ax.scatter(g["center_ppb"], g["cv_w"], alpha=0.6, label=st)
        ax.legend(fontsize=8)
    else:
        ax.scatter(pairs["center_ppb"], pairs["cv_w"], alpha=0.6)

    ax.set_xlabel("center_ppb")
    ax.set_ylabel("cv_w")
    ax.set_title(f"Scatter: center_ppb vs cv_w | Pearson r={corr:.3f}" if pd.notna(corr) else "Scatter: center_ppb vs cv_w")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return pairs, corr, fig, ax


def plot_center_vs_cv_scatter_by_season(
    df, stations=None, mode=None, sector_type=None,
    aggregate_cv_over_sectors=True,
    out_path=None, figsize=(14, 10)
):
    pairs = prepare_center_cv_pairs(
        df=df,
        stations=stations,
        mode=mode,
        sector_type=sector_type,
        aggregate_cv_over_sectors=aggregate_cv_over_sectors
    )

    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, season in zip(axes, seasons):
        sub = pairs[pairs["season"] == season]
        corr = safe_corr(sub["center_ppb"], sub["cv_w"])

        if not sub.empty:
            if "station" in sub.columns:
                for st, g in sub.groupby("station"):
                    ax.scatter(g["center_ppb"], g["cv_w"], alpha=0.6, label=st)
            else:
                ax.scatter(sub["center_ppb"], sub["cv_w"], alpha=0.6)

        ax.set_title(f"{season} | r={corr:.3f}" if pd.notna(corr) else f"{season} | r=nan")
        ax.set_xlabel("center_ppb")
        ax.set_ylabel("cv_w")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if labels:
        fig.legend(handles, labels, loc="upper right")

    fig.suptitle("center_ppb vs cv_w by season", y=1.02)
    fig.tight_layout()
    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return pairs, fig, axes


def plot_center_vs_cv_scatter_by_month(
    df, stations=None, mode=None, sector_type=None,
    aggregate_cv_over_sectors=True,
    out_path=None, figsize=(18, 12)
):
    pairs = prepare_center_cv_pairs(
        df=df,
        stations=stations,
        mode=mode,
        sector_type=sector_type,
        aggregate_cv_over_sectors=aggregate_cv_over_sectors
    )

    fig, axes = plt.subplots(3, 4, figsize=figsize, sharex=True, sharey=True)
    axes = axes.ravel()

    for i, m in enumerate(range(1, 13)):
        ax = axes[i]
        sub = pairs[pairs["month"] == m]
        corr = safe_corr(sub["center_ppb"], sub["cv_w"])

        if not sub.empty:
            if "station" in sub.columns:
                for st, g in sub.groupby("station"):
                    ax.scatter(g["center_ppb"], g["cv_w"], alpha=0.6, label=st)
            else:
                ax.scatter(sub["center_ppb"], sub["cv_w"], alpha=0.6)

        ax.set_title(f"{calendar.month_abbr[m]} | r={corr:.3f}" if pd.notna(corr) else f"{calendar.month_abbr[m]} | r=nan")
        ax.set_xlabel("center_ppb")
        ax.set_ylabel("cv_w")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if labels:
        fig.legend(handles, labels, loc="upper right")

    fig.suptitle("center_ppb vs cv_w by month", y=1.02)
    fig.tight_layout()
    if out_path:
        ensure_dir(out_path)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    return pairs, fig, axes


# ============================================================
# 12. ALTITUDE RELATIONS FOR MULTIPLE STATIONS
# ============================================================
def plot_altitude_relations(
    df, stations=None, mode=None, sector_type=None,
    center_col="center_ppb", cv_col="cv_w",
    out_prefix=None, figsize=(8, 6)
):
    tbl = altitude_relation_table(
        df=df,
        stations=stations,
        center_col=center_col,
        cv_col=cv_col,
        mode=mode,
        sector_type=sector_type
    )

    results = {}

    # center_mean vs altitude
    fig1, ax1 = plt.subplots(figsize=figsize)
    ax1.scatter(tbl["Altitude"], tbl["center_mean"], alpha=0.8)
    for _, r in tbl.iterrows():
        ax1.text(r["Altitude"], r["center_mean"], str(r["station"]), fontsize=8)
    r1 = safe_corr(tbl["Altitude"], tbl["center_mean"])
    ax1.set_xlabel("Altitude (m)")
    ax1.set_ylabel("Mean center_ppb")
    ax1.set_title(f"Mean center_ppb vs Altitude | r={r1:.3f}" if pd.notna(r1) else "Mean center_ppb vs Altitude")
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    if out_prefix:
        p = f"{out_prefix}_center_mean_vs_altitude.png"
        ensure_dir(p)
        fig1.savefig(p, dpi=200, bbox_inches="tight")
    plt.show()
    results["center_mean_vs_altitude_corr"] = r1

    # cv_w_mean vs altitude
    fig2, ax2 = plt.subplots(figsize=figsize)
    ax2.scatter(tbl["Altitude"], tbl["cv_w_mean"], alpha=0.8)
    for _, r in tbl.iterrows():
        ax2.text(r["Altitude"], r["cv_w_mean"], str(r["station"]), fontsize=8)
    r2 = safe_corr(tbl["Altitude"], tbl["cv_w_mean"])
    ax2.set_xlabel("Altitude (m)")
    ax2.set_ylabel("Mean cv_w")
    ax2.set_title(f"Mean cv_w vs Altitude | r={r2:.3f}" if pd.notna(r2) else "Mean cv_w vs Altitude")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    if out_prefix:
        p = f"{out_prefix}_cvw_mean_vs_altitude.png"
        ensure_dir(p)
        fig2.savefig(p, dpi=200, bbox_inches="tight")
    plt.show()
    results["cvw_mean_vs_altitude_corr"] = r2

    # variability of center_ppb vs altitude
    fig3, ax3 = plt.subplots(figsize=figsize)
    ax3.scatter(tbl["Altitude"], tbl["center_std"], alpha=0.8)
    for _, r in tbl.iterrows():
        ax3.text(r["Altitude"], r["center_std"], str(r["station"]), fontsize=8)
    r3 = safe_corr(tbl["Altitude"], tbl["center_std"])
    ax3.set_xlabel("Altitude (m)")
    ax3.set_ylabel("Std of center_ppb")
    ax3.set_title(f"Variability of center_ppb vs Altitude | r={r3:.3f}" if pd.notna(r3) else "Variability of center_ppb vs Altitude")
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    if out_prefix:
        p = f"{out_prefix}_center_std_vs_altitude.png"
        ensure_dir(p)
        fig3.savefig(p, dpi=200, bbox_inches="tight")
    plt.show()
    results["center_std_vs_altitude_corr"] = r3

    return tbl, results
#%%
stations_path = "/home/agkiokas/CAMS/CHINESE_STATIONS_INFO_2015_2023.txt"
stations_df = load_stations_file(stations_path)

species = "O3"
mode = "A"
input_dir = "/home/agkiokas/CAMS/plots"
out_dir = "/home/agkiokas/CAMS/plots"

# one station
station = "1461A"
df_one = load_station_timeseries_csv(f"{input_dir}/{station}_{species}_{mode}_30min.csv", station_name=station)
df_one = attach_station_metadata(df_one, stations_df)
#%%
# multiple stations
stations = ["1461A", "1006A", "2629A","2686A"]
df_all = load_many_station_csvs(input_dir=input_dir, species=species, mode=mode, station_names=stations)
df_all = attach_station_metadata(df_all, stations_df)
#--------------------------------------------------------
plot_one_station_seasonal_center(
    df=df_one,
    station=station,
    stations_df=stations_df,
    center_col="center_ppb",
    mode=mode,
    species=species,
    units="ppb",
    out_path=f"{out_dir}/{station}_{species}_seasonal_center.png"
)
#------------------------------------------------------
plot_multi_station_seasonal_center(
    df=df_all,
    stations=stations,
    stations_df=stations_df,
    center_col="center_ppb",
    mode=mode,
    species=species,
    units="ppb",
    out_path=f"{out_dir}/multi_station_{species}_seasonal_center.png"
)

plot_one_station_seasonal_cv_by_sector(
    df=df_one,
    station=station,
    stations_df=stations_df,
    cv_col="cv_w",
    sector_col="sector",
    sector_type="CUM",
    mode=mode,
    species=species,
    out_path=f"{out_dir}/{station}_{species}_seasonal_cvw_by_sector.png"
)
#--------------------------------------------------------------------------
monthly_one = monthly_stats_table(df_one, value_cols=["center_ppb", "cv_w"])
seasonal_one = seasonal_stats_table(df_one, value_cols=["center_ppb", "cv_w"])

print("\nMONTHLY STATS - ONE STATION")
print(monthly_one.to_string(index=False))

print("\nSEASONAL STATS - ONE STATION")
print(seasonal_one.to_string(index=False))

monthly_one.to_csv(f"{out_dir}/{station}_{species}_monthly_stats.csv", index=False)
seasonal_one.to_csv(f"{out_dir}/{station}_{species}_seasonal_stats.csv", index=False)
#------------------------------------------------------------------------------
monthly_all = monthly_stats_table(df_all, value_cols=["center_ppb", "cv_w"])
seasonal_all = seasonal_stats_table(df_all, value_cols=["center_ppb", "cv_w"])

monthly_all.to_csv(f"{out_dir}/multi_station_{species}_monthly_stats.csv", index=False)
seasonal_all.to_csv(f"{out_dir}/multi_station_{species}_seasonal_stats.csv", index=False)

plot_monthly_boxplot(
    df=df_one,
    value_col="center_ppb",
    stations=station,
    mode=mode,
    title=f"{station} monthly boxplot of center_ppb",
    ylabel="center_ppb",
    out_path=f"{out_dir}/{station}_{species}_monthly_box_center.png"
)

plot_monthly_boxplot(
    df=df_one,
    value_col="cv_w",
    stations=station,
    mode=mode,
    sector_type="CUM",
    title=f"{station} monthly boxplot of cv_w",
    ylabel="cv_w",
    out_path=f"{out_dir}/{station}_{species}_monthly_box_cvw.png"
)

plot_seasonal_boxplot(
    df=df_one,
    value_col="center_ppb",
    stations=station,
    mode=mode,
    title=f"{station} seasonal boxplot of center_ppb",
    ylabel="center_ppb",
    out_path=f"{out_dir}/{station}_{species}_seasonal_box_center.png"
)

plot_seasonal_boxplot(
    df=df_one,
    value_col="cv_w",
    stations=station,
    mode=mode,
    sector_type="CUM",
    title=f"{station} seasonal boxplot of cv_w",
    ylabel="cv_w",
    out_path=f"{out_dir}/{station}_{species}_seasonal_box_cvw.png"
)
#------------------------------------
pairs_one, corr_one, _, _ = plot_center_vs_cv_scatter(
    df=df_one,
    stations=station,
    mode=mode,
    sector_type="CUM",
    aggregate_cv_over_sectors=True,
    out_path=f"{out_dir}/{station}_{species}_scatter_center_vs_cvw.png"
)

print("Overall correlation center_ppb vs cv_w:", corr_one)
#-------------------------------------------------------------
pairs_season, _, _ = plot_center_vs_cv_scatter_by_season(
    df=df_one,
    stations=station,
    mode=mode,
    sector_type="CUM",
    aggregate_cv_over_sectors=True,
    out_path=f"{out_dir}/{station}_{species}_scatter_center_vs_cvw_by_season.png"
)

corr_season_tbl = correlation_by_season(pairs_season)
print("\nCorrelation by season")
print(corr_season_tbl.to_string(index=False))
corr_season_tbl.to_csv(f"{out_dir}/{station}_{species}_corr_by_season.csv", index=False)
#----------------------------------------------------------------------
pairs_month, _, _ = plot_center_vs_cv_scatter_by_month(
    df=df_one,
    stations=station,
    mode=mode,
    sector_type="CUM",
    aggregate_cv_over_sectors=True,
    out_path=f"{out_dir}/{station}_{species}_scatter_center_vs_cvw_by_month.png"
)

corr_month_tbl = correlation_by_month(pairs_month)
print("\nCorrelation by month")
print(corr_month_tbl.to_string(index=False))
corr_month_tbl.to_csv(f"{out_dir}/{station}_{species}_corr_by_month.csv", index=False)
#-------------------------------------------------------------------
alt_tbl, alt_corrs = plot_altitude_relations(
    df=df_all,
    stations=stations,
    mode=mode,
    sector_type="CUM",
    center_col="center_ppb",
    cv_col="cv_w",
    out_prefix=f"{out_dir}/multi_station_{species}_altitude"
)

print("\nPer-station altitude relation table")
print(alt_tbl.to_string(index=False))

print("\nAltitude correlations")
print(alt_corrs)

alt_tbl.to_csv(f"{out_dir}/multi_station_{species}_altitude_relation_table.csv", index=False)
# ============================================================
# 13. WRAPPERS: RUN EVERYTHING AUTOMATICALLY
#    PUT THIS AT THE END OF YOUR FILE
# ============================================================
def run_full_station_analysis(
    df,
    station,
    stations_df=None,
    species="O3",
    units="ppb",
    mode=None,
    sector_type="CUM",
    out_dir=".",
    center_col="center_ppb",
    cv_col="cv_w",
    sector_col="sector",
    aggregate_cv_over_sectors=True,
):
    """
    Runs the full analysis for ONE station and saves all outputs.
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. Seasonal center timeseries
    plot_one_station_seasonal_center(
        df=df,
        station=station,
        stations_df=stations_df,
        center_col=center_col,
        mode=mode,
        species=species,
        units=units,
        out_path=os.path.join(out_dir, f"{station}_{species}_seasonal_center.png")
    )

    # 2. Seasonal cv_w by sector
    plot_one_station_seasonal_cv_by_sector(
        df=df,
        station=station,
        stations_df=stations_df,
        cv_col=cv_col,
        sector_col=sector_col,
        sector_type=sector_type,
        mode=mode,
        species=species,
        out_path=os.path.join(out_dir, f"{station}_{species}_seasonal_{cv_col}_by_sector.png")
    )

    # 3. Monthly boxplots
    plot_monthly_boxplot(
        df=df,
        value_col=center_col,
        stations=station,
        mode=mode,
        title=f"{station} monthly boxplot of {center_col}",
        ylabel=center_col,
        out_path=os.path.join(out_dir, f"{station}_{species}_monthly_box_{center_col}.png")
    )

    plot_monthly_boxplot(
        df=df,
        value_col=cv_col,
        stations=station,
        mode=mode,
        sector_type=sector_type,
        title=f"{station} monthly boxplot of {cv_col}",
        ylabel=cv_col,
        out_path=os.path.join(out_dir, f"{station}_{species}_monthly_box_{cv_col}.png")
    )

    # 4. Seasonal boxplots
    plot_seasonal_boxplot(
        df=df,
        value_col=center_col,
        stations=station,
        mode=mode,
        title=f"{station} seasonal boxplot of {center_col}",
        ylabel=center_col,
        out_path=os.path.join(out_dir, f"{station}_{species}_seasonal_box_{center_col}.png")
    )

    plot_seasonal_boxplot(
        df=df,
        value_col=cv_col,
        stations=station,
        mode=mode,
        sector_type=sector_type,
        title=f"{station} seasonal boxplot of {cv_col}",
        ylabel=cv_col,
        out_path=os.path.join(out_dir, f"{station}_{species}_seasonal_box_{cv_col}.png")
    )

    # 5. Scatter overall
    pairs, corr_all, _, _ = plot_center_vs_cv_scatter(
        df=df,
        stations=station,
        mode=mode,
        sector_type=sector_type,
        aggregate_cv_over_sectors=aggregate_cv_over_sectors,
        out_path=os.path.join(out_dir, f"{station}_{species}_scatter_{center_col}_vs_{cv_col}.png")
    )

    # 6. Scatter by season
    pairs_season, _, _ = plot_center_vs_cv_scatter_by_season(
        df=df,
        stations=station,
        mode=mode,
        sector_type=sector_type,
        aggregate_cv_over_sectors=aggregate_cv_over_sectors,
        out_path=os.path.join(out_dir, f"{station}_{species}_scatter_{center_col}_vs_{cv_col}_by_season.png")
    )

    # 7. Scatter by month
    pairs_month, _, _ = plot_center_vs_cv_scatter_by_month(
        df=df,
        stations=station,
        mode=mode,
        sector_type=sector_type,
        aggregate_cv_over_sectors=aggregate_cv_over_sectors,
        out_path=os.path.join(out_dir, f"{station}_{species}_scatter_{center_col}_vs_{cv_col}_by_month.png")
    )

    # 8. Stats tables
    monthly_stats = monthly_stats_table(df[df["station"] == station], value_cols=[center_col, cv_col])
    seasonal_stats = seasonal_stats_table(df[df["station"] == station], value_cols=[center_col, cv_col])

    monthly_stats.to_csv(
        os.path.join(out_dir, f"{station}_{species}_monthly_stats.csv"),
        index=False
    )
    seasonal_stats.to_csv(
        os.path.join(out_dir, f"{station}_{species}_seasonal_stats.csv"),
        index=False
    )

    # 9. Correlation tables
    corr_month_tbl = correlation_by_month(pairs_month, x_col="center_ppb", y_col="cv_w")
    corr_season_tbl = correlation_by_season(pairs_season, x_col="center_ppb", y_col="cv_w")

    corr_month_tbl.to_csv(
        os.path.join(out_dir, f"{station}_{species}_corr_by_month.csv"),
        index=False
    )
    corr_season_tbl.to_csv(
        os.path.join(out_dir, f"{station}_{species}_corr_by_season.csv"),
        index=False
    )

    print(f"\nSaved full one-station analysis for {station} in: {out_dir}")
    print(f"Overall Pearson correlation ({center_col} vs {cv_col}): {corr_all}")

    return {
        "monthly_stats": monthly_stats,
        "seasonal_stats": seasonal_stats,
        "corr_month": corr_month_tbl,
        "corr_season": corr_season_tbl,
        "overall_corr": corr_all,
        "pairs": pairs,
    }


def run_full_multi_station_analysis(
    df,
    stations,
    stations_df=None,
    species="O3",
    units="ppb",
    mode=None,
    sector_type="CUM",
    out_dir=".",
    center_col="center_ppb",
    cv_col="cv_w",
    aggregate_cv_over_sectors=True,
):
    """
    Runs the full analysis for MULTIPLE stations and saves all outputs.
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. Seasonal multi-station center timeseries
    plot_multi_station_seasonal_center(
        df=df,
        stations=stations,
        stations_df=stations_df,
        center_col=center_col,
        mode=mode,
        species=species,
        units=units,
        out_path=os.path.join(out_dir, f"multi_station_{species}_seasonal_center.png")
    )

    # 2. Monthly boxplots
    plot_monthly_boxplot(
        df=df,
        value_col=center_col,
        stations=stations,
        mode=mode,
        title=f"Multi-station monthly boxplot of {center_col}",
        ylabel=center_col,
        out_path=os.path.join(out_dir, f"multi_station_{species}_monthly_box_{center_col}.png")
    )

    plot_monthly_boxplot(
        df=df,
        value_col=cv_col,
        stations=stations,
        mode=mode,
        sector_type=sector_type,
        title=f"Multi-station monthly boxplot of {cv_col}",
        ylabel=cv_col,
        out_path=os.path.join(out_dir, f"multi_station_{species}_monthly_box_{cv_col}.png")
    )

    # 3. Seasonal boxplots
    plot_seasonal_boxplot(
        df=df,
        value_col=center_col,
        stations=stations,
        mode=mode,
        title=f"Multi-station seasonal boxplot of {center_col}",
        ylabel=center_col,
        out_path=os.path.join(out_dir, f"multi_station_{species}_seasonal_box_{center_col}.png")
    )

    plot_seasonal_boxplot(
        df=df,
        value_col=cv_col,
        stations=stations,
        mode=mode,
        sector_type=sector_type,
        title=f"Multi-station seasonal boxplot of {cv_col}",
        ylabel=cv_col,
        out_path=os.path.join(out_dir, f"multi_station_{species}_seasonal_box_{cv_col}.png")
    )

    # 4. Scatter overall
    pairs, corr_all, _, _ = plot_center_vs_cv_scatter(
        df=df,
        stations=stations,
        mode=mode,
        sector_type=sector_type,
        aggregate_cv_over_sectors=aggregate_cv_over_sectors,
        out_path=os.path.join(out_dir, f"multi_station_{species}_scatter_{center_col}_vs_{cv_col}.png")
    )

    # 5. Scatter by season
    pairs_season, _, _ = plot_center_vs_cv_scatter_by_season(
        df=df,
        stations=stations,
        mode=mode,
        sector_type=sector_type,
        aggregate_cv_over_sectors=aggregate_cv_over_sectors,
        out_path=os.path.join(out_dir, f"multi_station_{species}_scatter_{center_col}_vs_{cv_col}_by_season.png")
    )

    # 6. Scatter by month
    pairs_month, _, _ = plot_center_vs_cv_scatter_by_month(
        df=df,
        stations=stations,
        mode=mode,
        sector_type=sector_type,
        aggregate_cv_over_sectors=aggregate_cv_over_sectors,
        out_path=os.path.join(out_dir, f"multi_station_{species}_scatter_{center_col}_vs_{cv_col}_by_month.png")
    )

    # 7. Stats tables
    monthly_stats = monthly_stats_table(df[df["station"].isin(stations)], value_cols=[center_col, cv_col])
    seasonal_stats = seasonal_stats_table(df[df["station"].isin(stations)], value_cols=[center_col, cv_col])

    monthly_stats.to_csv(
        os.path.join(out_dir, f"multi_station_{species}_monthly_stats.csv"),
        index=False
    )
    seasonal_stats.to_csv(
        os.path.join(out_dir, f"multi_station_{species}_seasonal_stats.csv"),
        index=False
    )

    # 8. Correlation tables
    corr_month_tbl = correlation_by_month(pairs_month, x_col="center_ppb", y_col="cv_w")
    corr_season_tbl = correlation_by_season(pairs_season, x_col="center_ppb", y_col="cv_w")

    corr_month_tbl.to_csv(
        os.path.join(out_dir, f"multi_station_{species}_corr_by_month.csv"),
        index=False
    )
    corr_season_tbl.to_csv(
        os.path.join(out_dir, f"multi_station_{species}_corr_by_season.csv"),
        index=False
    )

    # 9. Altitude relations
    alt_tbl, alt_corrs = plot_altitude_relations(
        df=df,
        stations=stations,
        mode=mode,
        sector_type=sector_type,
        center_col=center_col,
        cv_col=cv_col,
        out_prefix=os.path.join(out_dir, f"multi_station_{species}_altitude")
    )

    alt_tbl.to_csv(
        os.path.join(out_dir, f"multi_station_{species}_altitude_relation_table.csv"),
        index=False
    )

    print(f"\nSaved full multi-station analysis in: {out_dir}")
    print(f"Overall Pearson correlation ({center_col} vs {cv_col}): {corr_all}")
    print("Altitude correlations:", alt_corrs)

    return {
        "monthly_stats": monthly_stats,
        "seasonal_stats": seasonal_stats,
        "corr_month": corr_month_tbl,
        "corr_season": corr_season_tbl,
        "overall_corr": corr_all,
        "altitude_table": alt_tbl,
        "altitude_corrs": alt_corrs,
        "pairs": pairs,
    }
#%%
species = "O3"
mode = "A"
station = "1461A"
out_dir = "/home/agkiokas/CAMS/plots/full_analysis_one_station"

# df_one should already be loaded
# df_one = load_station_timeseries_csv(...)
# df_one = attach_station_metadata(df_one, stations_df)

res_one = run_full_station_analysis(
    df=df_one,
    station=station,
    stations_df=stations_df,
    species=species,
    units="ppb",
    mode=mode,
    sector_type="CUM",
    out_dir=out_dir,
    center_col="center_ppb",
    cv_col="cv_w",
    sector_col="sector",
    aggregate_cv_over_sectors=True
)
#%%
species = "O3"
mode = "A"
stations = ["1461A", "1006A", "2686A"]
out_dir = "/home/agkiokas/CAMS/plots/full_analysis_multi_station"

# df_all should already be loaded
# df_all = load_many_station_csvs(...)
# df_all = attach_station_metadata(df_all, stations_df)

res_multi = run_full_multi_station_analysis(
    df=df_all,
    stations=stations,
    stations_df=stations_df,
    species=species,
    units="ppb",
    mode=mode,
    sector_type="CUM",
    out_dir=out_dir,
    center_col="center_ppb",
    cv_col="cv_w",
    aggregate_cv_over_sectors=True
)

#%%

ONE_STATION=True
def plot_stats():
    if ONE_STATION:
        res_one
    else:
        res_multi  
if __name__ == "__plot_stats":
    plot_stats()          