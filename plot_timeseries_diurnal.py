#%%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# ============================================================
# USER SETTINGS
# ============================================================

PARQUET_DIR = Path("/mnt/store01/agkiokas/CAMS/stations_parquet/")
OUT_DIR = Path("/mnt/store01/agkiokas/CAMS/station_timeseries_plots/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# One station:
STATIONS = ["1001A"]

# Multiple stations:
#STATIONS = ["1001A", "1146A"]

# Or all available parquet files:
#STATIONS = "all"

# Columns to plot
PLOT_COLUMNS = ["center_ppb"]   
# Example:
# PLOT_COLUMNS = ["center_ppb", "cv_w", "mean_w"]

# Choose sector
# Because each timestep appears 10 times, one row per sector C1-C10
SECTOR = "C1"

# Optional filtering
# Use None for full period
#START_DATE = "2006-05-01 00:00:00" #"YYYY-MM-DD HH:MM:SS"
#END_DATE = "2006-05-08 00:30:00"

START_DATE = None
END_DATE = None

# Plot options
MAKE_TIMESERIES = True
MAKE_DIURNAL_CYCLE = True

LOCAL_TIME_OFFSET_HOURS = 8
SECTOR_DEPENDENT_COLUMNS = ["cv_w","mean_w"]

error_mode="shade" # 'shade',or 'boxplot'
error_type='std' #'std','iqr'
FIGSIZE = (13, 5)
DPI = 150


# ============================================================
# HELPERS
# ============================================================

def get_station_files(parquet_dir, stations):

    # Exclude summary parquet files
    all_parquet = [
        fp for fp in parquet_dir.glob("*.parquet")
        if "summary" not in fp.name.lower()
    ]

    if stations == "all":
        return sorted(all_parquet)

    files = []

    for st in stations:
        matches = [
            fp for fp in all_parquet
            if st in fp.name
        ]

        if not matches:
            print(f"Warning: no parquet file found for station {st}")

        files.extend(matches)

    return sorted(files)


def build_datetime(df):
    date_str = df["timestamp"].astype(str).str[:8]
    time_str = df["time"].astype(str).str.zfill(4)

    df["datetime_utc"] = pd.to_datetime(
        date_str + time_str,
        format="%Y%m%d%H%M",
        errors="coerce"
    )

    df["datetime_local"] = df["datetime_utc"] + pd.Timedelta(hours=LOCAL_TIME_OFFSET_HOURS)

    df = df.dropna(subset=["datetime_utc"])

    return df

def filter_period(df, start_date=None, end_date=None):
    if start_date is not None:
        df = df[df["datetime_utc"] >= pd.to_datetime(start_date)]

    if end_date is not None:
        df = df[df["datetime_utc"] <= pd.to_datetime(end_date)]

    return df
def period_label(start_date, end_date):
    if start_date is None and end_date is None:
        return "full_period"

    s = "start" if start_date is None else pd.to_datetime(start_date).strftime("%Y%m%d_%H%M")
    e = "end" if end_date is None else pd.to_datetime(end_date).strftime("%Y%m%d_%H%M")

    return f"{s}_to_{e}"


def title_sector_part(column, sector):
    if column in SECTOR_DEPENDENT_COLUMNS:
        return f" - {sector}"
    return ""


def filename_sector_part(column, sector):
    if column in SECTOR_DEPENDENT_COLUMNS:
        return f"_{sector}"
    return ""


def prepare_station_df(fp, sector, start_date=None, end_date=None, plot_columns=None):
    df = pd.read_parquet(fp)

    df = build_datetime(df)
    df = filter_period(df, start_date, end_date)

    if df.empty:
        return df

    if plot_columns is None:
        plot_columns = []

    needs_sector = any(col in SECTOR_DEPENDENT_COLUMNS for col in plot_columns)

    if needs_sector:
        if "sector" in df.columns:
            df = df[df["sector"] == sector].copy()
        else:
            print(f"Warning: no sector column in {fp.name}")
    else:
        df = df.drop_duplicates(subset=["datetime_utc"]).copy()

    df = df.sort_values("datetime_utc")

    return df


def plot_timeseries(df, station, column, sector, out_dir, time_label):
    if column not in df.columns:
        print(f"Skipping {station} {column}: column not found")
        return

    plot_df = df[["datetime_utc", "datetime_local", column]].dropna()

    if plot_df.empty:
        print(f"Skipping {station} {column}: no data")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(
        plot_df["datetime_utc"],
        plot_df[column],
        marker="o",
        linewidth=1,
        markersize=3
    )

    ax.set_title(
        f"{station} - {column} timeseries"
        f"{title_sector_part(column, sector)} - {time_label}"
    )
    ax.set_xlabel("UTC time")
    ax.set_ylabel(column)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())

    utc_ticks = ax.get_xticks()
    utc_tick_dates = pd.to_datetime(
        [mdates.num2date(x).replace(tzinfo=None) for x in utc_ticks]
    )
    local_tick_labels = [
        (x + pd.Timedelta(hours=LOCAL_TIME_OFFSET_HOURS)).strftime("%Y-%m-%d\n%H:%M")
        for x in utc_tick_dates
    ]

    ax2.set_xticks(utc_ticks)
    ax2.set_xticklabels(local_tick_labels, rotation=30, ha="left")
    ax2.set_xlabel(f"Local time UTC+{LOCAL_TIME_OFFSET_HOURS}")

    fig.autofmt_xdate()
    fig.tight_layout()

    out_file = (
        out_dir
        / f"{station}{filename_sector_part(column, sector)}_{column}_timeseries_{time_label}.png"
    )
    #plt.savefig(out_file, dpi=DPI)
    plt.show()

    print(f"Saved: {out_file}")
    


def plot_diurnal_cycle(
    df,
    station,
    column,
    sector,
    out_dir,
    time_label,
    error_mode="shade",      # "shade", "boxplot", or "none"
    error_type="std"         # "std", "sem", "iqr"
):
    if column not in df.columns:
        print(f"Skipping {station} {column}: column not found")
        return

    plot_df = df[["datetime_local", column]].dropna().copy()

    if plot_df.empty:
        print(f"Skipping {station} {column}: no data")
        return

    plot_df["hour_decimal_local"] = (
        plot_df["datetime_local"].dt.hour
        + plot_df["datetime_local"].dt.minute / 60
    )

    grouped = plot_df.groupby("hour_decimal_local")[column]

    diurnal = grouped.agg(
        mean="mean",
        std="std",
        count="count",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75)
    ).reset_index().sort_values("hour_decimal_local")

    diurnal["sem"] = diurnal["std"] / np.sqrt(diurnal["count"])

    plt.figure(figsize=FIGSIZE)

    if error_mode == "boxplot":
        hours = sorted(plot_df["hour_decimal_local"].unique())

        data_by_hour = [
            plot_df.loc[plot_df["hour_decimal_local"] == h, column].values
            for h in hours
        ]

        plt.boxplot(
            data_by_hour,
            positions=hours,
            widths=0.25,
            showfliers=False
        )

        plt.plot(
            diurnal["hour_decimal_local"],
            diurnal["mean"],
            marker="o",
            linewidth=1.5,
            markersize=4,
            label="Mean"
        )

    else:
        plt.plot(
            diurnal["hour_decimal_local"],
            diurnal["mean"],
            marker="o",
            linewidth=1.5,
            markersize=4,
            label="Mean"
        )

        if error_mode == "shade":
            if error_type == "std":
                lower = diurnal["mean"] - diurnal["std"]
                upper = diurnal["mean"] + diurnal["std"]
                error_label = "Mean ± std"

            elif error_type == "sem":
                lower = diurnal["mean"] - diurnal["sem"]
                upper = diurnal["mean"] + diurnal["sem"]
                error_label = "Mean ± SEM"

            elif error_type == "iqr":
                lower = diurnal["q25"]
                upper = diurnal["q75"]
                error_label = "IQR"

            else:
                raise ValueError("error_type must be 'std', 'sem', or 'iqr'")

            plt.fill_between(
                diurnal["hour_decimal_local"],
                lower,
                upper,
                alpha=0.25,
                label=error_label
            )

    plt.title(
        f"{station} - {column} diurnal cycle"
        f"{title_sector_part(column, sector)} - {time_label}"
    )

    plt.xlabel(f"Hour of day, local time UTC+{LOCAL_TIME_OFFSET_HOURS}")
    plt.ylabel(f"Mean {column}")
    plt.xticks(range(0, 25, 2))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_file = (
        out_dir
        / f"{station}{filename_sector_part(column, sector)}_{column}_diurnal_cycle_{error_mode}_{error_type}_{time_label}.png"
    )

    plt.savefig(out_file, dpi=DPI)
    plt.show()

    print(f"Saved: {out_file}")


# ============================================================
# MAIN
# ============================================================

def main():
    files = get_station_files(PARQUET_DIR, STATIONS)

    if not files:
        raise FileNotFoundError("No parquet files found.")

    print(f"Found {len(files)} parquet files")

    for fp in files:
        station = fp.stem.split("_")[0]

        print(f"\nProcessing station: {station}")
        print(f"File: {fp.name}")

        time_label = period_label(START_DATE, END_DATE)

        df = prepare_station_df(
           fp,sector=SECTOR, start_date=START_DATE,
           end_date=END_DATE, plot_columns=PLOT_COLUMNS
        )

        if df.empty:
            print(f"No data for {station} after filtering")
            continue

        for column in PLOT_COLUMNS:
            if MAKE_TIMESERIES:
                plot_timeseries(df, station, column, SECTOR, OUT_DIR, time_label)

            if MAKE_DIURNAL_CYCLE:
                plot_diurnal_cycle(df, station, column, SECTOR, OUT_DIR, time_label)


if __name__ == "__main__":
    main()
# %%
