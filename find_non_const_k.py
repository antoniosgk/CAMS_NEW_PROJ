#%%
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# USER INPUT
# ============================================================

LEVEL_TS_FILE = Path("/home/agkiokas/CAMS/lookups/station_level_timeseries_all.parquet")
K_DECISION_FILE = Path("/home/agkiokas/CAMS/lookups/station_k_decision_table.csv")

OUT_DIR = Path("/home/agkiokas/CAMS/output_k_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def month_to_season(m):
    if m in [12, 1, 2]:
        return "Winter"
    elif m in [3, 4, 5]:
        return "Spring"
    elif m in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"


def add_time_features(df):
    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    elif "datetime" in df.columns:
        df["timestamp"] = pd.to_datetime(df["datetime"])
    elif "time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["time"])
    else:
        raise ValueError("No time column found")

    df["month"] = df["timestamp"].dt.month
    df["season"] = df["month"].map(month_to_season)

    # simple day/night
    df["hour"] = df["timestamp"].dt.hour
    df["day_night"] = np.where((df["hour"] >= 6) & (df["hour"] < 18), "day", "night")

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    df = pd.read_parquet(LEVEL_TS_FILE)
    df = add_time_features(df)

    station_col = "Station_Name" if "Station_Name" in df.columns else "station"

    if "k_star_center" in df.columns:
        k_col = "k_star_center"
    else:
        k_col = "level_idx"

    # --------------------------------------------------------
    # find NON-constant stations dynamically
    # --------------------------------------------------------
    k_decision = pd.read_csv(K_DECISION_FILE)

    var_stations = k_decision.loc[
        k_decision["k_min"] != k_decision["k_max"],
        "Station_Name"
    ].tolist()

    print(f"Found {len(var_stations)} non-constant-k stations")

    df = df[df[station_col].isin(var_stations)].copy()

    # --------------------------------------------------------
    # 1. TOTAL DISTRIBUTION
    # --------------------------------------------------------
    total = (
        df.groupby([station_col, k_col])
        .size()
        .rename("count")
        .reset_index()
    )

    total["pct"] = total.groupby(station_col)["count"].transform(lambda x: 100 * x / x.sum())

    total = total.sort_values([station_col, "pct"], ascending=[True, False])

    total.to_csv(OUT_DIR / "k_distribution_total.csv", index=False)

    # --------------------------------------------------------
    # 2. SEASONAL DISTRIBUTION
    # --------------------------------------------------------
    season = (
        df.groupby([station_col, "season", k_col])
        .size()
        .rename("count")
        .reset_index()
    )

    season["pct"] = season.groupby([station_col, "season"])["count"].transform(lambda x: 100 * x / x.sum())

    season = season.sort_values([station_col, "season", "pct"], ascending=[True, True, False])

    season.to_csv(OUT_DIR / "k_distribution_by_season.csv", index=False)

    # --------------------------------------------------------
    # 3. DAY/NIGHT DISTRIBUTION
    # --------------------------------------------------------
    dn = (
        df.groupby([station_col, "day_night", k_col])
        .size()
        .rename("count")
        .reset_index()
    )

    dn["pct"] = dn.groupby([station_col, "day_night"])["count"].transform(lambda x: 100 * x / x.sum())

    dn = dn.sort_values([station_col, "day_night", "pct"], ascending=[True, True, False])

    dn.to_csv(OUT_DIR / "k_distribution_day_night.csv", index=False)

    print("\nSaved:")
    print(" - k_distribution_total.csv")
    print(" - k_distribution_by_season.csv")
    print(" - k_distribution_day_night.csv")


if __name__ == "__main__":
    main()
# %%
