#%%
import pandas as pd

LEVEL_LOOKUP_PATH = "/home/agkiokas/CAMS/lookups/station_level_timeseries_all.parquet"

df = pd.read_parquet(LEVEL_LOOKUP_PATH)
df["time"] = pd.to_datetime(df["time"], errors="coerce")

# summary by station
station_k_summary = (
    df.groupby(["idx", "Station_Name"], as_index=False)
      .agg(
          n_rows=("level_idx", "size"),
          n_times=("time", "nunique"),
          n_unique_k=("level_idx", "nunique"),
          k_min=("level_idx", "min"),
          k_max=("level_idx", "max"),
          level_idx_first=("level_idx", "first"),
      )
)

station_k_summary["k_is_constant"] = station_k_summary["n_unique_k"] == 1
station_k_summary["level_idx_constant"] = station_k_summary["level_idx_first"].where(
    station_k_summary["k_is_constant"], pd.NA
)

station_k_summary = station_k_summary.sort_values(["k_is_constant", "idx"], ascending=[False, True]).reset_index(drop=True)

print(station_k_summary)

# save full decision table
DECISION_PATH = "/home/agkiokas/CAMS/lookups/station_k_decision_table.csv"
station_k_summary.to_csv(DECISION_PATH, index=False)
print("Saved:", DECISION_PATH)

# save compact constant-k lookup
constant_k_lookup = (
    station_k_summary[station_k_summary["k_is_constant"]]
    [["idx", "Station_Name", "level_idx_constant"]]
    .rename(columns={"level_idx_constant": "level_idx"})
    .reset_index(drop=True)
)

FAST_K_PATH = "/home/agkiokas/CAMS/lookups/station_constant_k_lookup.csv"
constant_k_lookup.to_csv(FAST_K_PATH, index=False)
print("Saved:", FAST_K_PATH)

# print changing stations only
changing_k = station_k_summary[~station_k_summary["k_is_constant"]].copy()
print("\nStations whose k changes with time:")
print(changing_k[["idx", "Station_Name", "n_times", "n_unique_k", "k_min", "k_max"]])
# %%
