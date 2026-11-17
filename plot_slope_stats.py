#%%
"""
plot_slope_stats.py
====================
Reads slope_stats_box04.csv (produced by aggregate_slope_stats.py) and
generates plots of the mean and median analytic derivative at the 0.4° box
area for all 7 fits × 2 variables, across all groupings.

Plot types produced:
  1. Network-wide summary bar charts: mean and median across all stations
     — one figure per variable, bars for each model, grouped by period (all / seasons / day-night)
  2. Boxplots of mean, median, std slope distributions (10th-90th pctl whiskers, no outliers)
  3. Cartopy maps of mean, median, std slope per station (spatial patterns)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ============================================================
# USER SETTINGS
# ============================================================

STATS_CSV = Path("/mnt/store01/agkiokas/CAMS/fit_outputs/summary/slope_stats_box04.csv")
OUT_DIR   = Path("/mnt/store01/agkiokas/CAMS/fit_outputs/summary/plots/slope_stats")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPECIES = "O3"

VARIABLES = ["ratio", "cv_w"]
MODELS    = [ "cubic", "power", "saturating"]

MODEL_LABELS = {
    "cubic": "Cubic",
    "power": "Power-law", "saturating": "Saturating",
}

MODEL_COLORS = {
    "linear": "#1f77b4", "quadratic": "#ff7f0e", "cubic": "#2ca02c",
    "logarithmic": "#9467bd", "exponential": "#d62728",
    "power": "#8c564b", "saturating": "#17becf",
}

VAR_LABELS = {"ratio": "Concentration Ratio", "cv_w": "CV (%)"}

GROUPS        = ["all", "DJF", "MAM", "JJA", "SON", "day", "night"]
GROUP_LABELS  = {"all": "All", "DJF": "DJF", "MAM": "MAM", "JJA": "JJA",
                 "SON": "SON", "day": "Day", "night": "Night"}

# Figure settings
DPI         = 300
TITLE_SIZE  = 15
LABEL_SIZE  = 12
TICK_SIZE   = 11
LEGEND_SIZE = 11
BG_COLOR    = "gainsboro"

# Use Cartopy for spatial maps? Set to False to skip map plots.
PLOT_MAPS = True

plt.rcParams.update({
    "font.weight":        "bold",
    "axes.labelweight":   "bold",
    "axes.titlesize":     TITLE_SIZE,
    "axes.labelsize":     LABEL_SIZE,
    "xtick.labelsize":    TICK_SIZE,
    "ytick.labelsize":    TICK_SIZE,
    "legend.fontsize":    LEGEND_SIZE,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.facecolor":     BG_COLOR,
})


# ============================================================
# HELPERS
# ============================================================

def savefig(name):
    path = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=DPI, facecolor="white")
    plt.close()
    print(f"  Saved: {path.name}")


def bold_ticks(ax):
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


# ============================================================
# PLOT 1: NETWORK-WIDE SUMMARY BAR CHARTS
#   For each variable, show the network-average of each model's
#   mean and median slope, grouped by period.
# ============================================================

def plot_network_summary(df):
    """
    For each variable: one figure with grouped bars.
    X-axis = 7 models, one group of bars per period (all, seasons, day/night).
    Y-axis = network-average of per-station mean (or median) slope.
    Two figures per variable: one for mean, one for median.
    """
    print("\n[1] Network-wide summary bar charts")

    for var in VARIABLES:
        for stat in ["mean", "median"]:
            fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

            # Top panel: all + seasons
            # Bottom panel: day vs night
            panel_groups = [
                (axes[0], ["all", "DJF", "MAM", "JJA", "SON"], "Whole period & seasons"),
                (axes[1], ["day", "night"], "Day vs Night"),
            ]

            for ax, groups, panel_title in panel_groups:
                n_models = len(MODELS)
                n_groups = len(groups)
                bar_width = 0.8 / n_groups
                x = np.arange(n_models)

                for gi, grp in enumerate(groups):
                    vals = []
                    for model in MODELS:
                        col = f"{var}_{model}_{grp}_{stat}"
                        if col in df.columns:
                            vals.append(df[col].median())  # network median of per-station stat
                        else:
                            vals.append(np.nan)

                    offset = (gi - (n_groups - 1) / 2) * bar_width
                    color_alpha = 0.7 + 0.3 * (gi / max(n_groups - 1, 1))
                    ax.bar(x + offset, vals, bar_width * 0.9,
                           label=GROUP_LABELS.get(grp, grp),
                           alpha=color_alpha)

                ax.set_ylabel(f"Network median of {stat} dy/dx\n(per km²)")
                ax.axhline(0, color="black", linewidth=0.7, linestyle="-", alpha=0.5)
                ax.set_xticks(x)
                ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS],
                                  rotation=20, ha="right")
                ax.legend(frameon=True, loc="best")
                ax.set_title(panel_title, fontsize=LABEL_SIZE)
                bold_ticks(ax)

            fig.suptitle(
                f"{SPECIES}  |  {VAR_LABELS[var]}  —  {stat.capitalize()} slope "
                f"at 0.4° box edge\nNetwork summary ({len(df)} stations)",
                fontsize=TITLE_SIZE, y=1.01)

            savefig(f"network_summary_{var}_{stat}.png")


# ============================================================
# PLOT 2: HISTOGRAM / VIOLIN OF MEAN, MEDIAN, STD SLOPES ACROSS STATIONS
#   Shows the distribution of representativeness across the network.
# ============================================================

def plot_slope_distributions(df):
    """
    For each variable × stat (mean, median, std): box plot of per-station
    slope values, one box per model, all 7 side by side.
    Whiskers span 10th-90th percentile, no outliers shown.
    """
    print("\n[2] Slope distribution across stations (boxplots)")

    for var in VARIABLES:
        for stat in ["mean", "median", "std"]:
            data = []
            labels = []
            colors = []
            for model in MODELS:
                col = f"{var}_{model}_all_{stat}"
                if col in df.columns:
                    vals = df[col].dropna().values
                    data.append(vals)
                    labels.append(MODEL_LABELS[model])
                    colors.append(MODEL_COLORS[model])

            if not data:
                continue

            # Custom percentile-based boxplot: box = 10%-90%, no outliers
            # We build custom stats for each box
            bstats = []
            for vals in data:
                p10, q1, med, q3, p90 = np.nanpercentile(vals, [10, 25, 50, 75, 90])
                bstats.append({
                    "med": med,
                    "q1": p10,    # box lower edge = 10th percentile
                    "q3": p90,    # box upper edge = 90th percentile
                    "whislo": p10,  # whisker at 10th (same as box edge)
                    "whishi": p90,  # whisker at 90th (same as box edge)
                    "fliers": [],   # no outliers
                    "mean": np.nanmean(vals),
                })

            fig, ax = plt.subplots(figsize=(12, 6))
            bp = ax.bxp(bstats, showfliers=False, patch_artist=True,
                        widths=0.55)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.4)
            for line in bp["medians"]:
                line.set_color("black")
                line.set_linewidth(1.5)

            ax.set_xticklabels(labels, rotation=15, ha="right")
            ax.axhline(0, color="red", linewidth=1.0, linestyle="--", alpha=0.7)

            stat_display = stat.upper() if stat == "std" else stat.capitalize()
            ax.set_ylabel(f"Per-station {stat_display} dy/dx  (per km²)")
            ax.set_xlabel("Fitted model")
            ax.set_title(
                f"{SPECIES}  |  {VAR_LABELS[var]}  —  Distribution of {stat_display} slope "
                f"at 0.4° box  (10th–90th pctl)\n({len(df)} stations)",
                fontsize=TITLE_SIZE)
            bold_ticks(ax)
            savefig(f"boxplot_{stat}_slope_{var}.png")


# ============================================================
# PLOT 4: SEASONAL COMPARISON — grouped bars per model (unchanged)
#   Shows seasonal variation in slope.
# ============================================================

def plot_seasonal_comparison(df):
    """
    For each variable × stat: one figure with all 7 models,
    4 seasonal bars per model.
    """
    print("\n[4] Seasonal comparison (all models)")

    for var in VARIABLES:
        for stat in ["median"]:
            fig, ax = plt.subplots(figsize=(14, 6))
            n_models = len(MODELS)
            seasons = ["DJF", "MAM", "JJA", "SON"]
            n_seasons = len(seasons)
            bar_width = 0.8 / n_seasons
            x = np.arange(n_models)

            season_colors = {"DJF": "#4393c3", "MAM": "#66c164",
                             "JJA": "#d6604d", "SON": "#f4a460"}

            for si, szn in enumerate(seasons):
                vals = []
                for model in MODELS:
                    col = f"{var}_{model}_{szn}_{stat}"
                    if col in df.columns:
                        vals.append(df[col].median())
                    else:
                        vals.append(np.nan)
                offset = (si - (n_seasons - 1) / 2) * bar_width
                ax.bar(x + offset, vals, bar_width * 0.9,
                       color=season_colors[szn], alpha=0.8,
                       label=szn)

            ax.axhline(0, color="black", linewidth=0.7, linestyle="-", alpha=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS],
                              rotation=20, ha="right")
            ax.set_ylabel(f"Network median of {stat} dy/dx  (per km²)")
            ax.set_title(
                f"{SPECIES}  |  {VAR_LABELS[var]}  —  Seasonal {stat} slope "
                f"at 0.4° box\n({len(df)} stations)",
                fontsize=TITLE_SIZE)
            ax.legend(frameon=True, loc="best")
            bold_ticks(ax)
            savefig(f"seasonal_{var}_{stat}.png")


# ============================================================
# PLOT 5: SPATIAL MAPS (Cartopy) — median slope per station
#   Shows whether there is a geographic pattern.
# ============================================================

def plot_spatial_maps(df):
    """
    For each variable × model × stat (mean, median, std):
    scatter map colored by slope value.
    Bigger dots, colorbar shifted right.
    For cv_w: colorbar range >= 0.
    """
    if not PLOT_MAPS:
        return

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        print("  [SKIP] Cartopy not installed — skipping spatial maps")
        return

    print("\n[3] Spatial maps of mean / median / std slope")

    lat_col = "station_lat" if "station_lat" in df.columns else "model_lat"
    lon_col = "station_lon" if "station_lon" in df.columns else "model_lon"

    if lat_col not in df.columns or lon_col not in df.columns:
        print("  [SKIP] No lat/lon columns found")
        return

    for var in VARIABLES:
        for model in MODELS:
            for stat in ["mean", "median", "std"]:
                col = f"{var}_{model}_all_{stat}"
                if col not in df.columns:
                    continue

                sub = df[[lat_col, lon_col, col]].dropna().copy()
                if len(sub) < 10:
                    continue

                vals = sub[col].values

                # Colorbar range logic
                if stat == "std":
                    # std is always >= 0, use sequential colormap
                    vmin_val = 0.0
                    vmax_val = np.nanpercentile(vals, 95)
                    if vmax_val == 0:
                        vmax_val = 1e-6
                    cmap_use = "YlOrRd"
                elif var == "cv_w":
                    # CV values should be >= 0
                    vmin_val = 0.0
                    vmax_val = np.nanpercentile(vals, 95)
                    if vmax_val == 0:
                        vmax_val = 1e-6
                    cmap_use = "YlOrRd"
                else:
                    # ratio: symmetric diverging around 0
                    vabs = np.nanpercentile(np.abs(vals), 95)
                    if vabs == 0:
                        vabs = 1e-6
                    vmin_val = -vabs
                    vmax_val = vabs
                    cmap_use = "RdBu_r"

                fig = plt.figure(figsize=(14, 8))
                ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

                # Determine extent from data
                lat_min, lat_max = sub[lat_col].min() - 2, sub[lat_col].max() + 2
                lon_min, lon_max = sub[lon_col].min() - 2, sub[lon_col].max() + 2
                ax.set_extent([lon_min, lon_max, lat_min, lat_max],
                              crs=ccrs.PlateCarree())

                ax.add_feature(cfeature.LAND, facecolor="#f0f0f0")
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
                ax.gridlines(draw_labels=True, alpha=0.3)
                #add vmin,vmax inside ax.scatter if needed)
                sc = ax.scatter(sub[lon_col], sub[lat_col],
                                c=vals, cmap=cmap_use,
                                s=45, alpha=0.85, edgecolors="black",
                                linewidths=0.3,vmax=vmax_val,
                                transform=ccrs.PlateCarree())

                cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.08)
                stat_display = stat.upper() if stat == "std" else stat.capitalize()
                cbar.set_label(f"{stat_display} dy/dx  (per km²)",
                               fontsize=LABEL_SIZE)

                ax.set_title(
                    f"{SPECIES}  |  {VAR_LABELS[var]}  —  {MODEL_LABELS[model]} "
                    f"{stat_display} slope at 0.4° box\n({len(sub)} stations)",
                    fontsize=TITLE_SIZE)

                savefig(f"map_{stat}_slope_{var}_{model}.png")


# ============================================================
# PLOT 6: DAY vs NIGHT comparison
# ============================================================

def plot_day_night_comparison(df):
    """
    For each variable: grouped bar chart, day vs night median slope,
    one pair of bars per model.
    """
    print("\n[6] Day vs Night comparison")

    for var in VARIABLES:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(MODELS))
        bar_width = 0.35

        day_vals, night_vals = [], []
        for model in MODELS:
            col_day   = f"{var}_{model}_day_median"
            col_night = f"{var}_{model}_night_median"
            day_vals.append(df[col_day].median() if col_day in df.columns else np.nan)
            night_vals.append(df[col_night].median() if col_night in df.columns else np.nan)

        ax.bar(x - bar_width/2, day_vals, bar_width,
               color="#f4a460", alpha=0.85, label="Day")
        ax.bar(x + bar_width/2, night_vals, bar_width,
               color="#4a6fa5", alpha=0.85, label="Night")

        ax.axhline(0, color="black", linewidth=0.7, linestyle="-", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS],
                          rotation=20, ha="right")
        ax.set_ylabel(f"Network median of median dy/dx  (per km²)")
        ax.set_title(
            f"{SPECIES}  |  {VAR_LABELS[var]}  —  Day vs Night median slope "
            f"at 0.4° box\n({len(df)} stations)",
            fontsize=TITLE_SIZE)
        ax.legend(frameon=True, loc="best")
        bold_ticks(ax)
        savefig(f"day_night_{var}.png")


# ============================================================
# MAIN
# ============================================================

def main():
    print("plot_slope_stats.py")
    print("=" * 60)

    if not STATS_CSV.exists():
        print(f"ERROR: {STATS_CSV} not found.")
        print("Run aggregate_slope_stats.py first.")
        return

    df = pd.read_csv(STATS_CSV)
    print(f"Loaded: {len(df)} stations, {len(df.columns)} columns")

    plot_network_summary(df)
    plot_slope_distributions(df)
    plot_seasonal_comparison(df)
    plot_spatial_maps(df)
    plot_day_night_comparison(df)

    print(f"\nAll plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
# %%
