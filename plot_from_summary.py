#%%
"""
plot_from_summary.py
====================
All diagnostic plots for the multi-station model-selection analysis.
Reads only the two compact files produced by aggregate_station_data.py.

Functions — comment / uncomment in main() to choose what to produce:
  plot_type1_barplots()   5 figs × 2 subplots: global (station×timestep) winner frequency
  plot_type2_barplots()   5 figs × 2 subplots: station-level modal winner frequency
  plot_maps_option_a()    5 figs × 2 subplots: colour = winner model, size ∝ win%
  plot_maps_option_c()    5 figs × 2 subplots: colour = winner model, edge lw ∝ win%
  plot_maps_option_e()    5 figs × 2 subplots: contour background of win% + coloured circles
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import warnings

matplotlib.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    warnings.warn("cartopy not found — maps will use plain matplotlib axes.")

try:
    from scipy.interpolate import griddata
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not found — Option-E contour background disabled.")

# ============================================================
# CONFIGURATION  (edit here)
# ============================================================

SUMMARY_DIR  = Path("/mnt/store01/agkiokas/CAMS/fit_outputs/summary")
STATION_CSV  = SUMMARY_DIR / "station_summary.csv"
COUNTS_CSV   = SUMMARY_DIR / "global_counts.csv"
OUTPUT_DIR   = SUMMARY_DIR / "plots"

VARIABLES = ["ratio", "cv_w"]
MODELS    = ["linear", "quadratic", "cubic", "logarithmic",
             "exponential", "power", "saturating"]
CRITERIA  = ["r2", "adj_r2", "aic", "aicc", "bic"]

VAR_LABELS  = {"ratio": "Concentration Ratio", "cv_w": "Weighted CV (%)"}
CRIT_LABELS = {
    "r2":     r"$R^2$",
    "adj_r2": r"Adj-$R^2$",
    "aic":    "AIC",
    "aicc":   "AICc",
    "bic":    "BIC",
}
MODEL_LABELS = {
    "linear":      "Linear",
    "quadratic":   "Quadratic",
    "cubic":       "Cubic",
    "logarithmic": "Logarithmic",
    "exponential": "Exponential",
    "power":       "Power-law",
    "saturating":  "Saturating",
}
MODEL_COLORS = {
    "linear":      "#4C72B0",
    "quadratic":   "#DD8452",
    "cubic":       "#55A868",
    "logarithmic": "#C44E52",
    "exponential": "#8172B2",
    "power":       "#937860",
    "saturating":  "#DA8BC3",
}

# --- Figure aesthetics ---
FIG_BAR_W  = 14
FIG_BAR_H  = 9
FIG_MAP_W  = 20      # wider to give each map panel more breathing room
FIG_MAP_H  = 8
DPI        = 150
FONT_SIZE  = 11
TITLE_SIZE = 13

# --- Map extent [lon_min, lon_max, lat_min, lat_max] ---
MAP_EXTENT = [72, 136, 15, 55]

# --- Option A: circle size proportional to win% ---
# SIZE_INVERT = False  →  large marker = convincing win (high win%)
# SIZE_INVERT = True   →  small marker = convincing win
SIZE_INVERT     = False
MARKER_SIZE_MIN = 12    # pts²  floor so weakest wins are still visible
MARKER_SIZE_MAX = 130   # pts²  reduced from 280 to limit overlap
# Alpha is baked into RGBA colours so overlapping circles do NOT
# accumulate opacity.  Change this value to adjust global transparency.
MARKER_ALPHA    = 0.75

# --- Option C: uniform size, edge linewidth ∝ win% ---
MARKER_SIZE_UNIFORM = 40    # pts²
LINEWIDTH_MIN       = 0.2
LINEWIDTH_MAX       = 3.5
OPTION_C_ALPHA      = 0.80  # baked into fill colour for option C

# --- Option E: contour colormap and grid resolution ---
CONTOUR_CMAP        = "YlOrRd"
CONTOUR_LEVELS      = 10
CONTOUR_ALPHA       = 0.45
GRID_NX             = 300
GRID_NY             = 200
OPTION_E_MARKER_SIZE  = 35
OPTION_E_MARKER_ALPHA = 0.88  # slightly more opaque — sits on contour

# ============================================================
# SHARED HELPERS
# ============================================================

def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_s = pd.read_csv(STATION_CSV)
    df_c = pd.read_csv(COUNTS_CSV)
    print(f"Loaded {len(df_s)} stations and {len(df_c)} count rows.")
    return df_s, df_c


def _win_pct_to_size(win_pct_series: pd.Series) -> pd.Series:
    """Map win% → marker size (pts²), respecting SIZE_INVERT."""
    norm = win_pct_series.clip(0, 100) / 100.0
    if SIZE_INVERT:
        norm = 1.0 - norm
    return MARKER_SIZE_MIN + norm * (MARKER_SIZE_MAX - MARKER_SIZE_MIN)


def _win_pct_to_lw(win_pct_series: pd.Series) -> pd.Series:
    """Map win% → edge linewidth, for Option C."""
    norm = win_pct_series.clip(0, 100) / 100.0
    return LINEWIDTH_MIN + norm * (LINEWIDTH_MAX - LINEWIDTH_MIN)


def _model_color_series(winner_series: pd.Series, alpha: float = MARKER_ALPHA) -> list:
    """
    Return list of RGBA tuples for each station based on winner model.
    Alpha is baked in so overlapping circles do not accumulate opacity —
    each circle renders at a uniform colour regardless of local density.
    """
    result = []
    for w in winner_series:
        r, g, b = matplotlib.colors.to_rgb(MODEL_COLORS.get(w, "#cccccc"))
        result.append((r, g, b, alpha))
    return result


def _annotate_bar(ax, bars, values, total, fontsize=9):
    """Annotate each bar with absolute count and percentage."""
    for bar, val in zip(bars, values):
        pct = 100.0 * val / total if total > 0 else 0.0
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005 * ax.get_ylim()[1],
                f"{val:,}\n({pct:.1f}%)",
                ha="center", va="bottom",
                fontsize=fontsize, fontweight="bold", color="#111111",
            )


def _model_legend_patches() -> list:
    return [
        mpatches.Patch(facecolor=MODEL_COLORS[m], label=MODEL_LABELS[m])
        for m in MODELS
    ]


def _size_legend_handles(label_pcts=(25, 50, 75, 100)) -> list:
    handles = []
    r, g, b = matplotlib.colors.to_rgb("#888888")
    legend_color = (r, g, b, MARKER_ALPHA)
    for pct in label_pcts:
        s = _win_pct_to_size(pd.Series([pct])).iloc[0]
        h = plt.scatter([], [], s=s, c=[legend_color], alpha=1.0,
                        edgecolors="#333333", linewidths=0.35,
                        label=f"Win {pct}%")
        handles.append(h)
    return handles


def _lw_legend_handles(label_pcts=(25, 50, 75, 100)) -> list:
    handles = []
    for pct in label_pcts:
        lw = _win_pct_to_lw(pd.Series([pct])).iloc[0]
        h = mlines.Line2D(
            [], [], color="grey", linewidth=lw,
            marker="o", markerfacecolor="grey", markersize=8,
            label=f"Win {pct}%",
        )
        handles.append(h)
    return handles


def _make_map_ax(fig, subplot_spec):
    """Return a cartopy GeoAxes (or plain axes if cartopy absent).
    subplot_spec must be a tuple e.g. (1, 2, 1) — unpacked with * internally.
    """
    if HAS_CARTOPY:
        ax = fig.add_subplot(*subplot_spec, projection=ccrs.PlateCarree())
        ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND,      facecolor="#f0f0f0", zorder=0)
        ax.add_feature(cfeature.OCEAN,     facecolor="#d0e8f0", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6,       zorder=1)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.5,
                       linestyle=":", zorder=1)
        ax.gridlines(draw_labels=True, linewidth=0.3,
                     color="grey", alpha=0.5, zorder=1)
    else:
        ax = fig.add_subplot(*subplot_spec)
        ax.set_xlim(MAP_EXTENT[0], MAP_EXTENT[1])
        ax.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linewidth=0.3, alpha=0.5)
    return ax


def _scatter_on_map(ax, lon, lat, colors, sizes, lw_vals=None, zorder=5):
    """
    Scatter plot on map axis (cartopy-aware).
    Alpha must already be baked into the RGBA tuples in `colors`.
    alpha=1.0 is therefore always used on the scatter call so that
    overlapping circles do NOT accumulate opacity.
    Edge colour is a fixed dark grey for all markers; linewidth is
    either uniform (Option A/E) or variable (Option C via lw_vals).
    """
    kwargs = dict(
        c=colors, s=sizes, alpha=1.0, zorder=zorder,
        edgecolors="#333333",
        linewidths=0.35 if lw_vals is None else lw_vals,
    )
    if HAS_CARTOPY:
        kwargs["transform"] = ccrs.PlateCarree()
    ax.scatter(lon, lat, **kwargs)


# ============================================================
# TYPE 1 — Global (station × timestep) winner frequency
# ============================================================

def plot_type1_barplots(df_stations: pd.DataFrame,
                        df_counts: pd.DataFrame):
    """
    5 figures × 2 subplots (ratio top, cv_w bottom).
    Bar height = total (station × timestep) count across all stations.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for crit in CRITERIA:
        fig, axes = plt.subplots(2, 1, figsize=(FIG_BAR_W, FIG_BAR_H), dpi=DPI)
        fig.suptitle(
            f"Type 1 — Global model winner frequency  |  {CRIT_LABELS[crit]}\n"
            f"All stations × all timesteps",
            fontsize=TITLE_SIZE, fontweight="bold",
        )

        for row_idx, var in enumerate(VARIABLES):
            ax     = axes[row_idx]
            subset = df_counts[(df_counts["var"] == var) &
                               (df_counts["criterion"] == crit)]

            counts = [
                int(subset.loc[subset["model"] == m, "count"].sum())
                for m in MODELS
            ]
            total  = sum(counts)
            x      = np.arange(len(MODELS))
            bars   = ax.bar(
                x, counts, width=0.6,
                color=[MODEL_COLORS[m] for m in MODELS],
                edgecolor="white", linewidth=0.8, alpha=0.85, zorder=3,
            )
            _annotate_bar(ax, bars, counts, total, fontsize=FONT_SIZE - 2)

            ax.set_xticks(x)
            ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS],
                               fontsize=FONT_SIZE, rotation=20, ha="right")
            ax.set_ylabel("Winner count  (station × timestep)",
                          fontsize=FONT_SIZE)
            ax.set_title(VAR_LABELS[var], fontsize=FONT_SIZE + 1,
                         fontweight="bold", pad=5)
            ax.set_ylim(0, max(counts) * 1.30 if max(counts) > 0 else 10)
            ax.grid(axis="y", alpha=0.35, linestyle="--", zorder=0)
            ax.tick_params(labelsize=FONT_SIZE - 1)
            ax.text(0.99, 0.97, f"Total: {total:,}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=FONT_SIZE - 2, color="grey")

        fig.tight_layout()
        out = OUTPUT_DIR / f"type1_{crit}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out.name}")


# ============================================================
# TYPE 2 — Station-level modal winner frequency
# ============================================================

def plot_type2_barplots(df_stations: pd.DataFrame,
                        df_counts: pd.DataFrame):
    """
    5 figures × 2 subplots (ratio top, cv_w bottom).
    Bar height = number of stations for which that model is the modal winner.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n_stations = len(df_stations)

    for crit in CRITERIA:
        fig, axes = plt.subplots(2, 1, figsize=(FIG_BAR_W, FIG_BAR_H), dpi=DPI)
        fig.suptitle(
            f"Type 2 — Station-level modal winner frequency  |  {CRIT_LABELS[crit]}\n"
            f"{n_stations} stations — one modal winner per station",
            fontsize=TITLE_SIZE, fontweight="bold",
        )

        for row_idx, var in enumerate(VARIABLES):
            ax         = axes[row_idx]
            winner_col = f"{var}_{crit}_winner"

            counts = [
                int((df_stations[winner_col] == m).sum())
                for m in MODELS
            ]
            total  = sum(counts)
            x      = np.arange(len(MODELS))
            bars   = ax.bar(
                x, counts, width=0.6,
                color=[MODEL_COLORS[m] for m in MODELS],
                edgecolor="white", linewidth=0.8, alpha=0.85, zorder=3,
            )
            _annotate_bar(ax, bars, counts, n_stations, fontsize=FONT_SIZE - 2)

            ax.set_xticks(x)
            ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS],
                               fontsize=FONT_SIZE, rotation=20, ha="right")
            ax.set_ylabel("Number of stations", fontsize=FONT_SIZE)
            ax.set_title(VAR_LABELS[var], fontsize=FONT_SIZE + 1,
                         fontweight="bold", pad=5)
            ax.set_ylim(0, max(counts) * 1.30 if max(counts) > 0 else 10)
            ax.grid(axis="y", alpha=0.35, linestyle="--", zorder=0)
            ax.tick_params(labelsize=FONT_SIZE - 1)
            ax.text(0.99, 0.97,
                    f"Assigned: {total}/{n_stations}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=FONT_SIZE - 2, color="grey")

        fig.tight_layout()
        out = OUTPUT_DIR / f"type2_{crit}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out.name}")


# ============================================================
# MAP OPTION A — colour = winner, size ∝ win%
# ============================================================

def plot_maps_option_a(df_stations: pd.DataFrame,
                       df_counts: pd.DataFrame):
    """
    5 figures × 2 subplots (ratio left, cv_w right).
    Circle colour = modal winning model.
    Circle area   ∝ win% (large = convincing if SIZE_INVERT=False).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for crit in CRITERIA:
        fig = plt.figure(figsize=(FIG_MAP_W, FIG_MAP_H), dpi=DPI)
        fig.suptitle(
            f"Modal winner per station  |  {CRIT_LABELS[crit]}\n"
            f"Colour: winning model   Size: win% "
            f"({'large=convincing' if not SIZE_INVERT else 'small=convincing'})",
            fontsize=TITLE_SIZE, fontweight="bold",
        )

        for col_idx, var in enumerate(VARIABLES):
            ax = _make_map_ax(fig, (1, 2, col_idx + 1))
            ax.set_title(VAR_LABELS[var], fontsize=FONT_SIZE + 1,
                         fontweight="bold", pad=6)

            winner_col  = f"{var}_{crit}_winner"
            win_pct_col = f"{var}_{crit}_win_pct"

            sub = df_stations.dropna(
                subset=["lat", "lon", winner_col, win_pct_col]
            ).copy()

            if sub.empty:
                continue

            colors = _model_color_series(sub[winner_col], alpha=MARKER_ALPHA)
            sizes  = _win_pct_to_size(sub[win_pct_col])

            # Sort ascending by win% so the most convincing (largest)
            # circles are drawn last and sit on top
            order = sub[win_pct_col].argsort().values
            _scatter_on_map(
                ax, sub["lon"].values[order], sub["lat"].values[order],
                [colors[i] for i in order], sizes.values[order],
                zorder=5,
            )

            # Dominant model annotation (bottom-left of each panel)
            vc = sub[winner_col].value_counts()
            if not vc.empty:
                dom_model = vc.index[0]
                dom_pct   = 100.0 * vc.iloc[0] / len(sub)
                r, g, b   = matplotlib.colors.to_rgb(MODEL_COLORS[dom_model])
                ax.text(
                    0.02, 0.02,
                    f"Dominant: {MODEL_LABELS[dom_model]} ({dom_pct:.0f}%)",
                    transform=ax.transAxes,
                    fontsize=FONT_SIZE - 2, fontweight="bold",
                    color=(r, g, b, 1.0),
                    bbox=dict(facecolor="white", alpha=0.80,
                              edgecolor="none", pad=2),
                    zorder=10,
                )

        # Shared legends below maps
        color_leg = _model_legend_patches()
        size_leg  = _size_legend_handles()

        fig.legend(handles=color_leg, title="Model", loc="lower left",
                   bbox_to_anchor=(0.02, 0.0), fontsize=FONT_SIZE - 1,
                   title_fontsize=FONT_SIZE, framealpha=0.9,
                   ncol=2)
        fig.legend(handles=size_leg, title="Win %", loc="lower right",
                   bbox_to_anchor=(0.98, 0.0), fontsize=FONT_SIZE - 1,
                   title_fontsize=FONT_SIZE, framealpha=0.9)

        fig.tight_layout(rect=[0, 0.12, 1, 0.97])
        out = OUTPUT_DIR / f"map_optA_{crit}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out.name}")


# ============================================================
# MAP OPTION C — colour = winner, uniform size, edge lw ∝ win%
# ============================================================

def plot_maps_option_c(df_stations: pd.DataFrame,
                       df_counts: pd.DataFrame):
    """
    5 figures × 2 subplots.
    Circle colour = modal winning model.
    Edge linewidth ∝ win%  (thick edge = convincing win).
    All circles same size — minimal overlap.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for crit in CRITERIA:
        fig = plt.figure(figsize=(FIG_MAP_W, FIG_MAP_H), dpi=DPI)
        fig.suptitle(
            f"Modal winner per station  |  {CRIT_LABELS[crit]}\n"
            f"Colour: winning model   Edge width: win% (thick = convincing)",
            fontsize=TITLE_SIZE, fontweight="bold",
        )

        for col_idx, var in enumerate(VARIABLES):
            ax = _make_map_ax(fig, (1, 2, col_idx + 1))
            ax.set_title(VAR_LABELS[var], fontsize=FONT_SIZE + 1,
                         fontweight="bold", pad=6)

            winner_col  = f"{var}_{crit}_winner"
            win_pct_col = f"{var}_{crit}_win_pct"

            sub = df_stations.dropna(
                subset=["lat", "lon", winner_col, win_pct_col]
            ).copy()

            if sub.empty:
                continue

            colors  = _model_color_series(sub[winner_col], alpha=OPTION_C_ALPHA)
            lw_vals = _win_pct_to_lw(sub[win_pct_col]).values
            sizes   = np.full(len(sub), MARKER_SIZE_UNIFORM)

            _scatter_on_map(
                ax, sub["lon"].values, sub["lat"].values,
                colors, sizes, lw_vals=lw_vals,
                zorder=5,
            )

        color_leg = _model_legend_patches()
        lw_leg    = _lw_legend_handles()

        fig.legend(handles=color_leg, title="Model", loc="lower left",
                   bbox_to_anchor=(0.02, 0.0), fontsize=FONT_SIZE - 1,
                   title_fontsize=FONT_SIZE, framealpha=0.9, ncol=2)
        fig.legend(handles=lw_leg, title="Win %", loc="lower right",
                   bbox_to_anchor=(0.98, 0.0), fontsize=FONT_SIZE - 1,
                   title_fontsize=FONT_SIZE, framealpha=0.9)

        fig.tight_layout(rect=[0, 0.12, 1, 0.97])
        out = OUTPUT_DIR / f"map_optC_{crit}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out.name}")


# ============================================================
# MAP OPTION E — interpolated win% contour + coloured circles
# ============================================================

def plot_maps_option_e(df_stations: pd.DataFrame,
                       df_counts: pd.DataFrame):
    """
    5 figures × 2 subplots.
    Background: spatial interpolation (griddata) of win% → filled contour.
      Shows how regionally coherent the 'confidence' in the local winner is.
    Foreground: small coloured circles (colour = modal winner).
    """
    if not HAS_SCIPY:
        print("  Option E skipped: scipy not available.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lon_grid = np.linspace(MAP_EXTENT[0], MAP_EXTENT[1], GRID_NX)
    lat_grid = np.linspace(MAP_EXTENT[2], MAP_EXTENT[3], GRID_NY)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    for crit in CRITERIA:
        fig = plt.figure(figsize=(FIG_MAP_W, FIG_MAP_H), dpi=DPI)
        fig.suptitle(
            f"Option E — Spatial win% confidence + modal winner  |  "
            f"{CRIT_LABELS[crit]}\n"
            f"Background: interpolated win%   Circles: winning model",
            fontsize=TITLE_SIZE, fontweight="bold",
        )

        for col_idx, var in enumerate(VARIABLES):
            ax = _make_map_ax(fig, (1, 2, col_idx + 1))
            ax.set_title(VAR_LABELS[var], fontsize=FONT_SIZE + 1,
                         fontweight="bold", pad=6)

            winner_col  = f"{var}_{crit}_winner"
            win_pct_col = f"{var}_{crit}_win_pct"

            sub = df_stations.dropna(
                subset=["lat", "lon", winner_col, win_pct_col]
            ).copy()

            if len(sub) < 4:
                continue

            lons     = sub["lon"].values
            lats     = sub["lat"].values
            win_pcts = sub[win_pct_col].values

            # Interpolate win% onto regular grid
            z = griddata(
                (lons, lats), win_pcts,
                (lon_mesh, lat_mesh),
                method="linear",
            )
            z = np.clip(z, 0, 100)

            # Contour background
            contour_kwargs = dict(
                levels=CONTOUR_LEVELS,
                cmap=CONTOUR_CMAP,
                alpha=CONTOUR_ALPHA,
                vmin=0, vmax=100,
                zorder=2,
            )
            if HAS_CARTOPY:
                cf = ax.contourf(lon_mesh, lat_mesh, z,
                                 transform=ccrs.PlateCarree(),
                                 **contour_kwargs)
            else:
                cf = ax.contourf(lon_mesh, lat_mesh, z, **contour_kwargs)

            cbar = fig.colorbar(cf, ax=ax, orientation="vertical",
                                fraction=0.04, pad=0.02,
                                label="Win % (modal winner)")
            cbar.ax.tick_params(labelsize=FONT_SIZE - 2)

            # Coloured station circles on top
            colors = _model_color_series(sub[winner_col], alpha=OPTION_E_MARKER_ALPHA)
            _scatter_on_map(
                ax, lons, lats, colors,
                np.full(len(sub), OPTION_E_MARKER_SIZE),
                zorder=6,
            )

        color_leg = _model_legend_patches()
        fig.legend(handles=color_leg, title="Model", loc="lower center",
                   bbox_to_anchor=(0.5, 0.0), fontsize=FONT_SIZE - 1,
                   title_fontsize=FONT_SIZE, framealpha=0.9, ncol=4)

        fig.tight_layout(rect=[0, 0.10, 1, 0.97])
        out = OUTPUT_DIR / f"map_optE_{crit}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out.name}")


# ============================================================
# MAIN — comment out what you don't need
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_stations, df_counts = _load_data()

    print("\n--- Type 1: global (station × timestep) barplots ---")
    #plot_type1_barplots(df_stations, df_counts)

    print("\n--- Type 2: station-level modal winner barplots ---")
    #plot_type2_barplots(df_stations, df_counts)

    print("\n--- Maps Option A: colour + size ---")
    plot_maps_option_a(df_stations, df_counts)

    print("\n--- Maps Option C: colour + edge linewidth ---")
    plot_maps_option_c(df_stations, df_counts)

    print("\n--- Maps Option E: contour background + circles ---")
    #plot_maps_option_e(df_stations, df_counts)

    print(f"\nAll outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
# %%
