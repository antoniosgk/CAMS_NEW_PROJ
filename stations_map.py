#%%
import pandas as pd
import folium
from folium.plugins import MarkerCluster, Search
#%%
name_st=["1006A","2629A"]

def load_stations(stations_path):
    """Load station table. Keeps all rows; marks invalid rows (NaNs) for later filtering."""
    df = pd.read_csv(stations_path, sep="\t")
    df = df.reset_index().rename(columns={"index": "idx"})

    # normalize column names
    for col in df.columns:
        if col.lower().startswith("station"):
            df = df.rename(columns={col: "Station_Name"})
        if col.lower().startswith("lat"):
            df = df.rename(columns={col: "Latitude"})
        if col.lower().startswith("lon"):
            df = df.rename(columns={col: "Longitude"})
        if col.lower().startswith("alt"):
            df = df.rename(columns={col: "Altitude"})

    expected = ["idx", "Station_Name", "Latitude", "Longitude", "Altitude"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Stations file missing expected columns: {missing}")

    for col in ["Latitude", "Longitude", "Altitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["is_valid"] = df[["Latitude", "Longitude", "Altitude"]].notna().all(axis=1)

    return df[expected + ["is_valid"]]


def altitude_color(alt):
    if alt < 200:
        return "blue"
    elif alt < 500:
        return "green"
    elif alt < 1000:
        return "orange"
    elif alt < 2000:
        return "red"
    else: return "black"

def find_station(df, text):
    mask = df["Station_Name"].astype(str).str.contains(text, case=False, na=False)
    return df.loc[mask, ["idx", "Station_Name", "Latitude", "Longitude", "Altitude", "is_valid"]]   
def create_station_map(
    stations_df,
    output_html="stations_map.html",
    use_cluster=True,
    tiles="CartoDB positron",
    highlight_station=None,
    zoom_to_highlight=True
):
    """
    Create an interactive folium map with all valid stations.
    highlight_station can be:
      - None
      - a string
      - a list of strings
    Matching station(s) are shown larger.
    """

    valid = stations_df[stations_df["is_valid"]].copy()

    if valid.empty:
        raise ValueError("No valid stations found.")

    # default center: mean of all stations
    center_lat = valid["Latitude"].mean()
    center_lon = valid["Longitude"].mean()

    # normalize highlight_station into a list
    highlight_list = []
    if highlight_station is not None:
        if isinstance(highlight_station, str):
            highlight_list = [highlight_station]
        else:
            highlight_list = list(highlight_station)

    # find highlighted station(s)
    highlighted = pd.DataFrame(columns=valid.columns)
    if highlight_list:
        mask = pd.Series(False, index=valid.index)
        for st in highlight_list:
            mask |= valid["Station_Name"].astype(str).str.contains(st, case=False, na=False)
        highlighted = valid[mask].copy()

        # if exactly one match, center on it
        if zoom_to_highlight and len(highlighted) == 1:
            center_lat = highlighted.iloc[0]["Latitude"]
            center_lon = highlighted.iloc[0]["Longitude"]

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=4,
        tiles=tiles
    )

    # normal stations
    if use_cluster:
        marker_group = MarkerCluster(name="Stations").add_to(m)
    else:
        marker_group = folium.FeatureGroup(name="Stations").add_to(m)

    # highlighted stations
    highlight_group = folium.FeatureGroup(name="Highlighted stations").add_to(m)

    highlighted_idx = set(highlighted["idx"].values) if not highlighted.empty else set()

    for _, row in valid.iterrows():
        popup_html = f"""
        <b>{row['Station_Name']}</b><br>
        Latitude: {row['Latitude']:.5f}<br>
        Longitude: {row['Longitude']:.5f}<br>
        Altitude: {row['Altitude']:.2f} m
        """

        is_highlighted = row["idx"] in highlighted_idx

        if is_highlighted:
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=12,
                color="black",
                weight=3,
                fill=True,
                fill_color=altitude_color(row["Altitude"]),
                fill_opacity=1.0,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"SELECTED: {row['Station_Name']}",
            ).add_to(highlight_group)
        else:
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=6,
                color=altitude_color(row["Altitude"]),
                weight=1,
                fill=True,
                fill_color=altitude_color(row["Altitude"]),
                fill_opacity=0.6,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=row["Station_Name"],
            ).add_to(marker_group)

    # zoom behavior
    if zoom_to_highlight and not highlighted.empty:
        bounds = [
            [highlighted["Latitude"].min(), highlighted["Longitude"].min()],
            [highlighted["Latitude"].max(), highlighted["Longitude"].max()],
        ]
    else:
        bounds = [
            [valid["Latitude"].min(), valid["Longitude"].min()],
            [valid["Latitude"].max(), valid["Longitude"].max()],
        ]

    m.fit_bounds(bounds)

    folium.LayerControl().add_to(m)
    m.save(output_html)

    if highlight_station is not None:
        print("\nMatched highlighted station(s):")
        if highlighted.empty:
            print(f"No station matched: {highlight_station}")
        else:
            print(highlighted[["idx", "Station_Name", "Latitude", "Longitude", "Altitude"]])

    return m

stations_df = load_stations("/home/agkiokas/CAMS/CHINESE_STATIONS_INFO_2015_2023.txt")

for st in name_st:
    print(find_station(stations_df, st))

create_station_map(
    stations_df,
    output_html="stations_map.html",
    highlight_station=name_st
)
# %%
