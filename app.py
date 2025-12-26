import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from geopy.distance import geodesic
import random
import time
import os

# ---------------- FIX WINDOWS WARNING ----------------
os.environ["OMP_NUM_THREADS"] = "1"

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Tirupati DRT System", layout="wide")

st.title("🚍 AI-Powered Demand Responsive Transit (DRT)")
st.subheader("Tirupati Pilgrim Pickup Optimization")

# ---------------- TIRUPATI PILGRIM LOCATIONS ----------------
PILGRIM_SPOTS = {
    "Tirumala Temple": (13.6833, 79.3470),
    "Alipiri": (13.6288, 79.4192),
    "Srivari Mettu": (13.6516, 79.4025),
    "Kapila Theertham": (13.6284, 79.4190),
    "ISKCON Temple": (13.6341, 79.4184),
    "Chandragiri Fort": (13.5846, 79.3175),
}

DESTINATIONS = {
    "Tirupati Railway Station": (13.6288, 79.4192),
    "Tirupati Bus Stand": (13.6283, 79.4197)
}

# ---------------- SIDEBAR ----------------
st.sidebar.header("🔧 Control Panel")

scenario = st.sidebar.selectbox(
    "Demand Scenario",
    ["Normal", "Peak", "Festival"]
)

hour = st.sidebar.slider("Hour of Travel", 5, 22, 10)

num_stops = st.sidebar.slider("Number of Pickup Stops (AI)", 2, 6, 4)

destination_name = st.sidebar.selectbox(
    "Final Destination",
    list(DESTINATIONS.keys())
)

run_button = st.sidebar.button("🚀 Run Optimization")

# ---------------- SYNTHETIC REAL-TIME DATA ----------------
def generate_pilgrim_data(scenario):
    rows = []

    for place, (lat, lon) in PILGRIM_SPOTS.items():
        if scenario == "Normal":
            count = random.randint(20, 40)
        elif scenario == "Peak":
            count = random.randint(40, 70)
        else:
            count = random.randint(70, 120)

        for _ in range(count):
            rows.append([
                place,
                lat + np.random.normal(0, 0.002),
                lon + np.random.normal(0, 0.002)
            ])

    return pd.DataFrame(rows, columns=["location", "lat", "lon"])

# ---------------- DISTANCE FUNCTION ----------------
def route_distance(route):
    dist = 0
    for i in range(len(route) - 1):
        dist += geodesic(route[i], route[i+1]).km
    return dist

# ---------------- MAIN LOGIC ----------------
if run_button:

    with st.spinner("Running AI Optimization..."):
        time.sleep(2)

        df = generate_pilgrim_data(scenario)

        X = df[["lat", "lon"]].values
        num_stops = min(num_stops, len(np.unique(X, axis=0)))

        kmeans = KMeans(n_clusters=num_stops, random_state=42)
        df["cluster"] = kmeans.fit_predict(X)
        centers = kmeans.cluster_centers_

        dest = DESTINATIONS[destination_name]

        # Route optimization (nearest neighbor)
        route = [dest]
        remaining = centers.tolist()

        while remaining:
            last = route[-1]
            next_point = min(
                remaining,
                key=lambda p: geodesic((last[0], last[1]), (p[0], p[1])).km
            )
            route.append(tuple(next_point))
            remaining.remove(next_point)

        route.append(dest)

        total_distance = route_distance(route)

    # ---------------- MAP ----------------
    m = folium.Map(location=dest, zoom_start=13)

    for _, r in df.iterrows():
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=3,
            color="blue",
            fill=True,
            fill_opacity=0.6
        ).add_to(m)

    for c in centers:
        folium.Marker(
            location=[c[0], c[1]],
            icon=folium.Icon(color="green", icon="bus"),
            tooltip="AI Pickup Stop"
        ).add_to(m)

    folium.Marker(
        location=dest,
        icon=folium.Icon(color="red", icon="flag"),
        tooltip=destination_name
    ).add_to(m)

    folium.PolyLine(route, color="purple", weight=4).add_to(m)

    # ---------------- DISPLAY ----------------
    col1, col2 = st.columns([3, 1])

    with col1:
        st_folium(m, width=900, height=550)

    with col2:
        st.metric("Total Route Distance (km)", f"{total_distance:.2f}")
        st.metric("Passengers Served", len(df))
        st.metric("AI Pickup Stops", num_stops)

        st.write("📊 Demand by Location")
        st.dataframe(df["location"].value_counts())

else:
    st.info("👈 Select options and click **Run Optimization**")

