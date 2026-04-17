# warehouse_robot_plotly_persistent_clean.py
import streamlit as st
import heapq
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.set_page_config(layout="wide", page_title="Warehouse Robot — Smooth Animation with Obstacles")
# --- Custom Page Styling ---
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #f5f7fa;
        color: #2c3e50;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #2c3e50;
        color: white;
    }
    /* Sidebar text */
    section[data-testid="stSidebar"] .css-1v3fvcr, section[data-testid="stSidebar"] .css-1d391kg {
        color: white !important;
    }
    /* Titles */
    h1, h2, h3, h4 {
        color: #34495e;
    }
    /* Dataframe */
    .stDataFrame {background-color: white;}
    </style>
""", unsafe_allow_html=True)

# -----------------------
# Warehouse settings
# -----------------------
ROWS, COLS = 7, 7
OBSTACLES = {(2, 2), (2, 3), (4, 4), (3, 5)}

shelves = {
    "S1": (1, 5),
    "S2": (3, 4),
    "S3": (5, 5)
}

items_in_shelves = {
    "S1": ["itemB", "itemC"],
    "S2": ["itemA", "itemB"],
    "S3": ["itemA", "itemD"]
}

robot_start = (1, 1)
packing_station = (0, 0)

# Manhattan distance heuristic
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# GBFS pathfinding
def gbfs(start, goal):
    frontier = []
    heapq.heappush(frontier, (heuristic(start, goal), start))
    came_from = {start: None}
    visited = set()

    while frontier:
        _, current = heapq.heappop(frontier)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = came_from[node]
            return path[::-1]

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if (0 <= neighbor[0] < ROWS and
                0 <= neighbor[1] < COLS and
                neighbor not in OBSTACLES and
                neighbor not in visited):
                came_from[neighbor] = current
                heapq.heappush(frontier, (heuristic(neighbor, goal), neighbor))
    return []

# -----------------------
# UI Controls
# -----------------------
st.title("🤖 Warehouse Robot — Smooth Animation (Obstacles Persistent)")

with st.sidebar:
    st.header("Scenario Controls")
    requested_item = st.selectbox("Select item to retrieve", sorted({item for items in items_in_shelves.values() for item in items}))
    steps_per_cell = st.slider("Smoothness (frames per step)", 2, 10, 5)

# Find closest shelf
available_shelves = [s for s, items in items_in_shelves.items() if requested_item in items]
closest_shelf = min(available_shelves, key=lambda s: heuristic(robot_start, shelves[s]))

# Paths
path_to_shelf = gbfs(robot_start, shelves[closest_shelf])
path_to_station = gbfs(shelves[closest_shelf], packing_station)
full_path = path_to_shelf + path_to_station[1:]

total_cost = len(full_path) - 1

st.subheader("Chosen Shelf")
st.write(f"Requested item: **{requested_item}**")
st.write(f"Closest shelf: **{closest_shelf}** at {shelves[closest_shelf]}")
st.write(f"Total movement cost: **{total_cost} units**")

# -----------------------
# Base Traces (Clean Legend)
# -----------------------
obstacle_trace = go.Scatter(
    x=[o[1] for o in OBSTACLES], y=[o[0] for o in OBSTACLES],
    mode="markers",
    marker=dict(color="black", size=20),
    name="Obstacle",
    line=dict(width=0)
)

shelf_traces = []
for s, pos in shelves.items():
    shelf_traces.append(go.Scatter(
        x=[pos[1]], y=[pos[0]],
        mode="markers+text",
        marker=dict(color="yellow", size=20),
        text=[s], textposition="middle center",
        name=f"Shelf {s}",
        line=dict(width=0)
    ))

start_trace = go.Scatter(
    x=[robot_start[1]], y=[robot_start[0]],
    mode="markers",
    marker=dict(color="green", size=20),
    name="Start",
    line=dict(width=0)
)

station_trace = go.Scatter(
    x=[packing_station[1]], y=[packing_station[0]],
    mode="markers",
    marker=dict(color="blue", size=20),
    name="Packing Station",
    line=dict(width=0)
)

# -----------------------
# Smooth Animation Frames (Static + Robot)
# -----------------------
frames = []
for i in range(len(full_path)-1):
    x0, y0 = full_path[i][1], full_path[i][0]
    x1, y1 = full_path[i+1][1], full_path[i+1][0]
    for step in np.linspace(0, 1, steps_per_cell):
        xi = x0 + (x1 - x0) * step
        yi = y0 + (y1 - y0) * step
        frames.append(go.Frame(data=[
            obstacle_trace,
            *shelf_traces,
            start_trace,
            station_trace,
            go.Scatter(
                x=[xi], y=[yi],
                mode="markers",
                marker=dict(color="red", size=14),
                name="Robot",
                line=dict(width=0)
            )
        ]))

# -----------------------
# Initial Figure
# -----------------------
fig = go.Figure(
    data=[
        obstacle_trace,
        *shelf_traces,
        start_trace,
        station_trace,
        go.Scatter(
            x=[full_path[0][1]], y=[full_path[0][0]],
            mode="markers",
            marker=dict(color="red", size=14),
            name="Robot",
            line=dict(width=0)
        )
    ],
    frames=frames
)

fig.update_layout(
    width=600, height=600,
    xaxis=dict(scaleanchor="y", range=[-0.5, COLS-0.5], dtick=1),
    yaxis=dict(range=[ROWS-0.5, -0.5], dtick=1),
    updatemenus=[{
        "type": "buttons",
        "buttons": [
            {"label": "Play", "method": "animate",
             "args": [None, {"frame": {"duration": 200, "redraw": True}, "fromcurrent": True}]},
            {"label": "Pause", "method": "animate",
             "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]}
        ]
    }]
)

st.plotly_chart(fig)

# Path table
df_paths = pd.DataFrame({
    "Step #": list(range(1, len(full_path)+1)),
    "Position": full_path
})
st.dataframe(df_paths)
