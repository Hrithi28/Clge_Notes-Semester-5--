# app.py
import streamlit as st
import random
from heapq import heappush, heappop
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Smart Drone Delivery — Uniform Cost Search")

# ============================================================
# Problem Definition
# A drone must navigate from a warehouse (start) to a customer
# (goal) on a grid city, avoiding no-fly zones and choosing the
# path with the least total battery cost.
#
# Search Strategy
# Uniform Cost Search (UCS) — uninformed, optimal for nonnegative costs.
# ============================================================

# -----------------------
# Helpers
# -----------------------
Coord = tuple[int, int]  # (row, col)

def parse_coord_list(text: str) -> set[Coord]:
    """
    Parse coordinates from text like:
      0,1; 2,3; 4,0
    Returns set of (r, c).
    """
    blocked = set()
    text = text.strip()
    if not text:
        return blocked
    parts = [p.strip() for p in text.split(";")]
    for p in parts:
        if not p:
            continue
        if "," not in p:
            raise ValueError("Each coordinate must be 'r,c' and pairs separated by ';'")
        r_str, c_str = p.split(",", 1)
        r = int(r_str.strip())
        c = int(c_str.strip())
        blocked.add((r, c))
    return blocked

def in_bounds(r, c, rows, cols):
    return 0 <= r < rows and 0 <= c < cols

def neighbors_4(r, c, rows, cols):
    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
        nr, nc = r + dr, c + dc
        if in_bounds(nr, nc, rows, cols):
            yield nr, nc

def build_city(rows: int, cols: int, seed: int, cost_min: int, cost_max: int,
               random_block_prob: float) -> tuple[np.ndarray, set[Coord]]:
    """
    Create a grid of costs and a random set of blocked cells (no-fly zones).
    Costs represent per-cell traversal battery cost (to ENTER that cell).
    """
    rnd = random.Random(seed)
    costs = np.zeros((rows, cols), dtype=int)
    blocked = set()

    for r in range(rows):
        for c in range(cols):
            # Randomly block a cell based on probability
            if rnd.random() < random_block_prob:
                blocked.add((r, c))
                costs[r, c] = 0
            else:
                # Terrain cost in [cost_min, cost_max]
                costs[r, c] = rnd.randint(cost_min, cost_max)

    return costs, blocked

def ucs(start: Coord, goal: Coord, costs: np.ndarray, blocked: set[Coord]) -> tuple[list[Coord], float, int]:
    """
    Uniform Cost Search on a grid.
    - cost to move into a neighbor cell = costs[neighbor]
    - blocked cells are not traversable
    Returns (path, total_cost, nodes_expanded)
    """
    rows, cols = costs.shape
    if start in blocked or goal in blocked:
        return [], math.inf, 0

    # Priority queue: (g_cost, r, c)
    pq = []
    heappush(pq, (0.0, start[0], start[1]))

    came_from = {start: None}
    g_cost = {start: 0.0}

    expanded = 0

    while pq:
        g, r, c = heappop(pq)
        expanded += 1

        if (r, c) == goal:
            # Reconstruct path
            path = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            path.reverse()
            return path, g_cost[goal], expanded

        # If this entry is stale, skip
        if g > g_cost[(r, c)]:
            continue

        for nr, nc in neighbors_4(r, c, rows, cols):
            if (nr, nc) in blocked:
                continue
            step = costs[nr, nc]  # cost to ENTER the neighbor
            if step < 0:
                # Guard against negative (shouldn't happen in this UI)
                continue
            tentative = g + step
            if (nr, nc) not in g_cost or tentative < g_cost[(nr, nc)]:
                g_cost[(nr, nc)] = tentative
                came_from[(nr, nc)] = (r, c)
                heappush(pq, (tentative, nr, nc))

    return [], math.inf, expanded

def grid_dataframe(costs: np.ndarray, blocked: set[Coord], start: Coord, goal: Coord, path: set[Coord] | None = None) -> pd.DataFrame:
    rows, cols = costs.shape
    data = []
    for r in range(rows):
        row = []
        for c in range(cols):
            if (r, c) == start and (r, c) == goal:
                label = "S=G"
            elif (r, c) == start:
                label = "S"
            elif (r, c) == goal:
                label = "G"
            elif (r, c) in blocked:
                label = "X"  # no-fly
            else:
                label = str(costs[r, c])

            if path and (r, c) in path and (r, c) not in {start, goal}:
                label = f"{label} •"  # mark path cell
            row.append(label)
        data.append(row)
    df = pd.DataFrame(data, columns=[f"C{c+1}" for c in range(cols)])
    df.index = [f"R{r+1}" for r in range(rows)]
    return df

def plot_grid_with_path(costs: np.ndarray, blocked: set[Coord], start: Coord, goal: Coord, path: list[Coord]):
    rows, cols = costs.shape
    mat = costs.astype(float).copy()
    for (r, c) in blocked:
        mat[r, c] = np.nan

    fig, ax = plt.subplots(figsize=(6, 6))
    # use pastel colormap
    im = ax.imshow(mat, cmap="Blues", interpolation="none")

    # Overlay blocked as black squares
    for (r, c) in blocked:
        ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color="black"))

    # Start/Goal markers with colors
    ax.scatter(start[1], start[0], c="green", s=150, marker="o", edgecolors="black", label="Start")
    ax.scatter(goal[1], goal[0], c="red", s=150, marker="*", edgecolors="black", label="Goal")

    # Path (blue line + dots)
    if path:
        xs = [c for (_, c) in path]
        ys = [r for (r, _) in path]
        ax.plot(xs, ys, linewidth=2, color="blue", marker="o", markersize=6)

    ax.set_title("City Grid — Colored Path Visualization")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.grid(which="both", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.legend()
    st.pyplot(fig)

# -----------------------
# UI
# -----------------------
st.title("🚁 Smart Drone Delivery — Uniform Cost Search (UCS)")
st.markdown("""
**Goal**: Find the **least-battery-cost** path from warehouse (**S**) to customer (**G**) on a grid city, avoiding **no-fly zones (X)**.  
**Search**: **Uniform Cost Search (UCS)** — optimal for nonnegative costs, uninformed (no heuristic).
""")

with st.sidebar:
    st.header("🧭 Grid & Costs")
    rows = st.number_input("Rows", min_value=2, max_value=50, value=8)
    cols = st.number_input("Columns", min_value=2, max_value=50, value=10)
    seed = st.number_input("Random seed", min_value=0, max_value=100000, value=42, help="For reproducibility.")
    random_block_prob = st.slider("Random no-fly probability per cell", 0.0, 0.6, 0.15, 0.01)
    cost_min = st.number_input("Min cell cost", min_value=1, max_value=50, value=1)
    cost_max = st.number_input("Max cell cost", min_value=1, max_value=50, value=5)

    st.caption("Cost is the battery used to **enter** a cell. Random no-fly zones are marked as X.")

    st.divider()
    st.header("📍 Start / Goal")
    start_r = st.number_input("Start row (S)", min_value=0, max_value=rows-1, value=0)
    start_c = st.number_input("Start col (S)", min_value=0, max_value=cols-1, value=0)
    goal_r = st.number_input("Goal row (G)", min_value=0, max_value=rows-1, value=rows-1)
    goal_c = st.number_input("Goal col (G)", min_value=0, max_value=cols-1, value=cols-1)

    st.divider()
    st.header("⛔ Extra No-Fly Zones (optional)")
    blocked_text = st.text_area(
        "Enter coordinates as 'r,c; r,c; ...' (0-indexed). Example: 1,2; 3,4; 5,0",
        value=""
    )

# Build instance
error = None
try:
    costs, random_blocked = build_city(rows, cols, seed, cost_min, cost_max, random_block_prob)
    user_blocked = parse_coord_list(blocked_text)
    blocked = set(random_blocked) | set(user_blocked)

    start = (int(start_r), int(start_c))
    goal = (int(goal_r), int(goal_c))

    if not in_bounds(*start, rows, cols) or not in_bounds(*goal, rows, cols):
        error = "Start or Goal is out of bounds."
except Exception as e:
    error = str(e)

if error:
    st.error(error)
    st.stop()

# Buttons
c1, c2 = st.columns(2)
with c1:
    if st.button("🔁 Regenerate City"):
        # Rebuild with same settings (Streamlit rerun will rebuild automatically)
        pass
with c2:
    run_search = st.button("🚀 Run Uniform Cost Search")

# Show base grid
st.subheader("City Grid — Costs & No-Fly Zones")
base_df = grid_dataframe(costs, blocked, start, goal)
st.dataframe(base_df, use_container_width=True)

# Run UCS
if run_search:
    path, total_cost, expanded = ucs(start, goal, costs, blocked)

    if path:
        path_set = set(path)
        st.success(f"✅ Path found!  Total battery cost = **{total_cost:.0f}** | Steps = **{len(path)-1}** | Nodes expanded = **{expanded}**")
        st.subheader("Path Overlay (• on cells)")
        overlay_df = grid_dataframe(costs, blocked, start, goal, path_set)
        st.dataframe(overlay_df, use_container_width=True)

        st.subheader("Visualization")
        plot_grid_with_path(costs, blocked, start, goal, path)
    else:
        if total_cost == math.inf:
            st.error(f"❌ No path exists from S{start} to G{goal}. Try reducing no-fly probability, moving S/G, or adjusting extra blocks.")
        else:
            st.error("❌ Search failed unexpectedly.")

# Notes
st.markdown("---")
st.caption(
    "UCS expands the frontier in order of **lowest accumulated cost**. "
    "Edge cost = cost to enter the neighbor cell. No negative costs allowed. "
    "Grid uses 4-directional moves (↑ ↓ ← →)."
)
