# app.py
import streamlit as st
import random
import heapq
import math
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Grid Pathfinding — UCS & Best-First")

# -----------------------
# Utility / Generation
# -----------------------
def generate_grid(rows: int, cols: int, wall_prob: float, seed: int | None = None):
    rnd = random.Random(seed)
    grid = [[0 if rnd.random() > wall_prob else 1 for _ in range(cols)] for _ in range(rows)]
    # ensure start and goal are free
    grid[0][0] = 0
    grid[rows-1][cols-1] = 0
    return grid

def in_bounds(r, c, rows, cols):
    return 0 <= r < rows and 0 <= c < cols

def neighbors(r, c, rows, cols):
    for dr, dc in ((0,1),(1,0),(0,-1),(-1,0)):
        nr, nc = r+dr, c+dc
        if in_bounds(nr, nc, rows, cols):
            yield nr, nc

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

# -----------------------
# UCS (Dijkstra)
# -----------------------
def ucs_search(grid, start=(0,0), goal=None):
    rows, cols = len(grid), len(grid[0])
    if goal is None:
        goal = (rows-1, cols-1)
    pq = [(0.0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0.0}
    visited = set()

    while pq:
        g, current = heapq.heappop(pq)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            # reconstruct path
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path, cost_so_far[goal]

        r, c = current
        for nr, nc in neighbors(r, c, rows, cols):
            if grid[nr][nc] == 1:
                continue
            new_cost = cost_so_far[current] + 1  # uniform cost per move
            if (nr, nc) not in cost_so_far or new_cost < cost_so_far[(nr, nc)]:
                cost_so_far[(nr, nc)] = new_cost
                came_from[(nr, nc)] = current
                heapq.heappush(pq, (new_cost, (nr, nc)))

    return None, float("inf")

# -----------------------
# Greedy Best-First Search (Manhattan heuristic)
# -----------------------
def best_first_search(grid, start=(0,0), goal=None):
    rows, cols = len(grid), len(grid[0])
    if goal is None:
        goal = (rows-1, cols-1)
    pq = [(manhattan(start, goal), start)]
    came_from = {start: None}
    visited = set([start])

    while pq:
        _, current = heapq.heappop(pq)
        if current == goal:
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            # Note: greedy best-first doesn't track true path cost; return approximate cost = len-1
            return path, len(path)-1

        r, c = current
        for nr, nc in neighbors(r, c, rows, cols):
            if grid[nr][nc] == 1:
                continue
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                came_from[(nr, nc)] = current
                heapq.heappush(pq, (manhattan((nr,nc), goal), (nr, nc)))
    return None, float("inf")

# -----------------------
# Visualization helper
# -----------------------
def plot_grid(grid, path=None, figsize=(2,2)):
    rows, cols = len(grid), len(grid[0])
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    # draw cells
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                color = 'black'
            else:
                color = 'white'
            rect = plt.Rectangle((c, rows - 1 - r), 1, 1, facecolor=color, edgecolor='gray')
            ax.add_patch(rect)

    # draw path
    if path:
        xs = [c + 0.5 for (_, c) in path]
        ys = [rows - 1 - r + 0.5 for (r, _) in path]
        ax.plot(xs, ys, linewidth=3, color='green', zorder=5)
        ax.scatter([xs[0]], [ys[0]], s=120, color='blue', zorder=6, label='Start')
        ax.scatter([xs[-1]], [ys[-1]], s=120, color='red', zorder=6, label='Goal')
    else:
        # mark start and goal if no path
        ax.scatter([0.5], [rows-1-0 + 0.5], s=120, color='blue', zorder=6, label='Start')
        ax.scatter([cols-1 + 0.5], [0.5], s=120, color='red', zorder=6, label='Goal')

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()
    return fig

# -----------------------
# App UI (Uber-like layout)
# -----------------------
st.title("Grid Pathfinding — UCS & Best-First")

# Sidebar controls (like Uber reference)
with st.sidebar:
    st.header("Scenario controls")
    rows = st.slider("Rows", 5, 30, 10)
    cols = st.slider("Cols", 5, 30, 10)
    wall_prob = st.slider("Wall probability", 0.0, 0.6, 0.25, 0.05)
    show_grid_as_table = st.checkbox("Show grid as table", value=False)

    st.markdown("---")
    st.subheader("Solve using")
    # Algorithm buttons
    ucs_clicked = st.button("Run UCS ")
    best_clicked = st.button("Run Best-First")

    st.markdown("---")
    if st.button("Generate Grid"):
        st.session_state.grid = generate_grid(rows, cols, wall_prob)

    st.caption("Note: grid is regenerated only when you press Generate Grid. Start=top-left, Goal=bottom-right.")

# ensure a grid exists in session
if "grid" not in st.session_state:
    st.session_state.grid = generate_grid(rows, cols, wall_prob)

# Left column: problem description & controls summary (like Uber code)
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Instance")
    st.write(f"Rows: **{rows}**, Cols: **{cols}**")
    st.write(f"Wall probability: **{wall_prob:.2f}**")
    st.write("Start: **(0,0)** — Goal: **(rows-1, cols-1)**")
    st.markdown("**Legend**:")
    st.write("• Blue = Start | Red = Goal | Green line = Path | Black = Wall | White = Free")

    if show_grid_as_table:
        # show a small text-grid/table for quick inspection
        grid_text = "\n".join("".join("⬛" if cell==1 else "⬜" for cell in row) for row in st.session_state.grid)
        st.text(grid_text)

with col2:
    st.subheader("Grid & Result")

    # Run algorithms if their buttons were clicked
    solution_path = None
    solution_cost = None
    algo_name = None

    if ucs_clicked:
        solution_path, solution_cost = ucs_search(st.session_state.grid, (0,0), (rows-1,cols-1))
        algo_name = "UCS "
    elif best_clicked:
        solution_path, solution_cost = best_first_search(st.session_state.grid, (0,0), (rows-1,cols-1))
        algo_name = "Best-First"

    # Plot grid and path
    fig = plot_grid(st.session_state.grid, path=solution_path, figsize=(6,6))
    st.pyplot(fig)

    # Report results
    if algo_name:
        if solution_path:
            st.success(f"{algo_name} found a path (length = {len(solution_path)-1}, cost = {solution_cost}).")
            # Show path coordinates in a small table-like list
            with st.expander("Show path coordinates"):
                st.write(solution_path)
        else:
            st.error(f"{algo_name} found NO path in this instance.")

st.markdown("---")
st.caption("This demo shows uninformed-cost optimal search (UCS / Dijkstra) and a greedy informed search (Best-First using Manhattan heuristic).")
