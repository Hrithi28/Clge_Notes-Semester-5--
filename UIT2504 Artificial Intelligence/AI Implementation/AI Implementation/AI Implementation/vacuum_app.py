# vacuum_app.py
import streamlit as st
import matplotlib.pyplot as plt
import heapq

# --- Environment Setup ---
GRID_ROWS = 6
GRID_COLS = 6
OBSTACLES = {(1, 1), (2, 3), (4, 2), (3, 4)}
DIRTY_TILES = {(0, 5), (5, 0), (5, 5)}
START = (0, 0)


# --- Helper Functions ---
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def neighbors(pos):
    x, y = pos
    moves = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
    return [(nx, ny) for nx, ny in moves
            if 0 <= nx < GRID_ROWS and 0 <= ny < GRID_COLS and (nx, ny) not in OBSTACLES]


def mst_cost(points):
    """Compute MST cost with Prim's algorithm."""
    if not points:
        return 0
    points = list(points)
    visited = {points[0]}
    cost = 0
    while len(visited) < len(points):
        edges = []
        for v in visited:
            for u in points:
                if u not in visited:
                    edges.append((manhattan(v, u), v, u))
        min_edge = min(edges, key=lambda e: e[0])
        cost += min_edge[0]
        visited.add(min_edge[2])
    return cost


def heuristic(pos, dirty):
    if not dirty:
        return 0
    nearest = min(manhattan(pos, d) for d in dirty)
    return nearest + mst_cost(dirty)


# --- A* Search ---
def a_star(start, dirty_tiles):
    state = (start, frozenset(dirty_tiles))
    frontier = [(heuristic(start, dirty_tiles), 0, state, [start])]
    visited = set()

    while frontier:
        f, g, (pos, remaining), path = heapq.heappop(frontier)
        if not remaining:
            return path

        if (pos, remaining) in visited:
            continue
        visited.add((pos, remaining))

        for nb in neighbors(pos):
            new_remaining = set(remaining)
            if nb in new_remaining:
                new_remaining.remove(nb)
            g2 = g + 1
            f2 = g2 + heuristic(nb, new_remaining)
            heapq.heappush(frontier, (f2, g2, (nb, frozenset(new_remaining)), path + [nb]))
    return None


# --- Streamlit UI ---
st.title("Robotic Vacuum Cleaner - A* Search with Relaxed Heuristics")

st.write("**Environment Setup:**")
st.write(f"Grid size: {GRID_ROWS}x{GRID_COLS}")
st.write(f"Obstacles: {OBSTACLES}")
st.write(f"Dirty tiles: {DIRTY_TILES}")
st.write(f"Start position: {START}")

path = a_star(START, DIRTY_TILES)

if path:
    st.write(f"**Optimal Path ({len(path)-1} steps):** {path}")

    # Draw grid
    fig, ax = plt.subplots()
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if (r, c) in OBSTACLES:
                ax.add_patch(plt.Rectangle((c, GRID_ROWS - 1 - r), 1, 1, color="black"))
            elif (r, c) in DIRTY_TILES:
                ax.add_patch(plt.Rectangle((c, GRID_ROWS - 1 - r), 1, 1, color="yellow"))
            elif (r, c) == START:
                ax.add_patch(plt.Rectangle((c, GRID_ROWS - 1 - r), 1, 1, color="green"))
            else:
                ax.add_patch(plt.Rectangle((c, GRID_ROWS - 1 - r), 1, 1, edgecolor="gray", facecolor="white"))

    # Draw path
    for i in range(len(path) - 1):
        x1, y1 = path[i][1] + 0.5, GRID_ROWS - 1 - path[i][0] + 0.5
        x2, y2 = path[i + 1][1] + 0.5, GRID_ROWS - 1 - path[i + 1][0] + 0.5
        ax.arrow(x1 - 0.5, y1 - 0.5, x2 - x1, y2 - y1,
                 head_width=0.2, length_includes_head=True, color="red")

    ax.set_xlim(0, GRID_COLS)
    ax.set_ylim(0, GRID_ROWS)
    ax.set_aspect('equal')
    ax.axis("off")
    st.pyplot(fig)

else:
    st.error("No path found!")
