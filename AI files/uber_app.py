
# app.py
import streamlit as st
import heapq
import math
from copy import deepcopy

st.set_page_config(layout="wide", page_title="Uber Assignment — A* + Local Search")

# -----------------------
# Small example graph with traffic weights (edge costs)
# Nodes are strings; edges are undirected for simplicity
# -----------------------
GRAPH = {
    'A': {'B': 4, 'C': 2},
    'B': {'A': 4, 'D': 7, 'E': 3},
    'C': {'A': 2, 'D': 4},
    'D': {'B': 7, 'C': 4, 'E': 1},
    'E': {'B': 3, 'D': 1, 'F': 5},
    'F': {'E': 5, 'G': 2},
    'G': {'F': 2}
}

# (x,y) positions used only for a simple geometric heuristic
POS = {
    'A': (0, 0),
    'B': (2, 4),
    'C': (1, 1),
    'D': (5, 2),
    'E': (3, 3),
    'F': (6, 4),
    'G': (8, 4)
}

# -----------------------
# Heuristic (Euclidean) for A*
# -----------------------
def heuristic(n1, n2):
    x1, y1 = POS[n1]
    x2, y2 = POS[n2]
    return math.hypot(x1 - x2, y1 - y2)

# -----------------------
# A* search returning (path_list, cost)
# -----------------------
def astar(start, goal, graph):
    if start == goal:
        return [start], 0.0

    frontier = []
    # entries: (f_score, g_score, node)
    heapq.heappush(frontier, (heuristic(start, goal), 0.0, start))
    came_from = {start: None}
    gscore = {start: 0.0}
    closed = set()

    while frontier:
        fcurr, gcurr, current = heapq.heappop(frontier)
        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            # reconstruct path
            path = []
            n = current
            while n is not None:
                path.append(n)
                n = came_from[n]
            path.reverse()
            return path, gscore[current]

        for neigh, edge_cost in graph.get(current, {}).items():
            tentative = gscore[current] + edge_cost
            if tentative < gscore.get(neigh, float('inf')):
                came_from[neigh] = current
                gscore[neigh] = tentative
                f = tentative + heuristic(neigh, goal)
                heapq.heappush(frontier, (f, tentative, neigh))

    return [], float("inf")


# -----------------------
# Matching utilities (greedy + local swap improvement)
# -----------------------
def vehicle_compatible(driver_type, required):
    if required == "any":
        return True
    return driver_type.lower() == required.lower()

def build_cost_matrix(drivers, riders, graph):
    n = len(drivers)
    m = len(riders)
    cost = [[float('inf')] * m for _ in range(n)]
    paths = [[None] * m for _ in range(n)]
    for i, d in enumerate(drivers):
        for j, r in enumerate(riders):
            if vehicle_compatible(d["vehicle"], r["vehicle"]):
                p, c = astar(d["location"], r["location"], graph)
                cost[i][j] = c
                paths[i][j] = p
    return cost, paths

def greedy_assignment(cost):
    n = len(cost)
    m = len(cost[0]) if n else 0
    triples = [(i, j, cost[i][j]) for i in range(n) for j in range(m)]
    triples.sort(key=lambda x: x[2])
    used_i = set()
    used_j = set()
    assign = []
    for i, j, c in triples:
        if math.isinf(c):
            continue
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        assign.append((i, j))
        if len(used_j) == m or len(used_i) == n:
            break
    return assign

def total_weighted_cost(cost, assignment, riders, beta=1.5):
    total = 0.0
    for d_idx, r_idx in assignment:
        w = 1.0 + beta * riders[r_idx]["urgency"]
        total += w * cost[d_idx][r_idx]
    return total

def try_pairwise_swap(cost, assignment, riders, beta=1.5):
    best = list(assignment)
    best_val = total_weighted_cost(cost, best, riders, beta)
    k = len(assignment)
    improved = False
    for a in range(k):
        for b in range(a+1, k):
            d1, r1 = assignment[a]
            d2, r2 = assignment[b]
            # check compatibility via cost not inf
            if math.isinf(cost[d1][r2]) or math.isinf(cost[d2][r1]):
                continue
            cand = list(assignment)
            cand[a] = (d1, r2)
            cand[b] = (d2, r1)
            val = total_weighted_cost(cost, cand, riders, beta)
            if val < best_val - 1e-9:
                best_val = val
                best = cand
                improved = True
    return improved, best

def improve_local_search(cost, assignment, riders, beta=1.5, max_iters=50):
    cur = list(assignment)
    for _ in range(max_iters):
        improved, newa = try_pairwise_swap(cost, cur, riders, beta)
        if not improved:
            break
        cur = newa
    return cur


# -----------------------
# Simple demo scenario controls in UI
# -----------------------
st.title("🚖 Uber Driver–Rider Assignment — A* (routing) + Local Search (matching)")

with st.sidebar:
    st.header("Scenario controls")
    beta = st.slider("Urgency weight (beta)", 0.0, 3.0, 1.5, 0.1)
    show_cost_matrix = st.checkbox("Show cost matrix", value=True)

st.markdown("### Scenario description")
st.write(
    "Small synthetic city graph. Drivers and riders have node locations (A..G). "
    "A* computes full multi-node routes and ETAs (edge weights represent travel time/traffic). "
    "Matching first uses a greedy nearest-ETA assignment (respecting vehicle compatibility), "
    "then improves assignments with pairwise swap local search to reduce urgency-weighted ETA."
)

# sample drivers & riders for demo
drivers = [
    {"id": "D1", "location": "A", "vehicle": "Sedan"},
    {"id": "D2", "location": "B", "vehicle": "SUV"},
    {"id": "D3", "location": "C", "vehicle": "Sedan"}
]

riders = [
    {"id": "R1", "location": "E", "vehicle": "SUV", "urgency": 3},
    {"id": "R2", "location": "D", "vehicle": "Sedan", "urgency": 2},
    {"id": "R3", "location": "B", "vehicle": "Sedan", "urgency": 1}
]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Drivers")
    for d in drivers:
        st.write(f"• {d['id']}: node={d['location']}, vehicle={d['vehicle']}")

with col2:
    st.subheader("Riders")
    for r in riders:
        st.write(f"• {r['id']}: node={r['location']}, vehicle={r['vehicle']}, urgency={r['urgency']}")

# Build cost matrix using A*
cost_matrix, paths = build_cost_matrix(drivers, riders, GRAPH)

if show_cost_matrix:
    st.subheader("Cost matrix (ETA) — rows: drivers, cols: riders")
    # format for display
    header = ["Driver\\Rider"] + [r["id"] for r in riders]
    rows = []
    for i, d in enumerate(drivers):
        row = [d["id"]]
        for j, r in enumerate(riders):
            c = cost_matrix[i][j]
            if math.isinf(c):
                row.append("INF")
            else:
                row.append(f"{c:.1f}")
        rows.append(row)
    st.table([header] + rows)

# Greedy assignment
greedy_assign = greedy_assignment(cost_matrix)
greedy_cost = total_weighted_cost(cost_matrix, greedy_assign, riders, beta)
st.subheader("Initial (Greedy) assignment")
if not greedy_assign:
    st.write("No compatible matches found.")
else:
    for (d_idx, r_idx) in greedy_assign:
        st.write(f"• {drivers[d_idx]['id']} → {riders[r_idx]['id']} | ETA: {cost_matrix[d_idx][r_idx]:.2f} | Route: {' → '.join(paths[d_idx][r_idx])}")

st.write(f"Initial urgency-weighted total cost = **{greedy_cost:.2f}**")

# Local search improvement
improved_assign = improve_local_search(cost_matrix, greedy_assign, riders, beta=beta, max_iters=50)
improved_cost = total_weighted_cost(cost_matrix, improved_assign, riders, beta)
if improved_assign != greedy_assign:
    st.subheader("Improved assignment (after local search)")
    for (d_idx, r_idx) in improved_assign:
        st.write(f"• {drivers[d_idx]['id']} → {riders[r_idx]['id']} | ETA: {cost_matrix[d_idx][r_idx]:.2f} | Route: {' → '.join(paths[d_idx][r_idx])}")
    st.write(f"Improved urgency-weighted total cost = **{improved_cost:.2f}**")
else:
    st.info("Local search found no better pairwise swaps (greedy was already good).")

# Show side-by-side comparison table
st.subheader("Assignment comparison")
def assignment_to_rows(assign):
    rows = []
    for d_idx, r_idx in assign:
        rows.append({
            "driver": drivers[d_idx]["id"],
            "driver_loc": drivers[d_idx]["location"],
            "rider": riders[r_idx]["id"],
            "rider_loc": riders[r_idx]["location"],
            "urgency": riders[r_idx]["urgency"],
            "eta": (None if math.isinf(cost_matrix[d_idx][r_idx]) else round(cost_matrix[d_idx][r_idx],2)),
            "route": (paths[d_idx][r_idx] if paths[d_idx][r_idx] else [])
        })
    return rows

import pandas as pd
df_initial = pd.DataFrame(assignment_to_rows(greedy_assign))
df_improved = pd.DataFrame(assignment_to_rows(improved_assign))

cols = st.columns(2)
with cols[0]:
    st.write("Greedy")
    st.dataframe(df_initial)
with cols[1]:
    st.write("Improved")
    st.dataframe(df_improved)

st.markdown("---")
st.caption("Notes: A* produces full multi-node paths (shown under 'route'). Local search here is a simple pairwise-swap improvement — in production you'd add swaps, relocations, capacity constraints, and time windows.")
