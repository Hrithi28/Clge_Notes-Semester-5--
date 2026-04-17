import streamlit as st
import random
from collections import deque

# Grid size
ROWS, COLS = 10, 10

# Initialize session state
if "grid" not in st.session_state:
    st.session_state.grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]
if "start" not in st.session_state:
    st.session_state.start = (0, 0)
if "end" not in st.session_state:
    st.session_state.end = (ROWS - 1, COLS - 1)

# Functions
def generate_random_grid():
    st.session_state.grid = [
        [1 if random.random() < 0.2 else 0 for _ in range(COLS)] for _ in range(ROWS)
    ]
    st.session_state.grid[st.session_state.start[0]][st.session_state.start[1]] = 0
    st.session_state.grid[st.session_state.end[0]][st.session_state.end[1]] = 0

def bfs():
    return search("BFS")

def dfs():
    return search("DFS")

def iddfs():
    depth = 0
    while True:
        path = dls(st.session_state.start, [], set(), depth)
        if path:
            return path
        depth += 1
        if depth > ROWS * COLS:
            return None

def dls(node, path, visited, limit):
    if node == st.session_state.end:
        return path + [node]
    if limit <= 0:
        return None
    visited.add(node)
    r, c = node
    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS and st.session_state.grid[nr][nc] == 0 and (nr, nc) not in visited:
            res = dls((nr, nc), path + [node], visited, limit - 1)
            if res:
                return res
    return None

def search(method):
    start = st.session_state.start
    end = st.session_state.end
    visited = set()
    queue = deque([[start]]) if method == "BFS" else [[start]]

    while queue:
        path = queue.popleft() if method == "BFS" else queue.pop()
        node = path[-1]
        if node == end:
            return path
        if node in visited:
            continue
        visited.add(node)
        r, c = node
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS and st.session_state.grid[nr][nc] == 0:
                new_path = list(path)
                new_path.append((nr, nc))
                if method == "BFS":
                    queue.append(new_path)
                else:
                    queue.append(new_path)
    return None

def display_grid(path=None):
    for r in range(ROWS):
        row_str = ""
        for c in range(COLS):
            if (r, c) == st.session_state.start:
                row_str += "🟢 "
            elif (r, c) == st.session_state.end:
                row_str += "🔴 "
            elif path and (r, c) in path:
                row_str += "🟡 "
            elif st.session_state.grid[r][c] == 1:
                row_str += "⬛ "
            else:
                row_str += "⬜ "
        st.write(row_str)

# Sidebar Controls
st.sidebar.title("Controls")
if st.sidebar.button("Generate Random Grid"):
    generate_random_grid()

if st.sidebar.button("Solve with BFS"):
    path = bfs()
    st.session_state.path = path

if st.sidebar.button("Solve with DFS"):
    path = dfs()
    st.session_state.path = path

if st.sidebar.button("Solve with IDDFS"):
    path = iddfs()
    st.session_state.path = path

# Main display
st.title("Pathfinding Visualizer")
display_grid(st.session_state.get("path"))
