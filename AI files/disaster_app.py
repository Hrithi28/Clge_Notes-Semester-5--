# disaster_app.py
import streamlit as st
import heapq

# ------------------------
# A* Search Implementation
# ------------------------

def heuristic(node, goal):
    # Relaxed problem heuristic: straight-line distance (simulated with abs diff)
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

def astar_search(start, goal, grid):
    rows, cols = len(grid), len(grid[0])
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start, [start]))

    visited = set()

    while open_set:
        _, cost, current, path = heapq.heappop(open_set)

        if current == goal:
            return path, cost

        if current in visited:
            continue
        visited.add(current)

        x, y = current
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] != 1:
                new_cost = cost + 1
                heapq.heappush(open_set, (new_cost + heuristic((nx, ny), goal), new_cost, (nx, ny), path + [(nx, ny)]))

    return None, None

# ------------------------
# Streamlit App
# ------------------------

st.title("🚑 Disaster Response Routing with A* Search")

st.write("This app finds an optimal route from a **start location** to a **goal location** "
         "on a grid map using **A\\*** Search with a **relaxed problem heuristic**.")

rows = st.number_input("Number of rows in grid", min_value=3, max_value=20, value=5)
cols = st.number_input("Number of columns in grid", min_value=3, max_value=20, value=5)

# Create grid (0 = empty, 1 = obstacle)
grid = [[0 for _ in range(cols)] for _ in range(rows)]

st.subheader("Set Obstacles")
obstacle_coords = st.text_area("Enter obstacle coordinates (e.g., 0,1; 2,2)", value="1,1; 3,2")
if obstacle_coords.strip():
    for coord in obstacle_coords.split(";"):
        try:
            r, c = map(int, coord.strip().split(","))
            if 0 <= r < rows and 0 <= c < cols:
                grid[r][c] = 1
        except:
            pass

start_x = st.number_input("Start X", min_value=0, max_value=rows-1, value=0)
start_y = st.number_input("Start Y", min_value=0, max_value=cols-1, value=0)
goal_x = st.number_input("Goal X", min_value=0, max_value=rows-1, value=rows-1)
goal_y = st.number_input("Goal Y", min_value=0, max_value=cols-1, value=cols-1)

if st.button("Find Route"):
    path, cost = astar_search((start_x, start_y), (goal_x, goal_y), grid)
    if path:
        st.success(f"Path found with cost {cost}: {path}")
        
        # Show grid with path
        display_grid = [["⬜" if cell == 0 else "⬛" for cell in row] for row in grid]
        for (x, y) in path:
            if (x, y) != (start_x, start_y) and (x, y) != (goal_x, goal_y):
                display_grid[x][y] = "🟩"
        display_grid[start_x][start_y] = "🟦"  # Start
        display_grid[goal_x][goal_y] = "🟥"    # Goal

        st.write("### Grid Map")
        for row in display_grid:
            st.write(" ".join(row))

    else:
        st.error("No path found.")

st.info("🛈 **Legend:** 🟦 Start | 🟥 Goal | 🟩 Path | ⬛ Obstacle | ⬜ Free space")
