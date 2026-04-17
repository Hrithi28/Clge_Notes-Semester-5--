import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import heapq

# ----------------------------
# A* Search Implementation
# ----------------------------
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_search(graph, start, goal, traffic_delay, blocked_roads):
    queue = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}

    while queue:
        _, current = heapq.heappop(queue)
        if current == goal:
            break
        for neighbor in graph.neighbors(current):
            if (current, neighbor) in blocked_roads or (neighbor, current) in blocked_roads:
                continue
            new_cost = cost_so_far[current] + graph[current][neighbor]['weight'] + traffic_delay.get((current, neighbor), 0)
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal)
                heapq.heappush(queue, (priority, neighbor))
                came_from[neighbor] = current
    
    if goal not in came_from:
        return None
    
    path, current = [], goal
    while current:
        path.append(current)
        current = came_from[current]
    return path[::-1]

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="🚑 Emergency Ambulance Routing", page_icon="🚨", layout="wide")

st.title("🚑 Emergency Ambulance Routing System")
st.markdown("<p style='font-size:18px; color:gray;'>Find the fastest path for ambulances considering traffic and roadblocks.</p>", unsafe_allow_html=True)

# Sidebar input
st.sidebar.header("⚙️ Simulation Controls")
st.sidebar.markdown("Set up the grid, choose start and goal, and add traffic/roadblocks.")

# Grid setup
grid_size = st.sidebar.slider("Select Grid Size", 5, 10, 6)
graph = nx.grid_2d_graph(grid_size, grid_size)
for (u, v) in graph.edges():
    graph[u][v]['weight'] = 1

# Select nodes
nodes = list(graph.nodes())
start = st.sidebar.selectbox("🚩 Select Start Point", nodes, index=0)
goal = st.sidebar.selectbox("🏥 Select Goal Point", nodes, index=len(nodes)-1)

# Traffic delays
traffic_delay = {}
st.sidebar.subheader("🚦 Traffic Delays")
st.sidebar.markdown("Choose roads with heavy traffic (adds extra delay).")
traffic_edges = st.sidebar.multiselect("Select Roads with Traffic", list(graph.edges()))
for edge in traffic_edges:
    traffic_delay[edge] = 3

# Blocked roads
st.sidebar.subheader("⛔ Blocked Roads")
blocked_roads = st.sidebar.multiselect("Select Blocked Roads", list(graph.edges()))

# Run search
path = a_star_search(graph, start, goal, traffic_delay, blocked_roads)

# Display result
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🗺️ Route Visualization")
    plt.figure(figsize=(6, 6))
    pos = {node: node for node in graph.nodes()}
    nx.draw(graph, pos, node_size=600, node_color="lightgray", with_labels=True, font_size=8)
    nx.draw_networkx_edges(graph, pos, edgelist=traffic_edges, edge_color="orange", width=2, style="dashed")
    nx.draw_networkx_edges(graph, pos, edgelist=blocked_roads, edge_color="red", width=2)
    if path:
        edges_in_path = list(zip(path, path[1:]))
        nx.draw_networkx_nodes(graph, pos, nodelist=path, node_color="lightgreen")
        nx.draw_networkx_edges(graph, pos, edgelist=edges_in_path, edge_color="blue", width=3)
    st.pyplot(plt)

with col2:
    st.subheader("📋 Route Details")
    if path:
        st.success(f"Shortest Path Found: {path}")
        st.info(f"Path Length (with delays): {len(path)-1 + sum(traffic_delay.values())}")
    else:
        st.error("No path found! Please adjust traffic or blocked roads.")
