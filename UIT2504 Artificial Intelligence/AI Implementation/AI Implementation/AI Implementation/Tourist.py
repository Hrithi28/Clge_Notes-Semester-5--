import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import random
from collections import deque

# ---------------------------
# BFS Shortest Path
# ---------------------------
def bfs_shortest_path(graph, start, goal):
    visited = set()
    queue = deque([[start]])

    if start == goal:
        return [start]

    while queue:
        path = queue.popleft()
        node = path[-1]

        if node not in visited:
            for neighbor in graph.neighbors(node):
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)

                if neighbor == goal:
                    return new_path
            visited.add(node)
    return None

# ---------------------------
# Random Graph Generator
# ---------------------------
def generate_random_city(num_landmarks, num_connections):
    G = nx.Graph()
    landmarks = [f"Landmark_{i}" for i in range(1, num_landmarks+1)]
    G.add_nodes_from(landmarks)

    # Random connections
    all_possible_edges = [(a, b) for idx, a in enumerate(landmarks) for b in landmarks[idx+1:]]
    random.shuffle(all_possible_edges)
    for edge in all_possible_edges[:num_connections]:
        G.add_edge(*edge)
    return G

# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(page_title="Tourist Landmark Path Finder", layout="wide")
st.title("🏙 Tourist Landmark Path Finder (BFS)")
st.markdown("Find the **shortest path** between two landmarks (by number of stops) using **BFS**.")

# User controls for graph generation
st.sidebar.header("City Map Settings")
num_landmarks = st.sidebar.slider("Number of Landmarks", 5, 15, 8)
num_connections = st.sidebar.slider("Number of Connections", num_landmarks-1, num_landmarks*(num_landmarks-1)//2, num_landmarks+2)

# Generate graph
G = generate_random_city(num_landmarks, num_connections)

# Landmark selection
start_node = st.selectbox("Select Start Landmark", list(G.nodes))
end_node = st.selectbox("Select Destination Landmark", list(G.nodes))

# Path finding
if st.button("Find Shortest Path"):
    path = bfs_shortest_path(G, start_node, end_node)

    if path:
        st.success(f"Shortest path from **{start_node}** to **{end_node}**: {' ➡ '.join(path)}")
        st.info(f"Number of stops: {len(path) - 1}")

        # Draw graph with highlighted path
        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(8, 6))

        nx.draw_networkx_nodes(G, pos, node_size=800, node_color="lightblue")
        nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7)
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")

        if len(path) > 1:
            edge_list = list(zip(path, path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=edge_list, width=3, edge_color="red")
            nx.draw_networkx_nodes(G, pos, nodelist=path, node_color="orange")

        st.pyplot(plt)
    else:
        st.error("No path found between the selected landmarks.")

# Show generated graph (without path)
st.subheader("🗺 Current City Map")
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(8, 6))
nx.draw_networkx_nodes(G, pos, node_size=800, node_color="lightgreen")
nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7)
nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")
st.pyplot(plt)
