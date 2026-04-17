# app.py
import streamlit as st
import random
import math
import pandas as pd

st.set_page_config(layout="wide", page_title="Seating Assignment — Simulated Annealing")

# ============================================================
# Problem Definition
# Assign students to seats such that no two adjacent students
# belong to the same department, while satisfying spacing
# constraints. Aim for a near-optimal layout.
#
# Search Strategy
# Local Search with Randomness (Simulated Annealing-style),
# but UI keeps parameters minimal (no explicit temperature knobs).
# ============================================================

# -----------------------
# Helpers & Model
# -----------------------
def parse_dept_counts(text: str):
    """
    Parse department counts from text like:
      CSE:10, ECE:8, MECH:6
    Returns list of department labels (one per student).
    """
    items = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError("Format must be Dept:Count, comma-separated (e.g., CSE:10, ECE:8)")
        dept, cnt = part.split(":")
        dept = dept.strip()
        cnt = int(cnt.strip())
        if cnt < 0:
            raise ValueError("Counts must be non-negative.")
        items.extend([dept] * cnt)
    return items

def manhattan(p, q):
    return abs(p[0] - q[0]) + abs(p[1] - q[1])

def index_to_rc(idx, cols):
    return (idx // cols, idx % cols)

def rc_to_index(r, c, cols):
    return r * cols + c

def make_initial_state(dept_list, seed=None):
    rnd = random.Random(seed)
    arr = dept_list[:]
    rnd.shuffle(arr)
    return arr

def cost_components(state, rows, cols, spacing):
    """
    Returns (adjacent_conflicts, spacing_conflicts) for the layout.
    - Adjacent = 4-neighbors (up, down, left, right) with same dept.
    - Spacing = any pair within Manhattan distance <= spacing (excluding dist==1, counted separately)
    """
    n = len(state)
    adj_conf = 0
    space_conf = 0

    for idx in range(n):
        r, c = index_to_rc(idx, cols)
        d = state[idx]

        # Adjacent (4-neighbors)
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                j = rc_to_index(nr, nc, cols)
                if j > idx and state[j] == d:  # count each pair once
                    adj_conf += 1

        # Spacing (within <= spacing, but exclude dist==0 and dist==1 adjacents)
        if spacing >= 2:
            for rr in range(max(0, r - spacing), min(rows, r + spacing + 1)):
                # Tighten horizontal window by Manhattan distance
                max_h = spacing - abs(rr - r)
                for cc in range(max(0, c - max_h), min(cols, c + max_h + 1)):
                    if rr == r and cc == c:
                        continue
                    if abs(rr - r) + abs(cc - c) <= spacing:
                        j = rc_to_index(rr, cc, cols)
                        if j > idx and state[j] == d:
                            # exclude direct adjacency (already counted)
                            if abs(rr - r) + abs(cc - c) > 1:
                                space_conf += 1

    return adj_conf, space_conf

def total_cost(state, rows, cols, spacing, w_adj=1.0, w_space=0.5):
    adj_c, space_c = cost_components(state, rows, cols, spacing)
    return w_adj * adj_c + w_space * space_c, adj_c, space_c

def neighbor_swap(state, rnd: random.Random):
    """Swap two random seats."""
    n = len(state)
    i, j = rnd.sample(range(n), 2)
    new_state = state[:]
    new_state[i], new_state[j] = new_state[j], new_state[i]
    return new_state

def simulated_annealing_optimize(initial_state, rows, cols, spacing, max_iters=6000, seed=None):
    """
    SA-style local search with a fixed internal schedule.
    No temperature controls exposed in UI (kept simple).
    """
    rnd = random.Random(seed)
    current = initial_state
    current_cost, _, _ = total_cost(current, rows, cols, spacing)

    best = current[:]
    best_cost = current_cost

    # Internal schedule (simple & stable)
    # Start temp scaled by current cost (at least 1.0)
    T0 = max(1.0, current_cost * 0.5)
    Tmin = 0.01
    alpha = 0.997  # cooling rate

    T = T0
    for _ in range(max_iters):
        # generate neighbor
        cand = neighbor_swap(current, rnd)
        cand_cost, _, _ = total_cost(cand, rows, cols, spacing)
        delta = cand_cost - current_cost

        # accept if better or with probability
        if delta <= 0:
            current, current_cost = cand, cand_cost
        else:
            # accept worse move with probability
            if rnd.random() < math.exp(-delta / max(T, 1e-9)):
                current, current_cost = cand, cand_cost

        # track best
        if current_cost < best_cost:
            best, best_cost = current[:], current_cost
            if best_cost == 0:
                break  # perfect arrangement

        # cool
        T = max(Tmin, T * alpha)

    return best, best_cost

def layout_dataframe(state, rows, cols):
    data = []
    for r in range(rows):
        row = []
        for c in range(cols):
            row.append(state[rc_to_index(r, c, cols)])
        data.append(row)
    df = pd.DataFrame(data, columns=[f"C{c+1}" for c in range(cols)])
    df.index = [f"R{r+1}" for r in range(rows)]
    return df

# -----------------------
# UI
# -----------------------
st.title("🎓 Student Seating Assignment — Local Search (Simulated Annealing)")
st.markdown("""
**Goal**: Arrange students so **no two adjacent** seats have the **same department**, and enforce a **minimum spacing** between same departments.  
**Search**: Local search with randomness (simulated annealing style) to reach a **near-optimal** layout.
""")

with st.sidebar:
    st.header("🧩 Problem Setup")
    rows = st.number_input("Rows", min_value=1, max_value=20, value=4)
    cols = st.number_input("Columns", min_value=1, max_value=20, value=6)
    spacing = st.number_input("Spacing constraint (Manhattan distance)", min_value=1, max_value=10, value=2,
                              help="Same-department students should be farther than this Manhattan distance. 1 means only direct adjacency is forbidden in the adjacency term; spacing adds extra penalties for distances 2..k.")
    seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=42, help="For reproducibility.")
    iters = st.number_input("Max iterations (search effort)", min_value=100, max_value=100_000, value=6000, step=500)

    st.markdown("**Department counts (Dept:Count)**")
    dept_text = st.text_area(
        "Example: CSE:10, ECE:8, MECH:6",
        value="CSE:10, ECE:8, MECH:6"
    )
    st.caption("Total seats must equal Rows × Columns.")

# Validate inputs & generate instance
total_seats = rows * cols
error = None
dept_list = []
try:
    dept_list = parse_dept_counts(dept_text)
    if len(dept_list) != total_seats:
        error = f"Seat count mismatch: got {len(dept_list)} students for {total_seats} seats."
except Exception as e:
    error = str(e)

if error:
    st.error(error)
    st.stop()

# Initial & Optimize buttons
c1, c2 = st.columns(2)
with c1:
    if st.button("Generate Initial Layout"):
        st.session_state["initial_state"] = make_initial_state(dept_list, seed=seed)
with c2:
    if st.button("Optimize (Simulated Annealing)"):
        # Ensure initial state exists
        if "initial_state" not in st.session_state:
            st.session_state["initial_state"] = make_initial_state(dept_list, seed=seed)
        best, best_cost = simulated_annealing_optimize(
            st.session_state["initial_state"], rows, cols, spacing, max_iters=iters, seed=seed
        )
        st.session_state["final_state"] = best
        st.session_state["final_cost"] = best_cost

# Show initial layout
st.subheader("Initial Layout")
if "initial_state" not in st.session_state:
    st.info("Click **Generate Initial Layout** to start.")
else:
    init_state = st.session_state["initial_state"]
    icost, iadj, ispc = total_cost(init_state, rows, cols, spacing)
    st.write(f"**Initial cost** = {icost:.2f}  |  adjacency conflicts = {iadj}  |  spacing conflicts = {ispc}")
    st.dataframe(layout_dataframe(init_state, rows, cols), use_container_width=True)

# Show optimized layout
st.subheader("Optimized Layout (Near-Optimal)")
if "final_state" in st.session_state:
    final_state = st.session_state["final_state"]
    fcost, fadj, fspc = total_cost(final_state, rows, cols, spacing)
    st.write(f"**Final cost** = {fcost:.2f}  |  adjacency conflicts = {fadj}  |  spacing conflicts = {fspc}")
    st.dataframe(layout_dataframe(final_state, rows, cols), use_container_width=True)

    if fcost == 0:
        st.success("Perfect arrangement found! 🎉")
        st.balloons()
    else:
        st.info("Near-optimal arrangement found. You can try a different seed or higher iterations for potentially better layouts.")
else:
    st.info("Click **Optimize (Simulated Annealing)** to improve the layout.")

# Footer / Notes
st.markdown("---")
st.caption(
    "Notes: Adjacent conflicts use 4-neighborhood (up, down, left, right). "
    "Spacing conflicts penalize same-department pairs within Manhattan distance ≤ spacing (excluding direct adjacency which is already counted). "
    "Local search uses a swap neighbor and a fixed internal schedule."
)
