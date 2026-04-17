# manufacturing_hillclimb.py
import streamlit as st
import pandas as pd
import random

st.set_page_config(layout="wide", page_title="Manufacturing Task Sequencing — Hill Climbing")

# -----------------------
# Hill Climbing Functions
# -----------------------
def total_time(sequence, setup_times):
    return sum(setup_times[sequence[i]][sequence[i+1]] for i in range(len(sequence)-1))

def hill_climb(tasks, setup_times):
    current_seq = tasks[:]
    random.shuffle(current_seq)
    current_cost = total_time(current_seq, setup_times)
    improved = True
    steps = [(current_seq[:], current_cost)]

    while improved:
        improved = False
        for i in range(len(tasks)):
            for j in range(i+1, len(tasks)):
                neighbor = current_seq[:]
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                cost = total_time(neighbor, setup_times)
                if cost < current_cost:
                    current_seq, current_cost = neighbor, cost
                    steps.append((current_seq[:], current_cost))
                    improved = True
    return current_seq, current_cost, steps

# -----------------------
# Streamlit UI
# -----------------------
st.title("🏭 Manufacturing Task Sequencing — Hill Climbing")

with st.sidebar:
    st.header("Scenario Controls")
    num_tasks = st.number_input("Number of tasks", min_value=3, max_value=8, value=4)
    st.write("Enter setup times between tasks (rows → from task, cols → to task):")
    
    # Default matrix
    default_matrix = [[0 if i == j else random.randint(2, 6) for j in range(num_tasks)] for i in range(num_tasks)]
    df_input = pd.DataFrame(default_matrix, columns=[f"T{j}" for j in range(num_tasks)], index=[f"T{i}" for i in range(num_tasks)])
    setup_df = st.data_editor(df_input, num_rows="dynamic")

# Convert DF to list of lists
setup_times = setup_df.values.tolist()
tasks = list(range(num_tasks))

# Run Hill Climbing
best_seq, best_cost, steps = hill_climb(tasks, setup_times)

# -----------------------
# Output
# -----------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Initial Sequence")
    init_seq, init_cost = steps[0]
    st.write(f"Sequence: {['T'+str(t) for t in init_seq]}")
    st.write(f"Total setup time: **{init_cost}**")

with col2:
    st.subheader("Optimized Sequence (Hill Climbing)")
    st.write(f"Sequence: {['T'+str(t) for t in best_seq]}")
    st.write(f"Total setup time: **{best_cost}**")

# -----------------------
# Comparison Table
# -----------------------
st.subheader("Comparison Table")
comp_data = {
    "Step": list(range(1, len(steps)+1)),
    "Sequence": [[f"T{t}" for t in seq] for seq, _ in steps],
    "Cost": [cost for _, cost in steps]
}
st.dataframe(pd.DataFrame(comp_data))

st.markdown("---")
st.caption("Note: Hill Climbing searches for local optima by swapping two tasks at a time.")
