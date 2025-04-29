import streamlit as st
import pandas as pd
import numpy as np
import wntr
import matplotlib.pyplot as plt
from io import BytesIO
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

st.set_page_config(page_title="EPANET Calibration", layout="wide")
st.title("💧 EPANET Calibration Tool")

# --- Upload Files ---
inp_file = st.file_uploader("Upload EPANET .inp file", type=["inp"])
obs_file = st.file_uploader("Upload Observed Data (CSV)", type=["csv"])

# --- Compute RMSE Split ---
def compute_rmse_split(obs_data, results):
    sim_pressures = results.node.get('pressure', pd.DataFrame())
    sim_flows = results.link.get('flowrate', pd.DataFrame()) * 1000
    rmse_pressures, rmse_flows = [], []
    for col in obs_data.columns:
        if col in sim_pressures.columns:
            sim_series = sim_pressures[col]
            aligned = sim_series.reindex(obs_data.index).interpolate()
            diff = obs_data[col] - aligned
            rmse = np.sqrt(np.nanmean(diff**2)) if not diff.dropna().empty else np.nan
            rmse_pressures.append(rmse)
        elif col in sim_flows.columns:
            sim_series = sim_flows[col]
            aligned = sim_series.reindex(obs_data.index).interpolate()
            diff = obs_data[col] - aligned
            rmse = np.sqrt(np.nanmean(diff**2)) if not diff.dropna().empty else np.nan
            rmse_flows.append(rmse)
    avg_pressure_rmse = np.nanmean(rmse_pressures) if rmse_pressures else 0
    avg_flow_rmse = np.nanmean(rmse_flows) if rmse_flows else 0
    return avg_pressure_rmse, avg_flow_rmse

# --- Time Series Plot ---
def plot_results(obs_data, results):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for col in obs_data.columns:
        if col in results.node.get('pressure', pd.DataFrame()).columns:
            sim_series = results.node['pressure'][col]
            aligned = sim_series.reindex(obs_data.index).interpolate()
            ax1.plot(obs_data.index, obs_data[col], label=f"Obs {col}")
            ax1.plot(obs_data.index, aligned, label=f"Sim {col}")
        elif col in results.link.get('flowrate', pd.DataFrame()).columns:
            sim_series = results.link['flowrate'][col] * 1000
            aligned = sim_series.reindex(obs_data.index).interpolate()
            ax2.plot(obs_data.index, obs_data[col], label=f"Obs {col}")
            ax2.plot(obs_data.index, aligned, label=f"Sim {col}")
    ax1.set_title("Observed vs Simulated Pressure")
    ax1.set_ylabel("Pressure (m)")
    ax1.legend()
    ax2.set_title("Observed vs Simulated Flow")
    ax2.set_ylabel("Flow (L/s)")
    ax2.set_xlabel("Time")
    ax2.legend()
    st.pyplot(fig)

# --- Roughness Categories Plot ---
def plot_network_roughness_categories(wn, pipes):
    fig, ax = plt.subplots(figsize=(12, 8))
    node_pos = wn.query_node_attribute('coordinates')
    count_low, count_mid, count_high = 0, 0, 0
    for pipe in pipes:
        link = wn.get_link(pipe.name)
        n1 = node_pos[link.start_node_name]
        n2 = node_pos[link.end_node_name]
        r = link.roughness
        if r < 90:
            ax.plot([n1[0], n2[0]], [n1[1], n2[1]], color='blue', linewidth=2)
            count_low += 1
        elif 90 <= r <= 100:
            ax.plot([n1[0], n2[0]], [n1[1], n2[1]], color='green', linewidth=2)
            count_mid += 1
        else:
            ax.plot([n1[0], n2[0]], [n1[1], n2[1]], color='red', linewidth=2)
            count_high += 1
    ax.set_title('Network Roughness Categories')
    ax.axis('equal')
    ax.axis('off')
    ax.legend(handles=[
        plt.Line2D([0], [0], color='blue', lw=2, label=f'< 90 (Blue): {count_low} pipes'),
        plt.Line2D([0], [0], color='green', lw=2, label=f'90–100 (Green): {count_mid} pipes'),
        plt.Line2D([0], [0], color='red', lw=2, label=f'> 100 (Red): {count_high} pipes')
    ])
    st.pyplot(fig)

# --- Run Calibration ---
if inp_file and obs_file:
    temp_path = "temp_model.inp"
    with open(temp_path, "wb") as f:
        f.write(inp_file.getvalue())
    obs_data = pd.read_csv(obs_file, index_col=0)
    wn = wntr.network.WaterNetworkModel(temp_path)

    # Detect Pipes and TCVs
    pipes = [link for link_name, link in wn.links() if link.link_type == 'Pipe']
    valves = [link for link_name, link in wn.links() if link.link_type == 'Valve']
    tcvs = [valve for valve in valves if valve.valve_type.upper() == 'TCV']

    if st.button("Run Calibration with Persistency"):
        discrete_settings = [0, 5, 10, 20, 40, 60, 100, 200, 300, 500, 1000]

        class SimpleIntegerCalibration(Problem):
            def __init__(self):
                xl = np.array([50]*len(pipes) + [0]*len(tcvs))
                xu = np.array([150]*len(pipes) + [len(discrete_settings)-1]*len(tcvs))
                super().__init__(n_var=len(pipes) + len(tcvs), n_obj=2, n_constr=0, xl=xl, xu=xu)

            def _evaluate(self, X, out, *args, **kwargs):
                f1, f2 = [], []
                for row in X:
                    wn_temp = wntr.network.WaterNetworkModel(temp_path)
                    for i, pipe in enumerate(pipes):
                        wn_temp.get_link(pipe.name).roughness = round(row[i])
                    for j, tcv in enumerate(tcvs):
                        setting_index = int(round(row[len(pipes) + j]))
                        wn_temp.get_link(tcv.name).initial_setting = discrete_settings[setting_index]
                    try:
                        sim = wntr.sim.EpanetSimulator(wn_temp)
                        results = sim.run_sim()
                        rmse_pressure, rmse_flow = compute_rmse_split(obs_data, results)
                        f1.append(rmse_pressure)
                        f2.append(rmse_flow)
                    except:
                        f1.append(1e6)
                        f2.append(1e6)
                out["F"] = np.column_stack([f1, f2])

        problem = SimpleIntegerCalibration()
        algorithm = NSGA2(pop_size=20, sampling=IntegerRandomSampling(), crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20), eliminate_duplicates=True)
        res = minimize(problem, algorithm, ('n_gen', 50), verbose=False)

        # Select based on combined (sum) of Pressure RMSE + Flow RMSE
  
        # --- Corrected Best Solution Using Euclidean Distance ---
        distances = np.sqrt(res.F[:, 0]**2 + res.F[:, 1]**2)
        idx_best = np.argmin(distances)
        best_solution = res.X[idx_best]
        best_objectives = res.F[idx_best]

        # Display best solution RMSEs and Distance
        st.write("Selected Best Solution:")
        st.write(f"- Pressure RMSE: {best_objectives[0]:.4f}")
        st.write(f"- Flow RMSE: {best_objectives[1]:.4f}")
        st.write(f"- Distance to Origin: {distances[idx_best]:.4f}")

        # Optional: Display all distances
        st.subheader("All Pareto Solutions with Distances")
        for i, d in enumerate(distances):
            st.write(f"Solution {i}: Distance = {d:.4f}, Pressure RMSE = {res.F[i, 0]:.4f}, Flow RMSE = {res.F[i, 1]:.4f}")

        # Apply best_solution to the network
        for i, pipe in enumerate(pipes):
            wn.get_link(pipe.name).roughness = round(best_solution[i])
        for j, tcv in enumerate(tcvs):
            wn.get_link(tcv.name).initial_setting = discrete_settings[int(round(best_solution[len(pipes) + j]))]

        sim_best = wntr.sim.EpanetSimulator(wn)
        results_best = sim_best.run_sim()

        calibrated_data = []
        for pipe in pipes:
            calibrated_data.append([pipe.name, "Pipe", round(wn.get_link(pipe.name).roughness, 2)])
        for tcv in tcvs:
            calibrated_data.append([tcv.name, "TCV", wn.get_link(tcv.name).initial_setting])

        calibrated_df = pd.DataFrame(calibrated_data, columns=["Component", "Type", "Calibrated Value"])

        # Persistently store all outputs
        st.session_state['pareto_pressure'] = res.F[:, 0]
        st.session_state['pareto_flow'] = res.F[:, 1]
        st.session_state['best_results'] = results_best
        st.session_state['calibrated_df'] = calibrated_df
        st.session_state['obs_data'] = obs_data
        st.session_state['pipes'] = pipes
        st.session_state['wn'] = wn

# --- Persisted Outputs ---
if 'best_results' in st.session_state:
    st.subheader("📊 Measured vs Simulated Time Series")
    plot_results(st.session_state['obs_data'], st.session_state['best_results'])

    st.download_button("📥 Download Calibrated Values CSV",
        st.session_state['calibrated_df'].to_csv(index=False),
        file_name="calibrated_values_with_TCVs.csv")

    st.subheader("🌈 Network Roughness Categories Visualization")
    plot_network_roughness_categories(st.session_state['wn'], st.session_state['pipes'])

# --- Persisted Pareto Front with Highlight ---
# --- Corrected Pareto Front Plot with Best Highlighted ---
if 'pareto_pressure' in st.session_state and 'pareto_flow' in st.session_state:
    st.subheader("📊 Corrected Pareto Front with Best Solution Highlighted")

    combined_rmse = np.array(st.session_state['pareto_pressure']) + np.array(st.session_state['pareto_flow'])
    idx_best = np.argmin(combined_rmse)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(res.F[:, 0], res.F[:, 1], c='red', edgecolors='k', label="Pareto Front")
    ax.scatter(res.F[idx_best, 0], res.F[idx_best, 1], c='blue', edgecolors='k', s=120, label="Selected Best", marker='o')
               
    ax.set_xlabel("Pressure RMSE")
    ax.set_ylabel("Flow RMSE")
    ax.set_title("Pareto Front with Selected Best Solution (Corrected)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# --- STEP 2: Update INP File ---
st.subheader("Step 2: Update INP File with Calibrated Roughness & TCV Settings")
roughness_csv = st.file_uploader("Upload Calibrated CSV (Component, Type, Calibrated Value)", type=["csv"], key="update_csv")
inp_to_update = st.file_uploader("Upload INP File to Update", type=["inp"], key="update_inp")

if roughness_csv and inp_to_update:
    rough_df = pd.read_csv(roughness_csv)
    rough_dict = dict(zip(rough_df['Component'], rough_df['Calibrated Value']))

    inp_lines = inp_to_update.getvalue().decode("utf-8").splitlines()
    pipe_start, pipe_end = None, None
    valve_start, valve_end = None, None

    for i, line in enumerate(inp_lines):
        if line.strip().upper() == "[PIPES]":
            pipe_start = i
        elif pipe_start and line.strip().startswith("[") and line.strip().endswith("]"):
            pipe_end = i
            break
    for i, line in enumerate(inp_lines):
        if line.strip().upper() == "[VALVES]":
            valve_start = i
        elif valve_start and line.strip().startswith("[") and line.strip().endswith("]"):
            valve_end = i
            break

    # --- Update Pipes ---
    for i in range(pipe_start + 2, pipe_end):
        parts = inp_lines[i].split()
        if not parts or parts[0].startswith(";"):
            continue
        comp_id = parts[0]
        if comp_id in rough_dict:
            parts[5] = f"{rough_dict[comp_id]:.6f}"
            inp_lines[i] = "\t".join(parts)

    # --- Update TCVs ---
    if valve_start and valve_end:
        for i in range(valve_start + 2, valve_end):
            parts = inp_lines[i].split()
            if not parts or parts[0].startswith(";"):
                continue
            comp_id = parts[0]
            if comp_id in rough_dict:
                parts[5] = f"{rough_dict[comp_id]:.6f}"
                inp_lines[i] = "\t".join(parts)

    updated_content = "\n".join(inp_lines)
    st.download_button(
        label="📥 Download Updated INP File",
        data=BytesIO(updated_content.encode("utf-8")),
        file_name="updated_model.inp",
        mime="text/plain"
    )

