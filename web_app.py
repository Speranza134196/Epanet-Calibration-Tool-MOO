import streamlit as st
import pandas as pd
import numpy as np
import wntr
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from io import BytesIO

st.set_page_config(page_title="EPANET Calibration Tool", layout="wide")
st.title("💧 EPANET Calibration Tool with MOO")

# Constants
discrete_settings = [0, 5, 10, 20, 40, 60, 100, 200, 300, 500, 1000]

@st.cache_data
def load_observed_data(file):
    return pd.read_csv(file, index_col=0)

@st.cache_resource
def load_network(file_path):
    return wntr.network.WaterNetworkModel(file_path)

# Uploads
col1, col2 = st.columns(2)
with col1:
    inp_file = st.file_uploader("Upload EPANET .inp file", type=["inp"])
with col2:
    obs_file = st.file_uploader("Upload Observed Data (CSV)", type=["csv"])

# Calibration function
def compute_rmse_split(obs_data, results):
    sim_pressures = results.node.get('pressure', pd.DataFrame())
    sim_flows = results.link.get('flowrate', pd.DataFrame()) * 1000
    rmse_pressures, rmse_flows = [], []
    for col in obs_data.columns:
        if col in sim_pressures.columns:
            aligned = sim_pressures[col].reindex(obs_data.index).interpolate()
            rmse = np.sqrt(np.nanmean((obs_data[col] - aligned)**2))
            rmse_pressures.append(rmse)
        elif col in sim_flows.columns:
            aligned = sim_flows[col].reindex(obs_data.index).interpolate()
            rmse = np.sqrt(np.nanmean((obs_data[col] - aligned)**2))
            rmse_flows.append(rmse)
    return np.nanmean(rmse_pressures), np.nanmean(rmse_flows)

if inp_file and obs_file:
    temp_path = "temp_model.inp"
    with open(temp_path, "wb") as f:
        f.write(inp_file.getvalue())

    wn = load_network(temp_path)
    obs_data = load_observed_data(obs_file)
    
    pipes = [link for link_name, link in wn.links() if link.link_type == 'Pipe']
    valves = [link for link_name, link in wn.links() if link.link_type == 'Valve']
    tcvs = [valve for valve in valves if valve.valve_type.upper() == 'TCV']

    if st.button("🔄 Run Calibration"):
        with st.spinner('Calibrating, please wait...'):

            class CalibrationProblem(Problem):
                def __init__(self):
                    xl = np.array([50]*len(pipes) + [0]*len(tcvs))
                    xu = np.array([150]*len(pipes) + [len(discrete_settings)-1]*len(tcvs))
                    super().__init__(n_var=len(pipes)+len(tcvs), n_obj=2, n_constr=0, xl=xl, xu=xu)

                def _evaluate(self, X, out, *args, **kwargs):
                    f1, f2 = [], []
                    for row in X:
                        wn_temp = wntr.network.WaterNetworkModel(temp_path)
                        for i, pipe in enumerate(pipes):
                            wn_temp.get_link(pipe.name).roughness = round(row[i])
                        for j, tcv in enumerate(tcvs):
                            idx = int(round(row[len(pipes)+j]))
                            wn_temp.get_link(tcv.name).initial_setting = discrete_settings[idx]
                        try:
                            sim = wntr.sim.EpanetSimulator(wn_temp)
                            results = sim.run_sim()
                            rmse_p, rmse_f = compute_rmse_split(obs_data, results)
                            f1.append(rmse_p)
                            f2.append(rmse_f)
                        except:
                            f1.append(1e6)
                            f2.append(1e6)
                    out["F"] = np.column_stack([f1, f2])

            problem = CalibrationProblem()
            algorithm = NSGA2(
                pop_size=30,
                sampling=IntegerRandomSampling(),
                crossover=SBX(prob=0.9, eta=15),
                mutation=PM(eta=20),
                eliminate_duplicates=True
            )
            res = minimize(problem, algorithm, ('n_gen', 30), verbose=False)

            # Select Best Solution via Euclidean Distance
            distances = np.sqrt(res.F[:,0]**2 + res.F[:,1]**2)
            idx_best = np.argmin(distances)
            best_solution = res.X[idx_best]
            best_objectives = res.F[idx_best]

            st.success("Calibration Completed!")
            st.write(f"**Best Pressure RMSE:** {best_objectives[0]:.4f}")
            st.write(f"**Best Flow RMSE:** {best_objectives[1]:.4f}")
            st.write(f"**Distance to Ideal:** {distances[idx_best]:.4f}")

            # Update Network
            for i, pipe in enumerate(pipes):
                wn.get_link(pipe.name).roughness = round(best_solution[i])
            for j, tcv in enumerate(tcvs):
                wn.get_link(tcv.name).initial_setting = discrete_settings[int(round(best_solution[len(pipes)+j]))]

            sim_best = wntr.sim.EpanetSimulator(wn)
            results_best = sim_best.run_sim()

            # Plot Pareto Front
            fig, ax = plt.subplots(figsize=(8,6))
            ax.scatter(res.F[:,0], res.F[:,1], c='red', edgecolors='k', label='Pareto Front')
            ax.scatter(res.F[idx_best,0], res.F[idx_best,1], c='blue', edgecolors='k', s=100, label='Best Solution')
            ax.set_xlabel("Pressure RMSE")
            ax.set_ylabel("Flow RMSE")
            ax.set_title("Pareto Front with Selected Best Solution")
            ax.legend()
            st.pyplot(fig)

            # Download calibrated data
            calibrated_data = []
            for pipe in pipes:
                calibrated_data.append([pipe.name, "Pipe", round(wn.get_link(pipe.name).roughness, 2)])
            for tcv in tcvs:
                calibrated_data.append([tcv.name, "TCV", wn.get_link(tcv.name).initial_setting])

            calibrated_df = pd.DataFrame(calibrated_data, columns=["Component", "Type", "Calibrated Value"])

            st.download_button(
                label="💾 Download Calibrated Values",
                data=calibrated_df.to_csv(index=False),
                file_name="calibrated_values.csv",
                mime="text/csv"
            )
