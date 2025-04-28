# 💧 EPANET Calibration Tool (Pipes + TCVs)

This tool provides an intuitive interface for calibrating **pipe roughness** and **TCV (Throttle Control Valve) settings** in a water distribution network modeled with **EPANET**.

---

## 🚀 Features
- Upload your **EPANET INP** model.
- Upload **observed pressure and flow data**.
- Calibrate:
  - **Pipe roughness coefficients**.
  - **TCV initial settings**.
- Multi-objective optimization (minimize **pressure RMSE** and **flow RMSE**).
- Visualize:
  - **Pareto front** of optimal solutions.
  - **Observed vs Simulated** time series.
  - Network with **roughness categories** color-coded.
- Export:
  - Calibrated parameters to **CSV**.
  - Update your **INP** file with calibrated values.

---

## 📂 How to Use

### 1️⃣ Upload Files
- `.inp`: Your EPANET model.
- `.csv`: Observed data (time series of pressure/flow).

### 2️⃣ Run Calibration
- Click **"Run Calibration"**.
- The tool uses **NSGA-II** to explore trade-offs between pressure and flow errors.
- Outputs:
  - **Pareto front plot**.
  - **Best solution** highlighted.
  - Calibrated **CSV download**.

### 3️⃣ INP File Update
- Upload the calibrated **CSV** and the original **INP**.
- Download the **updated INP file** with:
  - New **pipe roughness**.
  - Updated **TCV initial settings**.

---

## 📊 Visual Outputs
- **Pareto Front**: Displays optimal trade-offs between pressure and flow RMSE.
- **Best Solution Highlighted**: The point with the lowest combined RMSE.
- **Network Roughness Visualization**:
  - Pipes color-coded:
    - Blue: Roughness < 90
    - Green: 90 ≤ Roughness ≤ 100
    - Red: Roughness > 100

---

## 🧰 Dependencies
- Python 3.8+
- `streamlit`
- `pandas`
- `numpy`
- `wntr`
- `matplotlib`
- `pymoo`

Install via:
```bash
pip install streamlit pandas numpy wntr matplotlib pymoo
```

---

## ⚙️ Customization Options
- Adjust **objective weights** for RMSE.
- Customize **roughness categories** or **valve settings**.
- Extend to other **valve types** (PRV, FCV, etc.).

---

## 👨‍💻 Author
Developed with ❤️ by Francesco De Paola and enhanced with AI guidance.

