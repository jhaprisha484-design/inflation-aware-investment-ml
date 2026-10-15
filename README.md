# Inflation-Aware Investment Growth
### Differential Equations + Machine Learning

This project extends the mathematical model in the presentation **“Modeling Investment Growth Under Inflation.”** It contains two distinct parts:

1. **ODE simulator:** models nominal investment growth and inflation-adjusted purchasing power under constant, cyclical, and piecewise inflation.
2. **ML demonstration:** trains a Random Forest regressor to forecast the next month's inflation from the previous 12 observations. The included dataset is synthetic and intended for demonstrating the ML workflow.

> **Important:** This is an academic demonstration, not investment advice. The ML demo does not use official historical inflation data, so its metrics and forecasts must not be interpreted as real-world forecasting performance.

## Model

Nominal investment growth:

\[
\frac{dV_{\mathrm{nom}}}{dt}=rV_{\mathrm{nom}}
\quad\Longrightarrow\quad
V_{\mathrm{nom}}(t)=V_0e^{rt}
\]

Inflation-adjusted purchasing power:

\[
\frac{dV_{\mathrm{real}}}{dt}=[r-i(t)]V_{\mathrm{real}}
\]

For time-varying inflation:

\[
V_{\mathrm{real}}(t)=V_0\exp\left(\int_0^t[r-i(s)]\,ds\right)
\]

Here, \(V_0\) is the initial investment, \(r\) is the nominal annual return, and \(i(t)\) is the inflation rate. Rates are represented as decimals in code.

## Features

- Interactive investment assumptions and inflation scenarios.
- Nominal vs. real purchasing-power plots.
- Constant, cyclical, and piecewise inflation.
- Synthetic inflation series and Random Forest forecasting.
- Time-ordered holdout evaluation with MAE and RMSE.
- CSV export of simulation results.

## Run locally

Python 3.10 or newer is recommended.

```bash
# 1. Create and activate a virtual environment (optional but recommended)
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

The app should open in your browser at the local URL printed in the terminal.

## Repository structure

```text
inflation-aware-investment-ml/
├── app.py                 # Streamlit dashboard and user controls
├── inflation_model.py     # ODE solutions and inflation scenarios
├── ml_forecast.py         # Synthetic data, lag features, ML training/evaluation
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

## Code walkthrough

### `inflation_model.py`
- `nominal_value()` implements \(V_0e^{rt}\).
- `real_value_constant()` implements \(V_0e^{(r-i)t}\).
- `real_value_time_varying()` approximates the integral using the trapezoidal rule, then exponentiates the cumulative integral.
- `make_inflation_scenario()` creates constant, sinusoidal, or regime-switching inflation rates.

### `ml_forecast.py`
- `generate_synthetic_inflation()` creates a reproducible demonstration series using a fixed random seed.
- `make_lagged_features()` turns the previous 12 months into input features and the following month into the target.
- The model is evaluated on a later time segment, not a random split, to respect time order.
- MAE and RMSE quantify error in inflation percentage points.
- The model is then refit on all observed synthetic samples and recursively forecasts future months.

### `app.py`
- Reads parameters from Streamlit controls.
- Calculates nominal and real investment values.
- Plots investment paths and the inflation path.
- Displays the ML evaluation metrics and forecast.
- Offers a CSV download for the simulation.

## Suggested presentation flow (5–7 minutes)

1. **Problem (45 sec):** nominal money growth does not guarantee purchasing-power growth.
2. **Mathematical model (1 min):** explain \(dV/dt=[r-i(t)]V\) and the accumulated-inflation solution.
3. **Implementation (1.5 min):** show how Python calculates the curves and compares scenarios.
4. **ML extension (1.5 min):** explain lag features, Random Forest, time-ordered test split, MAE/RMSE.
5. **Live demo (1 min):** change the return/inflation sliders and show how the real value changes.
6. **Limitations (30 sec):** synthetic data, simplified assumptions, no claim of real-world predictive accuracy.

## GitHub publishing

1. Create a new **public** repository on GitHub, for example `inflation-aware-investment-ml`.
2. Extract this project folder.
3. Open a terminal in the project folder and run:

```bash
git init
git add .
git commit -m "Build inflation-aware investment ODE and ML demo"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/inflation-aware-investment-ml.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. Do not commit passwords, API keys, or private data.

## Important interpretation note

The ODE model is the core of the original presentation. The Random Forest is a separate ML extension that demonstrates how a predicted inflation path could be passed to the ODE simulator. In this starter version, the ML forecast is displayed separately and is trained on synthetic data; it is not yet automatically connected to the simulator's inflation curve.
