import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from inflation_model import (
    nominal_value, real_value_constant, real_value_time_varying,
    make_inflation_scenario
)
from ml_forecast import train_and_forecast

st.set_page_config(page_title="Inflation-Aware Investment Model", page_icon="📈", layout="wide")
st.title("Inflation-Aware Investment Growth")
st.caption("Differential equations + a machine-learning forecasting demonstration")

st.info(
    "The ODE simulator is the mathematical core from the project PPT. "
    "The ML tab uses synthetic classroom data—not official inflation data—so its "
    "forecast demonstrates the workflow rather than making real-world predictions."
)

with st.sidebar:
    st.header("Investment assumptions")
    initial = st.number_input("Initial investment", min_value=1.0, value=1000.0, step=100.0)
    annual_return_pct = st.slider("Nominal annual return (%)", 0.0, 20.0, 8.0, 0.5)
    years_horizon = st.slider("Time horizon (years)", 1, 40, 25)
    scenario = st.selectbox("Inflation scenario", ["Constant", "Cyclical", "Piecewise"])
    base_inflation_pct = st.slider("Baseline inflation (%)", 0.0, 15.0, 5.0, 0.5)
    if scenario == "Cyclical":
        amplitude_pct = st.slider("Inflation cycle amplitude (%)", 0.0, 5.0, 2.0, 0.5)
        period = st.slider("Cycle period (years)", 1.0, 15.0, 5.0, 0.5)
    else:
        amplitude_pct, period = 2.0, 5.0
    if scenario == "Piecewise":
        shock_year = st.slider("Regime change year", 1, years_horizon, min(10, years_horizon))
        post_shock_pct = st.slider("Inflation after regime change (%)", 0.0, 20.0, 7.0, 0.5)
    else:
        shock_year, post_shock_pct = min(10, years_horizon), 7.0

r = annual_return_pct / 100.0
base_i = base_inflation_pct / 100.0
t = np.linspace(0, years_horizon, years_horizon * 12 + 1)
inflation = make_inflation_scenario(
    scenario, t, base_i, amplitude_pct / 100.0, period,
    float(shock_year), post_shock_pct / 100.0
)
nominal = nominal_value(initial, r, t)
real = real_value_time_varying(initial, r, t, inflation)
end_real = float(real[-1])
end_nominal = float(nominal[-1])

c1, c2, c3 = st.columns(3)
c1.metric("Nominal balance", f"${end_nominal:,.2f}")
c2.metric("Purchasing power (today's dollars)", f"${end_real:,.2f}")
c3.metric("Real purchasing-power change", f"{(end_real / initial - 1) * 100:.1f}%")

tab1, tab2, tab3 = st.tabs(["Investment simulator", "Inflation ML demo", "Model explanation"])

with tab1:
    st.subheader("Nominal value vs. inflation-adjusted value")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(t, nominal, label="Nominal balance", linewidth=2.5)
    ax.plot(t, real, label="Real purchasing power", linewidth=2.5)
    ax.set_xlabel("Years")
    ax.set_ylabel("Value (currency units)")
    ax.grid(alpha=0.25)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 3.5))
    ax2.plot(t, inflation * 100, label="Inflation rate", linewidth=2)
    ax2.axhline(annual_return_pct, linestyle="--", label="Nominal return")
    ax2.set_xlabel("Years")
    ax2.set_ylabel("Annualized rate (%)")
    ax2.grid(alpha=0.25)
    ax2.legend()
    st.pyplot(fig2)
    plt.close(fig2)

    result = pd.DataFrame({
        "year": t,
        "inflation_pct": inflation * 100,
        "nominal_value": nominal,
        "real_value": real
    })
    st.download_button(
        "Download simulation CSV", result.to_csv(index=False).encode("utf-8"),
        file_name="investment_simulation.csv", mime="text/csv"
    )

with tab2:
    st.subheader("Predicting inflation from previous observations")
    st.write(
        "A Random Forest regressor learns relationships between the previous 12 monthly "
        "inflation observations and the next month's value. The data below is synthetic."
    )
    forecast_horizon = st.slider("Months to forecast", 1, 24, 12)
    with st.spinner("Generating data and training the model..."):
        history_df, future_df, metrics = train_and_forecast(forecast_months=forecast_horizon)
    m1, m2 = st.columns(2)
    m1.metric("Holdout MAE", f"{metrics['mae']:.3f} percentage points")
    m2.metric("Holdout RMSE", f"{metrics['rmse']:.3f} percentage points")
    st.caption("Evaluation uses a time-ordered 80/20 split. Metrics are only for this synthetic dataset.")
    fig3, ax3 = plt.subplots(figsize=(10, 4.5))
    ax3.plot(history_df["month"], history_df["inflation_pct"], label="Synthetic historical inflation", linewidth=1.8)
    ax3.plot(future_df["month"], future_df["predicted_inflation_pct"], label="ML forecast", marker="o")
    ax3.axvline(history_df["month"].iloc[-1], linestyle="--", alpha=0.7, label="Forecast starts")
    ax3.set_xlabel("Synthetic month index")
    ax3.set_ylabel("Inflation (%)")
    ax3.grid(alpha=0.25)
    ax3.legend()
    st.pyplot(fig3)
    plt.close(fig3)
    st.dataframe(future_df.rename(columns={
        "month": "Synthetic month",
        "predicted_inflation_pct": "Predicted inflation (%)"
    }), use_container_width=True)

with tab3:
    st.subheader("What the equations mean")
    st.latex(r"\frac{dV_{nom}}{dt}=rV_{nom}\quad\Rightarrow\quad V_{nom}(t)=V_0e^{rt}")
    st.latex(r"\frac{dV_{real}}{dt}=(r-i(t))V_{real}")
    st.latex(r"V_{real}(t)=V_0\exp\left(\int_0^t[r-i(s)]\,ds\right)")
    st.markdown("""
    - **ODE component:** models continuous growth after subtracting inflation from the nominal rate.
    - **ML component:** uses the previous 12 monthly observations as features and predicts the next month's inflation.
    - **Evaluation:** MAE and RMSE compare predictions with held-out synthetic observations.
    - **Integration:** a real deployment could replace synthetic data with a documented, cleaned historical dataset; the forecasted inflation path could then be passed into the ODE simulator.
    - **Model limitation:** the simplified real-return equation uses the Fisher approximation `r - i`, rather than the exact discrete relation `(1+r)/(1+i)-1`.
    """)
