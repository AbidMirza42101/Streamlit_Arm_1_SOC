# app.py
"""
Streamlit UI for Arm 1 (SOC) ONLY.

Inputs:
- rS, rR, killS, days
- visit_every, noise_sigma
- toggle: show observed dots vs true line

Run locally:
  streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from model import simulate_soc


st.set_page_config(page_title="Arm 1 (SOC) Simulator", layout="wide")

st.title("Arm 1 (SOC) — Continuous Tamoxifen, No Fasting")
st.write(
    "This app simulates a two-population tumor model: **Sensitive (S)** and **Resistant (R)**.\n\n"
    "- **Tamoxifen is always ON** (u1=1)\n"
    "- **Fasting/Keto is always OFF** (u2=0)\n\n"
    "You can change growth/kill parameters and re-run the simulation."
)

# -------------------------------
# Sidebar inputs
# -------------------------------
st.sidebar.header("Inputs")

days = st.sidebar.number_input("Number of days (t_end)", min_value=30, max_value=5000, value=400, step=10)

rS = st.sidebar.number_input("rS (Sensitive growth per day)", min_value=0.0, max_value=1.0, value=0.035, step=0.001, format="%.3f")
rR = st.sidebar.number_input("rR (Resistant growth per day)", min_value=0.0, max_value=1.0, value=0.025, step=0.001, format="%.3f")

killS = st.sidebar.number_input("killS (Tamoxifen extra kill on S per day)", min_value=0.0, max_value=1.0, value=0.06, step=0.001, format="%.3f")

visit_every = st.sidebar.number_input("visit_every (days between scans)", min_value=1, max_value=365, value=14, step=1)
noise_sigma = st.sidebar.number_input("noise_sigma (scan noise; 0.05 ≈ 5%)", min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.2f")

show_obs = st.sidebar.toggle("Show observed dots vs true line", value=True)

seed = st.sidebar.number_input("Random seed", min_value=0, max_value=999999, value=7, step=1)

st.sidebar.header("Initial tumor (cells)")
S0 = st.sidebar.number_input("Initial S", min_value=0.0, value=5e8, step=1e7, format="%.0f")
R0 = st.sidebar.number_input("Initial R", min_value=0.0, value=1e6, step=1e5, format="%.0f")

# -------------------------------
# Fixed params (keep same as your baseline)
# -------------------------------
params = {
    "rS": float(rS),
    "rR": float(rR),
    "K": 2e9,
    "dS0": 0.005,
    "dR0": 0.005,
    "killS": float(killS),
    "killR": 0.0,
    "mu": 1e-6,
    "keto_eff_S": 0.10,  # unused in SOC, but kept for consistency
    "keto_eff_R": 0.45,  # unused in SOC, but kept for consistency
}

# -------------------------------
# Run simulation
# -------------------------------
run = st.button("Run simulation", type="primary")

if run:
    soc = simulate_soc(
        params=params,
        y0=[float(S0), float(R0)],
        t_end=int(days),
        visit_every=int(visit_every),
        noise_sigma=float(noise_sigma),
        seed=int(seed),
    )

    t = soc["t"]
    V_true = soc["V_true"]
    V_obs = soc["V_obs"]
    R = soc["R"]
    R_frac = R / np.maximum(V_true, 1e-12)

    # -------------------------------
    # Layout
    # -------------------------------
    col1, col2 = st.columns(2)

    # Plot 1: Total tumor burden
    fig1 = plt.figure()
    plt.plot(t, V_true, label="SOC: Tamoxifen ON (u1=1), Fasting OFF (u2=0)")
    plt.xlabel("Time (days)")
    plt.ylabel("Total tumor (S+R)")
    plt.title("Tumor Burden Over Time")
    plt.legend()
    col1.pyplot(fig1)

    # Plot 2: Resistant fraction
    fig2 = plt.figure()
    plt.plot(t, R_frac)
    plt.xlabel("Time (days)")
    plt.ylabel("Resistant fraction R/(S+R)")
    plt.title("Resistance Fraction Over Time")
    col2.pyplot(fig2)

    # Plot 3: True vs observed (optional)
    st.subheader("True vs Observed Tumor Burden")
    fig3 = plt.figure()
    plt.plot(t, V_true, label="True tumor (hidden)")
    if show_obs:
        plt.scatter(t, V_obs, s=10, alpha=0.6, label="Measured tumor (noisy scans)")
    plt.xlabel("Time (days)")
    plt.ylabel("Tumor (S+R)")
    plt.title("Observed Dots vs True Line")
    plt.legend()
    st.pyplot(fig3)

    # -------------------------------
    # Quick stats
    # -------------------------------
    st.subheader("Quick summary")
    st.write(
        {
            "Baseline (first noisy scan)": float(soc["baseline"]),
            "Final true tumor": float(V_true[-1]),
            "Final resistant fraction": float(R_frac[-1]),
            "Parameters used": {
                "rS": float(rS),
                "rR": float(rR),
                "killS": float(killS),
                "days": int(days),
                "visit_every": int(visit_every),
                "noise_sigma": float(noise_sigma),
                "seed": int(seed),
            },
        }
    )
else:
    st.info("Set parameters on the left and click **Run simulation**.")
