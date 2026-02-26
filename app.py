import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from model import simulate_model

st.set_page_config(layout="wide")
st.title("Tamoxifen Model — Arm 1 & Arm 2")

st.sidebar.header("Simulation Settings")

days = st.sidebar.number_input("Days", 100, 2000, 400)
rS = st.sidebar.number_input("rS (Sensitive growth)", value=0.035, format="%.4f")
rR = st.sidebar.number_input("rR (Resistant growth)", value=0.025, format="%.4f")
killS = st.sidebar.number_input("killS (Tamoxifen kill)", value=0.06, format="%.4f")
visit_every = st.sidebar.number_input("Visit interval (days)", 1, 60, 14)
noise_sigma = st.sidebar.number_input("Noise sigma", value=0.05, format="%.3f")
seed = st.sidebar.number_input("Random seed", value=7)

adaptive_toggle = st.sidebar.toggle("Activate Adaptive Therapy")

if adaptive_toggle:
    stop_frac = st.sidebar.slider("Stop threshold (%)", 0.1, 0.9, 0.5)
    restart_frac = st.sidebar.slider("Restart threshold (%)", 0.5, 1.5, 1.0)
else:
    stop_frac = 0.5
    restart_frac = 1.0

params = {
    "rS": rS,
    "rR": rR,
    "K": 2e9,
    "dS0": 0.005,
    "dR0": 0.005,
    "killS": killS,
    "mu": 1e-6,
}

y0 = [5e8, 1e6]

result = simulate_model(
    params,
    y0,
    t_end=days,
    visit_every=visit_every,
    noise_sigma=noise_sigma,
    seed=seed,
    adaptive=adaptive_toggle,
    stop_frac=stop_frac,
    restart_frac=restart_frac,
)

t = result["t"]
V = result["V_true"]
R_frac = result["R"] / np.maximum(V, 1e-12)
u1 = result["u1"]

col1, col2 = st.columns(2)

with col1:
    fig1 = plt.figure()
    plt.plot(t, V)
    plt.xlabel("Time (days)")
    plt.ylabel("Total Tumor")
    plt.title("Tumor Burden")
    st.pyplot(fig1)

with col2:
    fig2 = plt.figure()
    plt.plot(t, R_frac)
    plt.xlabel("Time (days)")
    plt.ylabel("Resistant Fraction")
    plt.title("Resistance Dynamics")
    st.pyplot(fig2)

st.subheader("Drug Schedule (u1)")
fig3 = plt.figure()
plt.step(t, u1, where="post")
plt.ylim(-0.1, 1.1)
plt.xlabel("Time (days)")
plt.ylabel("Drug ON (1) / OFF (0)")
st.pyplot(fig3)
