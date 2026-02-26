# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# from model import simulate_model

# st.set_page_config(layout="wide")
# st.title("Tamoxifen Model — Arm 1 & Arm 2")

# st.sidebar.header("Simulation Settings")

# days = st.sidebar.number_input("Days", 100, 2000, 400)
# rS = st.sidebar.number_input("rS (Sensitive growth)", value=0.035, format="%.4f")
# rR = st.sidebar.number_input("rR (Resistant growth)", value=0.025, format="%.4f")
# killS = st.sidebar.number_input("killS (Tamoxifen kill)", value=0.06, format="%.4f")
# visit_every = st.sidebar.number_input("Visit interval (days)", 1, 60, 14)
# noise_sigma = st.sidebar.number_input("Noise sigma", value=0.05, format="%.3f")
# seed = st.sidebar.number_input("Random seed", value=7)

# adaptive_toggle = st.sidebar.toggle("Activate Adaptive Therapy")

# if adaptive_toggle:
#     stop_frac = st.sidebar.slider("Stop threshold (%)", 0.1, 0.9, 0.5)
#     restart_frac = st.sidebar.slider("Restart threshold (%)", 0.5, 1.5, 1.0)
# else:
#     stop_frac = 0.5
#     restart_frac = 1.0

# params = {
#     "rS": rS,
#     "rR": rR,
#     "K": 2e9,
#     "dS0": 0.005,
#     "dR0": 0.005,
#     "killS": killS,
#     "mu": 1e-6,
# }

# y0 = [5e8, 1e6]

# result = simulate_model(
#     params,
#     y0,
#     t_end=days,
#     visit_every=visit_every,
#     noise_sigma=noise_sigma,
#     seed=seed,
#     adaptive=adaptive_toggle,
#     stop_frac=stop_frac,
#     restart_frac=restart_frac,
# )

# t = result["t"]
# V = result["V_true"]
# R_frac = result["R"] / np.maximum(V, 1e-12)
# u1 = result["u1"]

# col1, col2 = st.columns(2)

# with col1:
#     fig1 = plt.figure()
#     plt.plot(t, V)
#     plt.xlabel("Time (days)")
#     plt.ylabel("Total Tumor")
#     plt.title("Tumor Burden")
#     st.pyplot(fig1)

# with col2:
#     fig2 = plt.figure()
#     plt.plot(t, R_frac)
#     plt.xlabel("Time (days)")
#     plt.ylabel("Resistant Fraction")
#     plt.title("Resistance Dynamics")
#     st.pyplot(fig2)

# st.subheader("Drug Schedule (u1)")
# fig3 = plt.figure()
# plt.step(t, u1, where="post")
# plt.ylim(-0.1, 1.1)
# plt.xlabel("Time (days)")
# plt.ylabel("Drug ON (1) / OFF (0)")
# st.pyplot(fig3)

########################################################################################################################################
##########################################################################################################################################

# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from model import simulate_model

# st.set_page_config(layout="wide")

# st.title("ER+ Metastatic Breast Cancer Adaptive Therapy Simulator")
# st.markdown("""
# Two-population evolutionary model:

# - Sensitive cells (S)
# - Resistant cells (R)

# Compare:
# • Arm 1 — Continuous Tamoxifen (SOC)  
# • Arm 2 — Adaptive Therapy (ON/OFF switching)

# Research simulation only (not medical advice).
# """)

# # -------------------------
# # Sidebar Controls
# # -------------------------
# st.sidebar.header("Initial Tumor (cells)")

# S0 = st.sidebar.number_input("Initial Sensitive (S)", min_value=0.0,
#                              value=5e8, step=1e7, format="%.0f")

# R0 = st.sidebar.number_input("Initial Resistant (R)", min_value=0.0,
#                              value=1e6, step=1e5, format="%.0f")

# st.sidebar.markdown("---")
# st.sidebar.header("Growth & Drug Parameters")

# rS = st.sidebar.number_input("rS – Sensitive Growth Rate", value=0.035, format="%.4f")
# rR = st.sidebar.number_input("rR – Resistant Growth Rate", value=0.025, format="%.4f")
# killS = st.sidebar.number_input("killS – Tamoxifen Kill Rate", value=0.06, format="%.4f")

# st.sidebar.markdown("---")
# st.sidebar.header("Simulation Settings")

# days = st.sidebar.number_input("Simulation Days", 100, 3000, 400)
# visit_every = st.sidebar.number_input("Scan Interval (days)", 1, 60, 14)
# noise_sigma = st.sidebar.number_input("Scan Noise (0.05 ≈ 5%)", value=0.05, format="%.3f")
# seed = st.sidebar.number_input("Random Seed", value=7)

# st.sidebar.markdown("---")
# adaptive_toggle = st.sidebar.toggle("Enable Adaptive Therapy (Arm 2)")

# if adaptive_toggle:
#     stop_frac = st.sidebar.slider("Stop Threshold (% baseline)", 0.1, 0.9, 0.5)
#     restart_frac = st.sidebar.slider("Restart Threshold (% baseline)", 0.5, 1.5, 1.0)
# else:
#     stop_frac = 0.5
#     restart_frac = 1.0

# show_dots = st.sidebar.toggle("Show Observed Scan Dots")

# # -------------------------
# # Run Button
# # -------------------------
# if st.button("Run Simulation"):

#     params = {
#         "rS": rS,
#         "rR": rR,
#         "K": 2e9,
#         "dS0": 0.005,
#         "dR0": 0.005,
#         "killS": killS,
#         "mu": 1e-6,
#     }

#     y0 = [S0, R0]

#     # SOC always runs
#     soc = simulate_model(
#         params,
#         y0,
#         t_end=days,
#         visit_every=visit_every,
#         noise_sigma=noise_sigma,
#         seed=seed,
#         adaptive=False
#     )

#     # Adaptive only if toggled
#     if adaptive_toggle:
#         adapt = simulate_model(
#             params,
#             y0,
#             t_end=days,
#             visit_every=visit_every,
#             noise_sigma=noise_sigma,
#             seed=seed,
#             adaptive=True,
#             stop_frac=stop_frac,
#             restart_frac=restart_frac,
#         )

#     # -------------------------
#     # Plots
#     # -------------------------
#     col1, col2 = st.columns(2)

#     with col1:
#         fig1 = plt.figure()
#         plt.plot(soc["t"], soc["V_true"], label="SOC")
#         if adaptive_toggle:
#             plt.plot(adapt["t"], adapt["V_true"], label="Adaptive")
#         if show_dots:
#             plt.scatter(soc["t"], soc["V_obs"], s=8, alpha=0.4)
#         plt.xlabel("Time (days)")
#         plt.ylabel("Total Tumor (S+R)")
#         plt.title("Tumor Burden Comparison")
#         plt.legend()
#         st.pyplot(fig1)

#     with col2:
#         fig2 = plt.figure()
#         soc_Rfrac = soc["R"] / np.maximum(soc["V_true"], 1e-12)
#         plt.plot(soc["t"], soc_Rfrac, label="SOC")

#         if adaptive_toggle:
#             adapt_Rfrac = adapt["R"] / np.maximum(adapt["V_true"], 1e-12)
#             plt.plot(adapt["t"], adapt_Rfrac, label="Adaptive")

#         plt.xlabel("Time (days)")
#         plt.ylabel("Resistant Fraction")
#         plt.title("Resistance Dynamics")
#         plt.legend()
#         st.pyplot(fig2)

#     # -------------------------
#     # Drug Schedule (Adaptive)
#     # -------------------------
#     if adaptive_toggle:
#         st.subheader("Adaptive Drug Schedule")
#         fig3 = plt.figure()
#         plt.step(adapt["t"], adapt["u1"], where="post")
#         plt.ylim(-0.1, 1.1)
#         plt.xlabel("Time (days)")
#         plt.ylabel("Drug ON (1) / OFF (0)")
#         st.pyplot(fig3)

#     # -------------------------
#     # Summary Metrics
#     # -------------------------
#     st.subheader("Simulation Summary")

#     final_soc = soc["V_true"][-1]
#     final_soc_Rfrac = soc["R"][-1] / final_soc

#     st.write(f"SOC Final Tumor: {final_soc:,.0f} cells")
#     st.write(f"SOC Final Resistant Fraction: {final_soc_Rfrac:.4f}")

#     if adaptive_toggle:
#         final_adapt = adapt["V_true"][-1]
#         final_adapt_Rfrac = adapt["R"][-1] / final_adapt
#         st.write(f"Adaptive Final Tumor: {final_adapt:,.0f} cells")
#         st.write(f"Adaptive Final Resistant Fraction: {final_adapt_Rfrac:.4f}")

#     # -------------------------
#     # Export CSV
#     # -------------------------
#     df = pd.DataFrame({
#         "Day": soc["t"],
#         "SOC_Tumor": soc["V_true"],
#         "SOC_Resistant_Fraction": soc_Rfrac
#     })

#     if adaptive_toggle:
#         df["Adaptive_Tumor"] = adapt["V_true"]
#         df["Adaptive_Resistant_Fraction"] = adapt_Rfrac

#     st.download_button(
#         label="Download Results as CSV",
#         data=df.to_csv(index=False),
#         file_name="simulation_results.csv",
#         mime="text/csv",
#     )

######################################################################################################################################################
#######################################################################################################################################################

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model import simulate_model

st.set_page_config(layout="wide")

st.title("ER+ Breast Cancer Evolution Simulator")

st.markdown("""
Two-population evolutionary model:

- Sensitive cells (S)
- Resistant cells (R)

Compare:
• Arm 1 — Continuous Tamoxifen (SOC)  
• Arm 2 — Adaptive Therapy (ON/OFF switching)

Research simulation only (not medical advice).
""")

# -------------------------
# Sidebar Controls
# -------------------------
st.sidebar.header("Initial Tumor (cells)")

S0 = st.sidebar.number_input(
    "Initial Sensitive (S)",
    min_value=0.0,
    value=5e8,
    step=1e7,
    format="%.0f",
)

R0 = st.sidebar.number_input(
    "Initial Resistant (R)",
    min_value=0.0,
    value=1e6,
    step=1e5,
    format="%.0f",
)

st.sidebar.markdown("---")
st.sidebar.header("Growth & Drug Parameters")

rS = st.sidebar.number_input("rS – Sensitive Growth Rate", value=0.035, format="%.4f")
rR = st.sidebar.number_input("rR – Resistant Growth Rate", value=0.025, format="%.4f")
killS = st.sidebar.number_input("killS – Tamoxifen Kill Rate", value=0.06, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.header("Simulation Settings")

days = st.sidebar.number_input("Simulation Days", 100, 3000, 400)
visit_every = st.sidebar.number_input("Scan Interval (days)", 1, 60, 14)
noise_sigma = st.sidebar.number_input("Scan Noise (0.05 ≈ 5%)", value=0.05, format="%.3f")
seed = st.sidebar.number_input("Random Seed", value=7)

st.sidebar.markdown("---")
adaptive_toggle = st.sidebar.toggle("Enable Adaptive Therapy (Arm 2)")

if adaptive_toggle:
    stop_frac = st.sidebar.slider("Stop Threshold (% baseline)", 0.1, 0.9, 0.5)
    restart_frac = st.sidebar.slider("Restart Threshold (% baseline)", 0.5, 1.5, 1.0)
else:
    stop_frac = 0.5
    restart_frac = 1.0

show_dots = st.sidebar.toggle("Show Observed Scan Dots")

# -------------------------
# Run Simulation
# -------------------------
if st.button("Run Simulation"):

    params = {
        "rS": rS,
        "rR": rR,
        "K": 2e9,
        "dS0": 0.005,
        "dR0": 0.005,
        "killS": killS,
        "mu": 1e-6,
    }

    y0 = [S0, R0]

    # SOC run
    soc = simulate_model(
        params,
        y0,
        t_end=days,
        visit_every=visit_every,
        noise_sigma=noise_sigma,
        seed=seed,
        adaptive=False,
    )

    # Adaptive run (if enabled)
    if adaptive_toggle:
        adapt = simulate_model(
            params,
            y0,
            t_end=days,
            visit_every=visit_every,
            noise_sigma=noise_sigma,
            seed=seed,
            adaptive=True,
            stop_frac=stop_frac,
            restart_frac=restart_frac,
        )

    # -------------------------
    # Plot 1: Tumor Burden
    # -------------------------
    col1, col2 = st.columns(2)

    with col1:
        fig1 = plt.figure()
        plt.plot(soc["t"], soc["V_true"], label="SOC")
        if adaptive_toggle:
            plt.plot(adapt["t"], adapt["V_true"], label="Adaptive")
        plt.xlabel("Time (days)")
        plt.ylabel("Total Tumor (S+R)")
        plt.title("Tumor Burden Comparison")
        plt.legend()
        st.pyplot(fig1)

    # -------------------------
    # Plot 2: Resistant Fraction
    # -------------------------
    with col2:
        fig2 = plt.figure()
        soc_Rfrac = soc["R"] / np.maximum(soc["V_true"], 1e-12)
        plt.plot(soc["t"], soc_Rfrac, label="SOC")

        if adaptive_toggle:
            adapt_Rfrac = adapt["R"] / np.maximum(adapt["V_true"], 1e-12)
            plt.plot(adapt["t"], adapt_Rfrac, label="Adaptive")

        plt.xlabel("Time (days)")
        plt.ylabel("Resistant Fraction")
        plt.title("Resistance Dynamics")
        plt.legend()
        st.pyplot(fig2)

    # -------------------------
    # Plot 3: True vs Observed (Noise)
    # -------------------------
    st.subheader("Observed vs True Tumor (Scan Noise)")

    fig_noise = plt.figure()

    plt.plot(soc["t"], soc["V_true"], label="True Tumor (SOC)")

    if show_dots:
        plt.scatter(
            soc["t"], soc["V_obs"], s=8, alpha=0.5, label="Observed (SOC)"
        )

    if adaptive_toggle:
        plt.plot(
            adapt["t"], adapt["V_true"],
            linestyle="--",
            label="True Tumor (Adaptive)"
        )
        if show_dots:
            plt.scatter(
                adapt["t"], adapt["V_obs"], s=8, alpha=0.5,
                label="Observed (Adaptive)"
            )

    plt.xlabel("Time (days)")
    plt.ylabel("Tumor (S+R)")
    plt.legend()
    st.pyplot(fig_noise)

    # -------------------------
    # Plot 4: Drug Schedule (Adaptive only)
    # -------------------------
    if adaptive_toggle:
        st.subheader("Adaptive Drug Schedule")
        fig3 = plt.figure()
        plt.step(adapt["t"], adapt["u1"], where="post")
        plt.ylim(-0.1, 1.1)
        plt.xlabel("Time (days)")
        plt.ylabel("Drug ON (1) / OFF (0)")
        st.pyplot(fig3)

    # -------------------------
    # Summary Metrics
    # -------------------------
    st.subheader("Simulation Summary")

    final_soc = soc["V_true"][-1]
    final_soc_Rfrac = soc["R"][-1] / final_soc

    st.write(f"SOC Final Tumor: {final_soc:,.0f} cells")
    st.write(f"SOC Final Resistant Fraction: {final_soc_Rfrac:.4f}")

    if adaptive_toggle:
        final_adapt = adapt["V_true"][-1]
        final_adapt_Rfrac = adapt["R"][-1] / final_adapt
        st.write(f"Adaptive Final Tumor: {final_adapt:,.0f} cells")
        st.write(f"Adaptive Final Resistant Fraction: {final_adapt_Rfrac:.4f}")

    # -------------------------
    # CSV Export
    # -------------------------
    df = pd.DataFrame({
        "Day": soc["t"],
        "SOC_Tumor": soc["V_true"],
        "SOC_Resistant_Fraction": soc_Rfrac
    })

    if adaptive_toggle:
        df["Adaptive_Tumor"] = adapt["V_true"]
        df["Adaptive_Resistant_Fraction"] = adapt_Rfrac

    st.download_button(
        label="Download Results as CSV",
        data=df.to_csv(index=False),
        file_name="simulation_results.csv",
        mime="text/csv",
    )
