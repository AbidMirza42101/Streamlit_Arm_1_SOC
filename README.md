# Tamoxifen Evolutionary Therapy Simulator

This Streamlit app simulates tumor dynamics in HR+ metastatic breast cancer using a two-population ODE model:

- Sensitive cells (S)
- Resistant cells (R)

The model supports:

## Arm 1 — Standard of Care (SOC)
Continuous Tamoxifen (drug always ON).

## Arm 2 — Adaptive Therapy
Tamoxifen automatically switches ON/OFF based on tumor burden thresholds:

- Stop when tumor falls below a chosen % of baseline.
- Restart when tumor rises back to a chosen %.

---

## Adjustable Parameters

Users can modify:

- rS — Sensitive growth rate
- rR — Resistant growth rate
- killS — Tamoxifen kill rate
- Simulation days
- Scan interval
- Measurement noise
- Random seed
- Adaptive toggle
- Stop & restart thresholds

---

## Outputs

The app generates:

1. Total tumor burden over time
2. Resistant fraction over time
3. Drug ON/OFF schedule (u1)

---

## Run Locally

