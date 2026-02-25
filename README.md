# Arm 1 (SOC) Tamoxifen Simulator — Streamlit App

This app simulates tumor dynamics using a 2-population ODE model:
- **S** = Tamoxifen-sensitive cells
- **R** = Tamoxifen-resistant cells

Arm 1 (SOC) assumptions:
- Tamoxifen always ON (u1 = 1)
- Fasting always OFF (u2 = 0)

## Files
- `app.py` — Streamlit UI (inputs + plots)
- `model.py` — Core model functions (ODE + simulator)
- `requirements.txt` — Python dependencies

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
