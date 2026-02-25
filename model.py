"""
Arm 1 (SOC) ONLY — Continuous Tamoxifen, No Fasting

This script:
1) Auto-checks required libraries (numpy, scipy, matplotlib)
2) If missing, installs them using: python -m pip install ...
3) Lets you ENTER ONLY these values:
   - rS (sensitive growth rate)
   - rR (resistant growth rate)
   - killS (tamoxifen kill on S)
   - t_end (number of days to simulate)
   - visit_every (scan/measurement frequency in days)
   - noise_sigma (measurement noise)
   - show_dots (True/False: show observed dots vs true line)
4) Keeps everything else exactly the same as your current code.
5) Runs Arm 1 (SOC): u1=1 always, u2=0 always
6) Plots:
   - Total tumor burden (S+R)
   - Resistant fraction over time
   - Optional: True vs Observed (dots)
"""

# -------------------------------
# 0) Auto-install missing packages
# -------------------------------
import sys
import subprocess

def ensure_packages(pkgs):
    """Install missing packages into the current Python environment."""
    missing = []
    for p in pkgs:
        try:
            __import__(p)
        except ImportError:
            missing.append(p)

    if missing:
        print(f"[INFO] Missing packages detected: {missing}")
        print("[INFO] Installing now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
        print("[INFO] Installation complete.\n")

ensure_packages(["numpy", "scipy", "matplotlib"])

# -------------------------------
# 1) Imports (now safe)
# -------------------------------
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# -------------------------------
# 2) Helper: safe user inputs (press Enter to keep defaults)
# -------------------------------
def ask_float(prompt, default):
    s = input(f"{prompt} [{default}]: ").strip()
    if s == "":
        return float(default)
    return float(s)

def ask_int(prompt, default):
    s = input(f"{prompt} [{default}]: ").strip()
    if s == "":
        return int(default)
    return int(s)

def ask_bool(prompt, default):
    d = "y" if default else "n"
    s = input(f"{prompt} (y/n) [{d}]: ").strip().lower()
    if s == "":
        return bool(default)
    if s in ["y", "yes", "true", "1"]:
        return True
    if s in ["n", "no", "false", "0"]:
        return False
    print("[WARN] Could not parse. Using default.")
    return bool(default)

# -------------------------------
# 3) ODE system
# -------------------------------
def tumor_ode(t, y, params, u1, u2):
    S, R = y

    rS = params["rS"]
    rR = params["rR"]
    K  = params["K"]
    dS0 = params["dS0"]
    dR0 = params["dR0"]
    killS = params["killS"]
    killR = params["killR"]
    mu = params["mu"]

    keto_eff_S = params["keto_eff_S"]
    keto_eff_R = params["keto_eff_R"]

    # Apply controls (u2 won't matter for SOC because u2=0)
    eff_rS = rS * (1 - u2 * keto_eff_S)
    eff_rR = rR * (1 - u2 * keto_eff_R)

    # Tamoxifen effect (SOC: u1=1 always)
    eff_dS = dS0 + u1 * killS
    eff_dR = dR0 + u1 * killR

    V = S + R

    # Logistic crowding + death + mutation
    dSdt = eff_rS * S * (1 - V / K) - eff_dS * S - mu * S
    dRdt = eff_rR * R * (1 - V / K) - eff_dR * R + mu * S

    return [dSdt, dRdt]

# -------------------------------
# 4) SOC simulator (Arm 1 only)
# -------------------------------
def simulate_soc(
    params,
    y0,
    t_end=400,
    visit_every=14,
    noise_sigma=0.05,
    seed=7,
):
    """
    SOC Arm:
      - Tamoxifen always ON: u1=1
      - Fasting always OFF:  u2=0
    Still keeps:
      - clinic-like measurements (every visit_every days)
      - log-normal measurement noise
    """
    rng = np.random.default_rng(seed)

    times = np.arange(0, t_end + 1)

    S_hist = np.zeros_like(times, dtype=float)
    R_hist = np.zeros_like(times, dtype=float)
    V_true = np.zeros_like(times, dtype=float)
    V_obs  = np.zeros_like(times, dtype=float)
    u1_hist = np.zeros_like(times, dtype=float)
    u2_hist = np.zeros_like(times, dtype=float)

    # initial state
    S, R = map(float, y0)
    S_hist[0], R_hist[0] = S, R
    V_true[0] = S + R

    # baseline measurement (with noise)
    V_obs[0] = V_true[0] * np.exp(rng.normal(0, noise_sigma))
    baseline = V_obs[0]

    # SOC controls
    u1, u2 = 1.0, 0.0
    last_measured = V_obs[0]

    for i in range(0, t_end):
        t = times[i]

        # clinic measurement every visit_every days (hold last value in-between)
        if t % visit_every == 0:
            last_measured = (S + R) * np.exp(rng.normal(0, noise_sigma))
        V_obs[i] = last_measured

        u1_hist[i] = u1
        u2_hist[i] = u2

        # integrate one day forward
        sol = solve_ivp(
            fun=lambda tt, yy: tumor_ode(tt, yy, params, u1=u1, u2=u2),
            t_span=(0, 1),
            y0=[S, R],
            t_eval=[1],
            method="RK45",
        )

        S, R = sol.y[0, -1], sol.y[1, -1]
        S = max(S, 0.0)
        R = max(R, 0.0)

        S_hist[i + 1] = S
        R_hist[i + 1] = R
        V_true[i + 1] = S + R

    # fill last day
    V_obs[-1] = last_measured
    u1_hist[-1] = u1
    u2_hist[-1] = u2

    return {
        "t": times,
        "S": S_hist,
        "R": R_hist,
        "V_true": V_true,
        "V_obs": V_obs,
        "u1": u1_hist,
        "u2": u2_hist,
        "baseline": baseline,
        "strategy": "SOC",
    }

# -------------------------------
# 5) Main: parameters (keep same defaults, only allow edits to requested ones)
# -------------------------------
DEFAULT_rS = 0.035
DEFAULT_rR = 0.025
DEFAULT_killS = 0.06
DEFAULT_t_end = 400
DEFAULT_visit_every = 14
DEFAULT_noise_sigma = 0.05
DEFAULT_show_dots = True

print("\n=== Arm 1 (SOC) Parameter Input ===")
rS_in = ask_float("Enter rS (Sensitive growth per day)", DEFAULT_rS)
rR_in = ask_float("Enter rR (Resistant growth per day)", DEFAULT_rR)
killS_in = ask_float("Enter killS (Tamoxifen kill on S per day)", DEFAULT_killS)
t_end_in = ask_int("Enter number of days to simulate (t_end)", DEFAULT_t_end)
visit_every_in = ask_int("Enter visit_every (scan frequency in days)", DEFAULT_visit_every)
noise_sigma_in = ask_float("Enter noise_sigma (measurement noise, e.g. 0.05)", DEFAULT_noise_sigma)
show_dots_in = ask_bool("Show observed dots vs true line?", DEFAULT_show_dots)

# Keep everything else unchanged
params = {
    "rS": rS_in,
    "rR": rR_in,
    "K": 2e9,
    "dS0": 0.005,
    "dR0": 0.005,
    "killS": killS_in,
    "killR": 0.0,
    "mu": 1e-6,
    "keto_eff_S": 0.10,
    "keto_eff_R": 0.45,
}

y0 = [5e8, 1e6]  # initial (S, R) kept same

soc = simulate_soc(
    params,
    y0,
    t_end=t_end_in,
    visit_every=visit_every_in,
    noise_sigma=noise_sigma_in,
    seed=7
)

# -------------------------------
# 6) Plots (Arm 1 only)
# -------------------------------
# Plot A: Total tumor burden (true)
plt.figure()
plt.plot(soc["t"], soc["V_true"], label="SOC: Tamoxifen ON (u1=1), Fasting OFF (u2=0)")
plt.xlabel("Time (days)")
plt.ylabel("Total tumor (S+R)")
plt.title("Arm 1 (SOC) — Tumor Burden Over Time (True)")
plt.legend()
plt.show()

# Plot B: Resistant fraction (true)
plt.figure()
Rfrac = soc["R"] / np.maximum(soc["V_true"], 1e-12)
plt.plot(soc["t"], Rfrac)
plt.xlabel("Time (days)")
plt.ylabel("Resistant fraction R/(S+R)")
plt.title("Arm 1 (SOC) — Resistance Fraction Over Time")
plt.show()

# Plot C (optional): True vs Observed (dots)
if show_dots_in:
    plt.figure()
    plt.plot(soc["t"], soc["V_true"], label="True tumor (hidden)")
    plt.scatter(soc["t"], soc["V_obs"], s=10, alpha=0.5, label="Measured tumor (noisy scans)")
    plt.xlabel("Time (days)")
    plt.ylabel("Tumor (S+R)")
    plt.title("Arm 1 (SOC) — True vs Observed Tumor Burden")
    plt.legend()
    plt.show()