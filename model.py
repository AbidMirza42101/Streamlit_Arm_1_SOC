# model.py
"""
Arm 1 (SOC) model utilities.

Contains:
- tumor_ode: two-population (S, R) ODE with logistic crowding + death + mutation
- simulate_soc: runs 0..t_end days with tamoxifen always ON (u1=1) and fasting OFF (u2=0)
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def tumor_ode(t: float, y: np.ndarray, params: dict, u1: float, u2: float):
    """
    ODE system for Sensitive (S) and Resistant (R) populations.

    Controls:
      u1 = tamoxifen (SOC fixed to 1)
      u2 = fasting/keto (SOC fixed to 0)
    """
    S, R = y

    rS = params["rS"]
    rR = params["rR"]
    K = params["K"]

    dS0 = params["dS0"]
    dR0 = params["dR0"]
    killS = params["killS"]
    killR = params["killR"]
    mu = params["mu"]

    keto_eff_S = params["keto_eff_S"]
    keto_eff_R = params["keto_eff_R"]

    # Apply controls
    eff_rS = rS * (1 - u2 * keto_eff_S)
    eff_rR = rR * (1 - u2 * keto_eff_R)

    eff_dS = dS0 + u1 * killS
    eff_dR = dR0 + u1 * killR

    V = S + R

    dSdt = eff_rS * S * (1 - V / K) - eff_dS * S - mu * S
    dRdt = eff_rR * R * (1 - V / K) - eff_dR * R + mu * S

    return [dSdt, dRdt]


def simulate_soc(
    params: dict,
    y0: tuple[float, float] | list[float],
    t_end: int = 400,
    visit_every: int = 14,
    noise_sigma: float = 0.05,
    seed: int = 7,
):
    """
    SOC Arm (Arm 1):
      - Tamoxifen always ON: u1 = 1
      - Fasting always OFF: u2 = 0

    Measurements:
      - every `visit_every` days
      - log-normal noise: measured = true * exp(N(0, noise_sigma))
      - between visits: hold last measured value
    """
    rng = np.random.default_rng(seed)

    times = np.arange(0, t_end + 1)

    S_hist = np.zeros_like(times, dtype=float)
    R_hist = np.zeros_like(times, dtype=float)
    V_true = np.zeros_like(times, dtype=float)
    V_obs = np.zeros_like(times, dtype=float)

    u1_hist = np.ones_like(times, dtype=float)   # always 1
    u2_hist = np.zeros_like(times, dtype=float)  # always 0

    # initial state
    S, R = map(float, y0)
    S_hist[0], R_hist[0] = S, R
    V_true[0] = S + R

    # baseline noisy measurement
    V_obs[0] = V_true[0] * np.exp(rng.normal(0, noise_sigma))
    baseline = V_obs[0]
    last_measured = V_obs[0]

    # fixed SOC controls
    u1, u2 = 1.0, 0.0

    for i in range(0, t_end):
        t = times[i]

        # clinic measurement at visit days
        if t % visit_every == 0:
            last_measured = (S + R) * np.exp(rng.normal(0, noise_sigma))
        V_obs[i] = last_measured

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

    # last point
    V_obs[-1] = last_measured

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
