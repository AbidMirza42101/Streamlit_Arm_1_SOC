# model.py
import numpy as np
from scipy.integrate import solve_ivp


def tumor_ode(t, y, params, u1):
    S, R = y

    rS = params["rS"]
    rR = params["rR"]
    K = params["K"]
    dS0 = params["dS0"]
    dR0 = params["dR0"]
    killS = params["killS"]
    mu = params["mu"]

    eff_dS = dS0 + u1 * killS
    eff_dR = dR0  # resistant not affected

    V = S + R

    dSdt = rS * S * (1 - V / K) - eff_dS * S - mu * S
    dRdt = rR * R * (1 - V / K) - eff_dR * R + mu * S

    return [dSdt, dRdt]


def simulate_model(
    params,
    y0,
    t_end=400,
    visit_every=14,
    noise_sigma=0.05,
    seed=7,
    adaptive=False,
    stop_frac=0.5,
    restart_frac=1.0,
):

    rng = np.random.default_rng(seed)
    times = np.arange(0, t_end + 1)

    S_hist = np.zeros_like(times, dtype=float)
    R_hist = np.zeros_like(times, dtype=float)
    V_true = np.zeros_like(times, dtype=float)
    V_obs = np.zeros_like(times, dtype=float)
    u1_hist = np.zeros_like(times, dtype=float)

    S, R = map(float, y0)
    S_hist[0], R_hist[0] = S, R
    V_true[0] = S + R

    V_obs[0] = V_true[0] * np.exp(rng.normal(0, noise_sigma))
    baseline = V_obs[0]
    last_measured = V_obs[0]

    u1 = 1.0  # start ON

    for i in range(0, t_end):

        t = times[i]

        if t % visit_every == 0:
            last_measured = (S + R) * np.exp(rng.normal(0, noise_sigma))

        V_obs[i] = last_measured

        if adaptive:
            if last_measured <= stop_frac * baseline:
                u1 = 0.0
            elif last_measured >= restart_frac * baseline:
                u1 = 1.0
        else:
            u1 = 1.0

        u1_hist[i] = u1

        sol = solve_ivp(
            fun=lambda tt, yy: tumor_ode(tt, yy, params, u1),
            t_span=(0, 1),
            y0=[S, R],
            t_eval=[1],
        )

        S, R = sol.y[0, -1], sol.y[1, -1]
        S = max(S, 0.0)
        R = max(R, 0.0)

        S_hist[i + 1] = S
        R_hist[i + 1] = R
        V_true[i + 1] = S + R

    return {
        "t": times,
        "S": S_hist,
        "R": R_hist,
        "V_true": V_true,
        "V_obs": V_obs,
        "u1": u1_hist,
        "baseline": baseline,
    }
