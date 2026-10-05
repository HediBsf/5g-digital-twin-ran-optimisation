"""
AI-driven energy- and EMF-aware management of a 5G RAN using a Digital Twin
and federated traffic prediction.

Simplified research prototype (7-cell macro cluster, 3.5 GHz, 100 MHz):
  1. Network simulator: path loss + shadowing, SINR, per-UE throughput,
     EARTH-style base-station power model, EM-exposure proxy.
  2. Digital Twin: ML surrogate (gradient boosting) of the simulator that predicts
     energy, QoS and exposure from observable data only (per-cell traffic + config).
  3. Twin-based optimisation: choose per-cell Tx power / sleep mode under a QoS
     constraint, then validate on the "real" network (simulator with true UE positions).
  4. Federated learning: FedAvg traffic forecasting across cells (raw data stays local),
     compared with local-only and centralised training.

Run:  python fiveg_twin.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

SEED = 42
rng = np.random.default_rng(SEED)

# ----------------------------------------------------------------------------
# 1. Network simulator
# ----------------------------------------------------------------------------
ISD, FC_GHZ, BW, NF_DB = 500.0, 3.5, 100e6, 7.0
CELL_R = ISD / np.sqrt(3)
CELLS = np.vstack([[0, 0]] + [[ISD * np.cos(k * np.pi / 3), ISD * np.sin(k * np.pi / 3)] for k in range(6)])
NC = len(CELLS)
ANT_GAIN = 10 ** (15 / 10)                      # 15 dBi
H_DIFF = 25.0 - 1.5                              # BS height - UE height (m)
dbm2w = lambda d: 10 ** ((d - 30) / 10)
LEVELS_DBM = [None, 33, 37, 40, 43]              # level 0 = sleep mode
TX_W = np.array([0.0] + [dbm2w(x) for x in LEVELS_DBM[1:]])
NOISE_W = dbm2w(-174 + 10 * np.log10(BW) + NF_DB)
# EARTH-style macro BS power model (Auer et al., 2011)
P0, DP, P_SLEEP, N_CAP = 130.0, 4.7, 75.0, 20
RATE_MIN = 5e6                                   # QoS: 5 Mbps per UE
QOS_TARGET = 0.95


# Fixed receiver grid for an area-averaged EM-exposure proxy (independent of the UE drop)
_gx, _gy = np.meshgrid(np.linspace(-ISD * 1.3, ISD * 1.3, 25), np.linspace(-ISD * 1.3, ISD * 1.3, 25))
GRID = np.c_[_gx.ravel(), _gy.ravel()]
_dg = np.sqrt(np.maximum(np.linalg.norm(GRID[:, None, :] - CELLS[None, :, :], axis=2), 30.0) ** 2 + H_DIFF ** 2)
GRID_S = ANT_GAIN / (4 * np.pi * _dg ** 2)       # W/m^2 per Watt radiated (30 m exclusion distance)


def make_drop(counts, r):
    """Random UE drop: positions + log-normal shadowing -> channel gain matrix (UE x cell)."""
    pos = []
    for c, n in enumerate(counts):
        rad = CELL_R * np.sqrt(r.random(n))
        th = 2 * np.pi * r.random(n)
        pos.append(CELLS[c] + np.c_[rad * np.cos(th), rad * np.sin(th)])
    pos = np.vstack(pos) if sum(counts) else np.zeros((0, 2))
    d2 = np.linalg.norm(pos[:, None, :] - CELLS[None, :, :], axis=2)
    d3 = np.sqrt(np.maximum(d2, 10.0) ** 2 + H_DIFF ** 2)
    pl_db = 28 + 22 * np.log10(d3) + 20 * np.log10(FC_GHZ) + r.normal(0, 6, d3.shape)
    return {"gain": ANT_GAIN * 10 ** (-pl_db / 10), "d3": d3}


def evaluate(drop, levels):
    """KPIs of a configuration on a given UE drop."""
    levels = np.asarray(levels)
    ptx = TX_W[levels]
    active = ptx > 0
    n_ue = drop["gain"].shape[0]
    if n_ue == 0:
        energy = np.where(active, P0, P_SLEEP).sum()
        return energy, 1.0, float(np.mean(GRID_S @ ptx))
    if not active.any():
        return NC * P_SLEEP, 0.0, 0.0
    rx = drop["gain"] * ptx[None, :]
    serv = rx.argmax(1)
    sig = rx[np.arange(n_ue), serv]
    sinr = sig / (rx.sum(1) - sig + NOISE_W)
    n_serv = np.bincount(serv, minlength=NC)
    rate = BW / n_serv[serv] * np.log2(1 + sinr)
    qos = float(np.mean(rate >= RATE_MIN))
    load = np.minimum(1.0, n_serv / N_CAP)
    energy = float(np.where(active, P0 + DP * ptx * load, P_SLEEP).sum())
    # EM-exposure proxy: area-averaged power density over a fixed grid (W/m^2)
    expo = float(np.mean(GRID_S @ ptx))
    return energy, qos, expo


# Traffic model: cell-specific daily profiles (business vs residential)
PEAK_H = np.array([13, 20, 13, 20, 13, 20, 13], dtype=float)
PEAK_UE = np.array([10, 10, 9, 9, 9, 9, 9], dtype=float)


def profile(h, peak):
    d = np.abs(((h - peak) + 12) % 24 - 12)
    return 0.12 + 0.88 * np.exp(-0.5 * (d / 3.5) ** 2)


def sample_counts(h, r):
    lam = PEAK_UE * profile(h, PEAK_H) * r.lognormal(0, 0.15, NC)
    return r.poisson(lam)


# ----------------------------------------------------------------------------
# 2. Digital Twin (ML surrogate)
# ----------------------------------------------------------------------------
def build_twin_dataset(n, r):
    X, Y = [], []
    for _ in range(n):
        h = r.integers(0, 24)
        counts = sample_counts(h, r)
        p = r.choice([0.05, 0.15, 0.3])
        levels = np.where(r.random(NC) < p, 0, r.integers(1, 5, NC))
        if not levels.any():                      # at least one active cell
            levels[r.integers(NC)] = r.integers(1, 5)
        e, q, x = evaluate(make_drop(counts, r), levels)
        X.append(np.r_[counts, levels])
        Y.append([e, q, np.log10(x + 1e-12)])
    return np.array(X, float), np.array(Y)


def train_twin():
    X, Y = build_twin_dataset(12000, rng)
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=SEED)
    twin = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.08,
                                                          random_state=SEED))
    twin.fit(Xtr, Ytr)
    pred = twin.predict(Xte)
    r2 = {k: float(r2_score(Yte[:, i], pred[:, i])) for i, k in enumerate(["energy", "qos", "exposure_log"])}
    r2["qos_MAE"] = float(np.abs(Yte[:, 1] - pred[:, 1]).mean())
    r2["energy_MAPE_pct"] = float(100 * np.mean(np.abs(Yte[:, 0] - pred[:, 0]) / Yte[:, 0]))
    return twin, r2


# ----------------------------------------------------------------------------
# 3. Twin-based optimisation, validated on the "real" network
# ----------------------------------------------------------------------------
def optimise_with_twin(twin, counts, r, n_cand=3000, qos_margin=0.04, lam=0.5):
    cand = np.where(r.random((n_cand, NC)) < 0.2, 0, r.integers(1, 5, (n_cand, NC)))
    X = np.c_[np.tile(counts, (n_cand, 1)), cand]
    e, q, x = twin.predict(X).T
    ok = q >= QOS_TARGET + qos_margin
    if not ok.any():
        return np.full(NC, 4)                    # safe fallback: all cells at max power
    e_n = (e - e[ok].min()) / (np.ptp(e[ok]) + 1e-9)
    x_n = (x - x[ok].min()) / (np.ptp(x[ok]) + 1e-9)
    cost = np.where(ok, e_n + lam * x_n, np.inf)
    return cand[cost.argmin()]


def run_optimisation(twin):
    r = np.random.default_rng(SEED + 1)
    rows = []
    for h in range(24):
        for _ in range(5):
            counts = sample_counts(h, r)
            drop = make_drop(counts, r)
            base = evaluate(drop, np.full(NC, 4))
            cfg = optimise_with_twin(twin, counts, r)
            opt = evaluate(drop, cfg)
            rows.append((h, *base, *opt, int((cfg == 0).sum())))
    a = np.array(rows)
    res = {
        "scenarios": len(a),
        "energy_saving_pct": float(100 * (1 - a[:, 4].sum() / a[:, 1].sum())),
        "exposure_reduction_pct": float(100 * (1 - a[:, 6].sum() / a[:, 3].sum())),
        "qos_baseline_mean": float(a[:, 2].mean()),
        "qos_twin_mean": float(a[:, 5].mean()),
        "qos_twin_ge_target_pct": float(100 * (a[:, 5] >= QOS_TARGET).mean()),
        "avg_sleeping_cells": float(a[:, 7].mean()),
        "night_energy_saving_pct_0_6h": float(100 * (1 - a[a[:, 0] <= 6, 4].sum() / a[a[:, 0] <= 6, 1].sum())),
        "peak_energy_saving_pct_12_21h": float(100 * (1 - a[(a[:, 0] >= 12) & (a[:, 0] <= 21), 4].sum()
                                                   / a[(a[:, 0] >= 12) & (a[:, 0] <= 21), 1].sum())),
    }
    return res, a


def plot_results(a, path):
    hrs = np.arange(24)
    eb = [a[a[:, 0] == h, 1].mean() for h in hrs]
    eo = [a[a[:, 0] == h, 4].mean() for h in hrs]
    qb = [a[a[:, 0] == h, 2].mean() for h in hrs]
    qo = [a[a[:, 0] == h, 5].mean() for h in hrs]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(hrs, eb, label="Baseline (all cells 43 dBm)")
    ax[0].plot(hrs, eo, label="Digital-Twin optimised")
    ax[0].set_xlabel("Hour of day"); ax[0].set_ylabel("Cluster power (W)"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(hrs, qb, label="Baseline"); ax[1].plot(hrs, qo, label="Digital-Twin optimised")
    ax[1].axhline(QOS_TARGET, ls="--", c="grey", label="QoS target")
    ax[1].set_xlabel("Hour of day"); ax[1].set_ylabel("UEs with >= 5 Mbps"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


# ----------------------------------------------------------------------------
# 4. Federated traffic forecasting (FedAvg) vs local-only vs centralised
# ----------------------------------------------------------------------------
LAGS = 6


def cell_series(c, days, r):
    h = np.arange(days * 24)
    base = PEAK_UE[c] * profile(h % 24, PEAK_H[c])
    return np.maximum(0, base * r.lognormal(0, 0.25, len(h)) + r.normal(0, 1.0, len(h)))


def make_xy(s):
    h = np.arange(len(s))
    rows, ys = [], []
    for t in range(LAGS, len(s)):
        rows.append(np.r_[s[t - LAGS:t] / 10, np.sin(2 * np.pi * h[t] / 24), np.cos(2 * np.pi * h[t] / 24),
                          np.sin(4 * np.pi * h[t] / 24), np.cos(4 * np.pi * h[t] / 24), 1.0])
        ys.append(s[t] / 10)
    return np.array(rows), np.array(ys)


def ridge(X, y, lam=1e-2):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def fedavg(data, rounds=60, local_steps=5, lr=0.05):
    w = np.zeros(data[0][0].shape[1])
    n = np.array([len(y) for _, y in data], float)
    for _ in range(rounds):
        ws = []
        for X, y in data:
            wl = w.copy()
            for _ in range(local_steps):
                wl -= lr * 2 * X.T @ (X @ wl - y) / len(y)
            ws.append(wl)
        w = np.average(ws, axis=0, weights=n)
    return w


def run_federated():
    r = np.random.default_rng(SEED + 2)
    train = [make_xy(cell_series(c, 2, r)) for c in range(NC)]      # only 2 days of local data per cell
    test = [make_xy(cell_series(c, 14, r)) for c in range(NC)]
    mae = lambda w: float(np.mean([np.abs(Xt @ w - yt).mean() * 10 for Xt, yt in test]))
    mae_local = float(np.mean([np.abs(Xt @ ridge(*train[c]) - yt).mean() * 10 for c, (Xt, yt) in enumerate(test)]))
    Xc, yc = np.vstack([t[0] for t in train]), np.concatenate([t[1] for t in train])
    persistence = float(np.mean([np.abs(Xt[:, LAGS - 1] * 10 - yt * 10).mean() for Xt, yt in test]))
    return {"MAE_persistence_UEs": persistence, "MAE_local_only_UEs": mae_local,
            "MAE_federated_FedAvg_UEs": mae(fedavg(train)), "MAE_centralised_UEs": mae(ridge(Xc, yc))}


if __name__ == "__main__":
    print("Training Digital Twin surrogate ...")
    twin, r2 = train_twin()
    print("Twin fidelity (R2 on held-out simulator runs):", r2)
    print("Optimising with the Digital Twin and validating on the simulator ...")
    res, table = run_optimisation(twin)
    print(json.dumps(res, indent=2))
    plot_results(table, "results_energy_qos.png")
    print("Federated traffic forecasting ...")
    fed = run_federated()
    print(json.dumps(fed, indent=2))
    with open("results.json", "w") as f:
        json.dump({"twin_r2": r2, "optimisation": res, "federated": fed}, f, indent=2)
