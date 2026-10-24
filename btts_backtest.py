# btts_backtest.py
"""
Walk-forward backtest of Both Teams To Score (BTTS) from the goals model.
Each match is predicted using ONLY earlier matches. Needs no extra data (uses full-time scores).

    python btts_backtest.py
    python btts_backtest.py --start 2026-01-01 --refit-days 14

Prints log loss vs the plain base rate, the best BTTS_SCALE / BTTS_CAL_A / BTTS_CAL_B for config.py,
a calibration table, and the hit rate of BTTS Yes picks by probability (what the BTTS acca uses).
"""
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import LAMBDA_BOOST
from model import fit_model, lambdas_from_fit, load_data


def p_btts(lh, la):
    """P(both teams score) with independent Poisson goals."""
    return (1 - np.exp(-np.asarray(lh))) * (1 - np.exp(-np.asarray(la)))


def ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float((-(y * np.log(p) + (1 - y) * np.log(1 - p))).mean())


def walk_forward(df, start, refit_days=14):
    fit, last, rows = None, None, []
    for r in df[df["date"] >= start].itertuples():
        if fit is None or (r.date - last).days >= refit_days:
            fit = fit_model(df[df["date"] < r.date], now=r.date)
            last = r.date
        lh, la = lambdas_from_fit(fit, r.home_team_id, r.away_team_id, r.league_id)
        rows.append((lh, la, int(r.home_goals > 0 and r.away_goals > 0)))
    return pd.DataFrame(rows, columns=["lh", "la", "y"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None)
    ap.add_argument("--refit-days", type=int, default=14)
    a = ap.parse_args()

    df = load_data().sort_values("date").reset_index(drop=True)
    start = pd.Timestamp(a.start, tz="UTC") if a.start else df["date"].quantile(0.5)
    print(f"{len(df)} matches; testing from {start.date()} ...")
    res = walk_forward(df, start, a.refit_days)
    y = res["y"].values.astype(float)
    base = y.mean()
    print(f"\nTest matches: {len(res)} | BTTS happened {base*100:.1f}% of the time")
    print(f"Log loss: always-base-rate {ll(np.full(len(y), base), y):.4f}")

    def raw(scale):
        return p_btts(res["lh"].values * scale, res["la"].values * scale)

    print(f"Log loss: model (boost {LAMBDA_BOOST:g}) {ll(raw(LAMBDA_BOOST), y):.4f}")
    best = min(((ll(raw(s), y), s) for s in np.arange(0.8, 1.41, 0.02)), key=lambda t: t[0])
    scale = round(float(best[1]), 2)
    p0 = np.clip(raw(scale), 1e-4, 1 - 1e-4)
    z = np.log(p0 / (1 - p0))
    w = minimize(lambda w: ll(1 / (1 + np.exp(-(w[0] * z + w[1]))), y), [1.0, 0.0], method="Nelder-Mead").x
    p = 1 / (1 + np.exp(-(w[0] * z + w[1])))
    print(f"Log loss: best scale {scale} -> {best[0]:.4f} | + calibration {ll(p, y):.4f}")
    print(f"\nPut in config.py:\n    BTTS_SCALE = {scale}\n    BTTS_CAL_A = {w[0]:.2f}\n    BTTS_CAL_B = {w[1]:.2f}")

    print("\nCalibration (P(BTTS Yes), with the settings above):")
    cut = pd.cut(pd.Series(p), [0, .35, .45, .5, .55, .6, .65, .7, 1.01], right=False)
    for b, g in pd.DataFrame({"p": p, "y": y}).groupby(cut, observed=True):
        print(f"  {str(b):<14} n={len(g):<5} predicted {g['p'].mean()*100:5.1f}%  actual {g['y'].mean()*100:5.1f}%")

    print("\nBTTS YES picks by probability (what the BTTS acca uses):")
    for t in (0.50, 0.55, 0.58, 0.60, 0.65):
        m = p >= t
        if m.any():
            print(f"   Yes >= {t*100:.0f}%: n={m.sum():<5} predicted {p[m].mean()*100:5.1f}%  hit {y[m].mean()*100:5.1f}%")
    verdict = "BEATS" if ll(p, y) < ll(np.full(len(y), base), y) else "DOES NOT BEAT"
    print(f"\nVerdict: the model {verdict} the plain base rate on log loss.")


if __name__ == "__main__":
    main()
