# ht_backtest.py
"""
Walk-forward backtest of the first-half goals model (Over/Under 1.5 HT goals).
Each match is predicted using ONLY earlier matches.

    python ht_backtest.py                              # full report
    python ht_backtest.py --start 2025-10-01 --refit-days 14

Needs data/ht_goals.csv  (python ht_data.py).  Output:
  * log loss vs always predicting the base rate   (must be lower to prove an edge)
  * best HT_SCALE and HT_CAL_A / HT_CAL_B to put in config.py
  * calibration table and hit rates of the picks (Under legs matter for the HT acca)
"""
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from ht_model import LINE, SCALE, HIGH_PROB, MEDIUM_PROB, load_training, fit_ht, expected_ht, calibrate


def walk_forward(df, start, refit_days=14):
    test = df[df["date"] >= start]
    fit, last, rows = None, None, []
    for r in test.itertuples():
        if fit is None or (r.date - last).days >= refit_days:
            fit = fit_ht(df[df["date"] < r.date], now=r.date)
            last = r.date
        mh, ma = expected_ht(fit, r.home_team_id, r.away_team_id, r.league_id, scale=1.0)
        rows.append((mh, ma, r.home_ht + r.away_ht))
    return pd.DataFrame(rows, columns=["mh", "ma", "goals"])


def raw_p(res, scale):
    need = int(np.floor(LINE)) + 1
    return poisson.sf(need - 1, (res["mh"] + res["ma"]).values * scale)


def ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float((-(y * np.log(p) + (1 - y) * np.log(1 - p))).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None)
    ap.add_argument("--refit-days", type=int, default=14)
    a = ap.parse_args()

    df = load_training()
    if df is None or len(df) < 300:
        print("Not enough half-time data yet: run  python ht_data.py  first.")
        return
    start = pd.Timestamp(a.start, tz="UTC") if a.start else df["date"].quantile(0.5)
    print(f"{len(df)} matches with a half-time score; testing from {start.date()} ...")
    res = walk_forward(df, start, a.refit_days)
    y = (res["goals"] > LINE).astype(float).values
    base = y.mean()
    print(f"\nTest matches: {len(res)} | Over {LINE:g} HT happened {base*100:.1f}% of the time")
    print(f"Average HT goals: actual {res['goals'].mean():.2f} | model {(res['mh']+res['ma']).mean():.2f}")
    print(f"Log loss: always-base-rate {ll(np.full(len(y), base), y):.4f}")

    best = min(((ll(raw_p(res, s), y), s) for s in np.arange(0.8, 1.31, 0.02)), key=lambda t: t[0])
    scale = round(float(best[1]), 2)
    p0 = raw_p(res, scale)
    z = np.log(np.clip(p0, 1e-4, 1 - 1e-4) / (1 - np.clip(p0, 1e-4, 1 - 1e-4)))
    w = minimize(lambda w: ll(1 / (1 + np.exp(-(w[0] * z + w[1]))), y), [1.0, 0.0], method="Nelder-Mead").x
    p_cal = calibrate(p0, w[0], w[1])
    print(f"Log loss: model (scale {SCALE:g}) {ll(raw_p(res, SCALE), y):.4f} | "
          f"best scale {scale} -> {best[0]:.4f} | + calibration {ll(p_cal, y):.4f}")
    print(f"\nPut in config.py:\n    HT_SCALE = {scale}\n    HT_CAL_A = {w[0]:.2f}\n    HT_CAL_B = {w[1]:.2f}")

    print("\nCalibration (P(Over 1.5 HT), with the settings above):")
    bins = [0, .3, .4, .5, .6, .7, 1.01]
    cut = pd.cut(pd.Series(p_cal), bins, right=False)
    for b, g in pd.DataFrame({"p": p_cal, "y": y}).groupby(cut, observed=True):
        print(f"  {str(b):<14} n={len(g):<5} predicted {g['p'].mean()*100:5.1f}%  actual {g['y'].mean()*100:5.1f}%")

    pick_over = p_cal >= 0.5
    prob = np.where(pick_over, p_cal, 1 - p_cal)
    hit = np.where(pick_over, y, 1 - y)
    print("\nPicks by confidence (probability of the chosen side):")
    for name, lo, hi in [("HIGH", HIGH_PROB, 1.01), ("MEDIUM", MEDIUM_PROB, HIGH_PROB), ("LOW", 0, MEDIUM_PROB)]:
        m = (prob >= lo) & (prob < hi)
        if m.any():
            print(f"  {name:<7} n={m.sum():<5} predicted {prob[m].mean()*100:5.1f}%  hit {hit[m].mean()*100:5.1f}%")
    u = ~pick_over
    if u.any():
        print(f"\nUnder {LINE:g} HT picks (what the HT acca uses): n={u.sum()}, "
              f"predicted {prob[u].mean()*100:.1f}%, hit {hit[u].mean()*100:.1f}%")
    for t in (0.58, 0.65, 0.70, 0.75):
        m = u & (prob >= t)
        if m.any():
            print(f"   Under picks >= {t*100:.0f}%: n={m.sum():<5} hit {hit[m].mean()*100:5.1f}%")
    verdict = "BEATS" if ll(p_cal, y) < ll(np.full(len(y), base), y) else "DOES NOT BEAT"
    print(f"\nVerdict: the model {verdict} the plain base rate on log loss.")


if __name__ == "__main__":
    main()
