# backtest.py
"""
Walk-forward backtest. Run: python backtest.py [--start 2025-10-01] [--refit-days 14]

Each match is predicted using ONLY earlier matches (model refit every N days).
Reports log loss / Brier / accuracy vs. a constant "base rate" baseline,
a calibration table, and hit rates for the confidence bands.

Log loss is the number to watch: it must beat the baseline to prove the
probabilities carry real information. (Accuracy alone is misleading.)
"""
import argparse
import numpy as np
import pandas as pd

from config import LAMBDA_BOOST, HIGH_CONF_OVER_PROB, HIGH_CONF_UNDER_PROB
from model import load_data, fit_model, lambdas_from_fit, over_under_2_5


def walk_forward(df, start, refit_days=14, **fit_kwargs):
    test = df[df["date"] >= start]
    fit, last, rows = None, None, []
    for _, r in test.iterrows():
        if fit is None or (r["date"] - last).days >= refit_days:
            fit = fit_model(df[df["date"] < r["date"]], now=r["date"], **fit_kwargs)
            last = r["date"]
        lh, la = lambdas_from_fit(fit, r["home_team_id"], r["away_team_id"], r["league_id"])
        rows.append((lh, la, int(r["total_goals"] > 2.5)))
    return pd.DataFrame(rows, columns=["lh", "la", "over"], index=test.index)


def p_over(res, scale):
    return np.array([over_under_2_5(a * scale, b * scale)[0] for a, b in zip(res.lh, res.la)])


def metrics(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return {"logloss": -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)),
            "brier": np.mean((p - y) ** 2),
            "acc": np.mean((p > 0.5) == y)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-10-01")
    ap.add_argument("--refit-days", type=int, default=14)
    args = ap.parse_args()

    df = load_data()
    start = pd.Timestamp(args.start, tz="UTC")
    res = walk_forward(df, start, args.refit_days)
    y = res["over"].values
    p = p_over(res, LAMBDA_BOOST)

    base_rate = df[df["date"] < start]["total_goals"].gt(2.5).mean()
    m, b = metrics(p, y), metrics(np.full(len(y), base_rate), y)

    print(f"\nTest matches: {len(y)}   actual over rate: {y.mean()*100:.1f}%   "
          f"mean predicted: {p.mean()*100:.1f}%")
    print(f"{'':<12}{'logloss':>9}{'brier':>9}{'acc':>8}")
    print(f"{'baseline':<12}{b['logloss']:>9.4f}{b['brier']:>9.4f}{b['acc']*100:>7.1f}%")
    print(f"{'model':<12}{m['logloss']:>9.4f}{m['brier']:>9.4f}{m['acc']*100:>7.1f}%")
    print(f"=> logloss improvement vs baseline: {b['logloss']-m['logloss']:+.4f} "
          f"({'GOOD' if m['logloss'] < b['logloss'] else 'NO EDGE'})")

    print("\nCalibration (predicted vs actual over rate)")
    print(f"{'bin':<12}{'n':>6}{'pred':>8}{'actual':>8}")
    bins = [0, .40, .45, .50, .55, .60, .65, .70, 1.0]
    cut = pd.cut(p, bins)
    t = pd.DataFrame({"p": p, "y": y, "bin": cut}).groupby("bin", observed=True).agg(
        n=("y", "size"), pred=("p", "mean"), act=("y", "mean"))
    for idx, r in t.iterrows():
        print(f"{str(idx):<12}{int(r.n):>6}{r.pred*100:>7.1f}%{r.act*100:>7.1f}%")

    print("\nConfidence bands")
    hi = p >= HIGH_CONF_OVER_PROB
    lo = p <= HIGH_CONF_UNDER_PROB
    for name, mask, hit in [("HIGH Over", hi, y[hi]), ("HIGH Under", lo, 1 - y[lo])]:
        if mask.sum():
            print(f"  {name:<11} n={int(mask.sum()):>4}  hit rate {hit.mean()*100:5.1f}%  "
                  f"(avg model prob {(p[mask].mean() if name=='HIGH Over' else 1-p[mask].mean())*100:.1f}%)")
        else:
            print(f"  {name:<11} n=0")
    print("\nNote: no historical odds are stored, so ROI can't be backtested here.\n"
          "Use tracker.py results over 100+ bets to judge real profitability.")
