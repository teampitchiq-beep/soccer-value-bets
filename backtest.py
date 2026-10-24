# backtest.py
"""
Walk-forward backtest. Run: python backtest.py [--start 2025-10-01] [--refit-days 14] [--min-n 60]

Each match is predicted using ONLY earlier matches (model refit every N days).
Reports log loss / Brier / accuracy vs. a constant "base rate" baseline,
a calibration table, hit rates for the confidence bands, and a PER-LEAGUE
breakdown showing where the model actually beats the base rate.

Log loss is the number to watch: it must beat the baseline to prove the
probabilities carry real information. (Accuracy alone is misleading.)
"""
import argparse
import numpy as np
import pandas as pd

from config import LAMBDA_BOOST, HIGH_CONF_OVER_PROB, HIGH_CONF_UNDER_PROB, league_name
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


def per_match_ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def metrics(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return {"logloss": -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)),
            "brier": np.mean((p - y) ** 2),
            "acc": np.mean((p > 0.5) == y)}


def league_breakdown(df, res, p, y, start, min_n):
    """Per-league comparison of the model vs that league's own base rate."""
    train = df[df["date"] < start]
    global_rate = (train["total_goals"] > 2.5).mean()
    lg = df.loc[res.index, "league_id"].values

    rows = []
    for lid in pd.unique(lg):
        mask = lg == lid
        n = int(mask.sum())
        tr = train[train["league_id"] == lid]
        base = (tr["total_goals"] > 2.5).mean() if len(tr) >= 30 else global_rate
        ll_model = per_match_ll(p[mask], y[mask])
        ll_base = per_match_ll(np.full(n, base), y[mask])
        diff = ll_base - ll_model                      # positive = model better
        gain = diff.mean()
        se = diff.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
        if n < min_n:
            verdict = "too few"
        elif gain - 1.96 * se > 0:
            verdict = "EDGE"
        elif gain + 1.96 * se < 0:
            verdict = "WORSE"
        else:
            verdict = "unclear"
        rows.append({"league": league_name(lid), "n": n,
                     "actual": y[mask].mean() * 100, "pred": p[mask].mean() * 100,
                     "gain": gain, "se": se, "acc": ((p[mask] > 0.5) == y[mask]).mean() * 100,
                     "verdict": verdict})
    out = pd.DataFrame(rows).sort_values("gain", ascending=False)

    print("\nPer-league (log-loss gain vs that league's own base rate; + = model better)")
    print(f"{'league':<20}{'n':>5}{'over%':>7}{'pred%':>7}{'gain':>9}{'±95%':>8}  verdict")
    for r in out.itertuples():
        print(f"{r.league[:19]:<20}{r.n:>5}{r.actual:>6.1f}%{r.pred:>6.1f}%"
              f"{r.gain:>+9.4f}{1.96 * r.se:>8.4f}  {r.verdict}")
    print("\nverdict: EDGE = gain clearly above 0 | unclear = could be luck | "
          "WORSE = clearly below 0 | too few = under --min-n matches")
    weak = out[out["verdict"] == "WORSE"]["league"].tolist()
    if weak:
        print(f"Consider dropping (model worse than base rate): {', '.join(weak)}")
    unclear = out[out["verdict"] == "unclear"]["league"].tolist()
    if unclear:
        print(f"No proven edge yet (re-check with more data): {', '.join(unclear)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-10-01")
    ap.add_argument("--refit-days", type=int, default=14)
    ap.add_argument("--min-n", type=int, default=60, help="min matches for a per-league verdict")
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

    league_breakdown(df, res, p, y, start, args.min_n)

    print("\nNote: no historical odds are stored, so ROI can't be backtested here.\n"
          "Use tracker.py results over 100+ bets to judge real profitability.\n")
