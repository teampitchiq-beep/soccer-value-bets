# corners_backtest.py
"""
Is the corners model calibrated on YOUR data?  Walk-forward: every match is predicted using only matches played before it
(refit every --refit-days), exactly as the Corners tab does it. Read-only: it never changes your data.

    python corners_backtest.py [--line 7.5] [--start YYYY-MM-DD] [--refit-days 14]

Reports, for the chosen line: how often Over happened vs how often the model said it would, log loss against always predicting the base
rate (lower is better), a calibration table by predicted probability, the bias and error of the expected total corners, the same
for the other common lines, and how much the negative-binomial spread helps over plain Poisson.
"""
import argparse
import sys

import numpy as np
import pandas as pd

import config
import match_stats as MS
import shots_data as SD
from model import lambdas_from_fit


def _ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def run(df, store, start=None, refit_days=14, lines=(6.5, 7.5, 8.5, 9.5, 10.5), verbose=True):
    """Walk-forward predictions. Returns a DataFrame with one row per test match, or None if there is too little data."""
    have = store.dropna(subset=["corners_h", "corners_a"])
    m = df[df["event_id"].isin(have.index)].sort_values("date").reset_index(drop=True)
    if len(m) < MS.MIN_FIT_MATCHES + 100:
        return None
    if start is None:
        start = m["date"].iloc[int(0.45 * len(m))]
    test = m[m["date"] >= pd.Timestamp(start)]
    out, mdl, last = [], None, None
    for r in test.itertuples():
        if mdl is None or (r.date - last).days >= refit_days:
            mdl = MS.stat_model(df, store, "corners", r.date)
            last = r.date
            if verbose:
                print(f"  refit {r.date.date()}", file=sys.stderr)
        if mdl is None or r.home_team_id not in mdl["fit"]["tidx"] or r.away_team_id not in mdl["fit"]["tidx"]:
            continue
        mh, ma = lambdas_from_fit(mdl["fit"], r.home_team_id, r.away_team_id, r.league_id)
        row = {"date": r.date, "mu": mh + ma, "k": mdl["k_total"],
               "actual": float(have.loc[r.event_id, "corners_h"] + have.loc[r.event_id, "corners_a"])}
        out.append(row)
    return pd.DataFrame(out) if out else None


def report(res, line):
    from scipy.stats import nbinom, poisson
    mu, k, actual = res["mu"].values, res["k"].values, res["actual"].values
    y = (actual > line).astype(int)

    def p_over(mu_, k_, ln):
        return 1.0 - nbinom.cdf(int(np.floor(ln)), k_, k_ / (k_ + mu_))
    p = p_over(mu, k, line)
    base = np.full(len(y), y.mean())
    print(f"\n=== Over {line:g} corners: {len(y)} test matches ===")
    print(f"Over happened {y.mean() * 100:.1f}% of the time | the model's average probability {p.mean() * 100:.1f}%")
    print(f"expected total corners: model average {mu.mean():.2f} vs actual {actual.mean():.2f} (bias {mu.mean() - actual.mean():+.2f}) | mean absolute error {np.abs(mu - actual).mean():.2f}")
    gain = _ll(base, y) - _ll(p, y)
    print(f"log loss {_ll(p, y).mean():.4f} vs {_ll(base, y).mean():.4f} for always predicting the base rate | gain {gain.mean():+.4f} ± {1.96 * gain.std(ddof=1) / np.sqrt(len(y)):.4f}")
    pp = 1.0 - poisson.cdf(int(np.floor(line)), mu)
    g2 = _ll(pp, y) - _ll(p, y)
    print(f"negative-binomial spread vs plain Poisson: {g2.mean():+.4f} ± {1.96 * g2.std(ddof=1) / np.sqrt(len(y)):.4f} log loss (positive = the spread helps)")
    print("calibration by predicted P(Over):")
    edges = [0, .5, .6, .7, .8, .9, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (p >= lo) & (p < hi)
        if s.sum() >= 25:
            print(f"  {lo:.1f}-{min(hi, 1):.1f}  n={s.sum():4d}  model {p[s].mean() * 100:5.1f}%  actual {y[s].mean() * 100:5.1f}%")
    print("other lines (same predictions):")
    for ln in (6.5, 7.5, 8.5, 9.5, 10.5):
        yl = (actual > ln).astype(int)
        pl = p_over(mu, k, ln)
        print(f"  Over {ln:>4}: actual {yl.mean() * 100:5.1f}% | model {pl.mean() * 100:5.1f}% | log loss {_ll(pl, yl).mean():.4f} vs base {_ll(np.full(len(yl), yl.mean()), yl).mean():.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", type=float, default=float(getattr(config, "CORNERS_LINE", 7.5)))
    ap.add_argument("--start", default=None)
    ap.add_argument("--refit-days", type=int, default=14)
    a = ap.parse_args()
    df, store = SD._matches_table(), MS.load_store()
    res = run(df, store, a.start, a.refit_days)
    if res is None or len(res) < 100:
        sys.exit("Not enough matches with corner stats yet (need roughly 300+). Let the stats loader run for a few days, then try again.")
    report(res, a.line)
