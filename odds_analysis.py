# odds_analysis.py
"""
Backtest the model against bookmaker Over/Under 2.5 odds.

  python odds_analysis.py                  # uses data/odds_log.csv (your own pre-kickoff snapshots)  <- valid
  python odds_analysis.py --max-hours 6    # only snapshots taken within 6h of kickoff (near-closing)
  python odds_analysis.py --source history # uses data/odds_history.csv (Bzzoiro backfill)
                                           #  WARNING: those odds were mostly updated AFTER kickoff
                                           #  and are NOT valid for judging the model.

Reports: odds timing, log loss (market vs model vs blends), flat-stake ROI of
your value-bet rule, ROI by EV band.
"""
import argparse
import numpy as np
import pandas as pd

from config import MIN_EV, MAX_EV, MIN_ODDS, LAMBDA_BOOST
from model import load_data, over_under_2_5


def ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def summarize(label, win, odds):
    n = len(win)
    if n == 0:
        print(f"  {label:<30} n=0")
        return
    profit = np.where(win, odds - 1, -1.0)
    roi = profit.mean()
    se = profit.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    print(f"  {label:<30} n={n:>4}  hit={win.mean()*100:5.1f}%  "
          f"ROI={roi*100:+6.1f}%  (95% CI {100*(roi-1.96*se):+.0f}% to {100*(roi+1.96*se):+.0f}%)")


# ---------------------------------------------------------------- data sources
def from_log(max_hours):
    log = pd.read_csv("data/odds_log.csv")
    log["logged_at"] = pd.to_datetime(log["logged_at"], utc=True)
    log["kickoff"] = pd.to_datetime(log["kickoff"], utc=True)
    log = log[log["logged_at"] < log["kickoff"]]
    if max_hours is not None:
        log = log[log["hours_to_kickoff"] <= max_hours]
    # last pre-kickoff snapshot per fixture = best available "closing" price
    snap = log.sort_values("logged_at").drop_duplicates("event_id", keep="last")
    df = load_data()
    res = df[["event_id", "total_goals"]]
    m = snap.merge(res, on="event_id", how="inner")
    pending = snap["event_id"].nunique() - len(m)
    m["y"] = (m["total_goals"] > 2.5).astype(int)
    m["p_model"] = m["p_over"]
    m["over_25"], m["under_25"] = m["over_odds"], m["under_odds"]
    m["pre"] = True
    print(f"Logged fixtures: {snap['event_id'].nunique()}   with results: {len(m)}   "
          f"still pending / not yet in data: {pending}")
    if len(m):
        h = m["hours_to_kickoff"]
        print(f"1) Snapshot timing: median {h.median():.1f}h before kickoff "
              f"(min {h.min():.1f}h, max {h.max():.1f}h). Odds are genuinely pre-match.")
    return m


def from_history():
    from backtest import walk_forward
    odds = pd.read_csv("data/odds_history.csv")
    odds = odds.dropna(subset=["over_25", "under_25"]).drop_duplicates("event_id", keep="last")
    df = load_data()
    start = df[df["event_id"].isin(odds["event_id"])]["date"].min()
    res = walk_forward(df, start, refit_days=7)
    m = df.loc[res.index, ["event_id", "date", "total_goals"]].join(res[["lh", "la"]])
    m = m.merge(odds, on="event_id", how="inner")
    m["y"] = (m["total_goals"] > 2.5).astype(int)
    m["p_model"] = [over_under_2_5(a * LAMBDA_BOOST, b * LAMBDA_BOOST)[0]
                    for a, b in zip(m["lh"], m["la"])]
    if "last_change_at" in m and m["last_change_at"].notna().any():
        t = pd.to_datetime(m["last_change_at"], errors="coerce", utc=True)
        hrs = (t - m["date"]).dt.total_seconds() / 3600
        print(f"1) Odds timing: median {hrs.median():+.1f}h vs kickoff; "
              f"{(hrs > 0).mean()*100:.0f}% changed AFTER kickoff -> results below are NOT reliable.")
        m["pre"] = (hrs <= 0).fillna(False)
    else:
        m["pre"] = True
    return m


# ---------------------------------------------------------------- analysis
def analyze(m):
    y = m["y"].values
    inv_o, inv_u = 1 / m["over_25"], 1 / m["under_25"]
    m["p_mkt"] = inv_o / (inv_o + inv_u)
    print(f"\n2) Log loss on {len(m)} matches (lower = better).  "
          f"avg bookmaker margin {(inv_o + inv_u - 1).mean()*100:.1f}%   actual over rate {y.mean()*100:.1f}%")
    print(f"   {'constant base rate':<26}{ll(np.full(len(y), y.mean()), y):.4f}")
    print(f"   {'market (no-vig)':<26}{ll(m['p_mkt'].values, y):.4f}")
    print(f"   {'model':<26}{ll(m['p_model'].values, y):.4f}")
    best_w, best_ll = 0, 9
    for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
        l = ll((w * m["p_model"] + (1 - w) * m["p_mkt"]).values, y)
        print(f"   blend {w*100:>3.0f}% model / {100-w*100:>3.0f}% mkt   {l:.4f}")
        if l < best_ll:
            best_w, best_ll = w, l
    print(f"   -> best blend here: {best_w*100:.0f}% model (hint only, not a setting)")

    print(f"\n3) Flat 1-unit ROI. Rule: EV in [{MIN_EV*100:.0f}%, {MAX_EV*100:.0f}%], odds >= {MIN_ODDS}")
    for label, probs in [("MODEL probabilities", m["p_model"]),
                         ("BLEND 50/50 probabilities", 0.5 * m["p_model"] + 0.5 * m["p_mkt"])]:
        print(f"\n  {label}")
        p_o = probs.values
        ev_o = p_o * m["over_25"].values - 1
        ev_u = (1 - p_o) * m["under_25"].values - 1
        take_over = ev_o >= ev_u
        ev = np.where(take_over, ev_o, ev_u)
        o_pick = np.where(take_over, m["over_25"].values, m["under_25"].values)
        win = np.where(take_over, y == 1, y == 0)
        for sub_label, sub in [("all", np.ones(len(m), bool)), ("pre-kickoff odds only", m["pre"].values)]:
            sel = sub & (ev >= MIN_EV) & (ev <= MAX_EV) & (o_pick >= MIN_ODDS)
            summarize(f"rule bets, {sub_label}", win[sel], o_pick[sel])
        print("  EV bands:")
        for lo, hi in [(-1, 0), (0, 0.03), (0.03, 0.08), (0.08, 0.15), (0.15, 9)]:
            sel = (ev > lo) & (ev <= hi) & (o_pick >= MIN_ODDS)
            name = f"EV {lo*100:+.0f}% to {hi*100:+.0f}%" if hi < 9 else "EV > +15%"
            summarize(name, win[sel], o_pick[sel])
    print("\nBenchmarks (cost of the bookmaker margin):")
    summarize("always Over", (y == 1), m["over_25"].values)
    summarize("always Under", (y == 0), m["under_25"].values)
    print("\nRule of thumb: ROI intervals that include 0% = no proven edge. Need ~300+ rule bets\n"
          "for the interval to narrow to roughly +/-10%.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["log", "history"], default="log")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="log source: only snapshots within N hours of kickoff")
    a = ap.parse_args()

    if a.source == "log":
        try:
            m = from_log(a.max_hours)
        except FileNotFoundError:
            print("No data/odds_log.csv yet. Run: python odds_logger.py --watch")
            return
    else:
        print("WARNING: history odds were mostly updated after kickoff; don't trust conclusions.\n")
        m = from_history()

    if len(m) < 30:
        print(f"\nOnly {len(m)} finished fixtures with odds so far - need at least ~100 "
              f"(ideally 300+). Keep logging, refresh data, and re-run later.")
        return
    analyze(m)


if __name__ == "__main__":
    main()
