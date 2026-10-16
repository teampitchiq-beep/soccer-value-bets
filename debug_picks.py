# debug_picks.py
"""Trace every upcoming match through the pick pipeline to see where it fails."""
import pandas as pd
from datetime import datetime, timezone, timedelta

from config import (
    LEAGUES_TO_FETCH, LAMBDA_BOOST, MIN_ODDS, MIN_EV, MAX_EV,
    HIGH_CONF_OVER_PROB, HIGH_CONF_UNDER_PROB, league_name,
)
from model import load_data, predict_lambdas, over_under_2_5
from scan import fetch_upcoming, fetch_event_odds, build_appearance_index

df = load_data()
df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
appearance_counts = build_appearance_index(df)

print(f"Training data: {len(df)} matches, {df['league_id'].nunique()} leagues\n")

upcoming = fetch_upcoming(days=2)
print(f"Upcoming matches: {len(upcoming)}\n")

MIN_APPS = 15
stats = {"no_history": 0, "low_conf": 0, "no_odds": 0, "low_odds": 0,
         "bad_ev": 0, "PASS": 0}

print(f"{'Match':<45} {'Apps':>8} {'Model%':>7} {'Odds':>6} {'EV%':>7}  Reason")
print("-" * 100)

for m in upcoming:
    label = f"{m['home_team'][:20]} vs {m['away_team'][:20]}"
    hk = (int(m["league_id"]), int(m["home_team_id"]))
    ak = (int(m["league_id"]), int(m["away_team_id"]))
    ha = appearance_counts.get(hk, 0)
    aa = appearance_counts.get(ak, 0)

    if ha < MIN_APPS or aa < MIN_APPS:
        stats["no_history"] += 1
        print(f"{label:<45} {ha:>3}/{aa:<3}  {'':>7} {'':>6} {'':>7}  no history")
        continue

    try:
        lh, la = predict_lambdas(df, m["home_team_id"], m["away_team_id"], m["league_id"])
    except Exception as e:
        print(f"{label:<45}  ERR: {e}")
        continue

    lh *= LAMBDA_BOOST
    la *= LAMBDA_BOOST
    po, pu, _ = over_under_2_5(lh, la)

    if not (po >= HIGH_CONF_OVER_PROB or po <= HIGH_CONF_UNDER_PROB):
        stats["low_conf"] += 1
        print(f"{label:<45} {ha:>3}/{aa:<3}  {po*100:>6.1f}  {'':>6} {'':>7}  not high-conf")
        continue

    odds = fetch_event_odds(m["event_id"])
    if not odds or (odds["over_25"] is None and odds["under_25"] is None):
        stats["no_odds"] += 1
        print(f"{label:<45} {ha:>3}/{aa:<3}  {po*100:>6.1f}  {'none':>6} {'':>7}  no odds")
        continue

    if po >= pu:
        pick, prob, book = "Over", po, odds["over_25"]
    else:
        pick, prob, book = "Under", pu, odds["under_25"]

    if book is None or book < MIN_ODDS:
        stats["low_odds"] += 1
        print(f"{label:<45} {ha:>3}/{aa:<3}  {po*100:>6.1f}  {book or 0:>6.2f} {'':>7}  low odds")
        continue

    ev = prob * book - 1

    if not (MIN_EV <= ev <= MAX_EV):
        stats["bad_ev"] += 1
        print(f"{label:<45} {ha:>3}/{aa:<3}  {po*100:>6.1f}  {book:>6.2f} {ev*100:>+6.1f}  EV out of range")
        continue

    stats["PASS"] += 1
    print(f"{label:<45} {ha:>3}/{aa:<3}  {po*100:>6.1f}  {book:>6.2f} {ev*100:>+6.1f}  ✅ PASS ({pick})")

print("\n" + "=" * 100)
print("FILTER BREAKDOWN:")
for k, v in stats.items():
    print(f"  {k:<15} {v}")
