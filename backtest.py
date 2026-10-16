# backtest.py
"""Confidence-band analysis. Run: python backtest.py"""
import pandas as pd
from config import MATCHES_XG_CSV, LAMBDA_BOOST
from model import predict_lambdas, over_under_2_5

df = pd.read_csv(MATCHES_XG_CSV)
df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
df = df.sort_values("date").reset_index(drop=True)

START = 200
rows = []
for i in range(START, len(df)):
    row = df.iloc[i]
    history = df.iloc[:i]
    try:
        lh, la = predict_lambdas(history,
                                 row["home_team_id"],
                                 row["away_team_id"],
                                 row["league_id"])
    except Exception:
        continue
    lh *= LAMBDA_BOOST
    la *= LAMBDA_BOOST
    po, pu, _ = over_under_2_5(lh, la)
    actual_total = row["home_goals"] + row["away_goals"]
    rows.append({"po": po, "is_over": actual_total > 2.5})

res = pd.DataFrame(rows)
base = res["is_over"].mean()

print(f"Total test matches: {len(res)}")
print(f"Base over rate:     {base*100:.1f}%\n")
print(f"{'Band':<16} {'n':>6} {'hit rate':>10}")
print("-" * 40)

bands = [
    ("Over > 85%",  lambda r: r["po"] > 0.85,  True),
    ("Over 75-85%", lambda r: (r["po"] > 0.75) & (r["po"] <= 0.85), True),
    ("Over 60-75%", lambda r: (r["po"] > 0.60) & (r["po"] <= 0.75), True),
    ("Fade 40-60%", lambda r: (r["po"] > 0.40) & (r["po"] <= 0.60), None),
    ("Under 25-40%",lambda r: (r["po"] > 0.25) & (r["po"] <= 0.40), False),
    ("Under 15-25%",lambda r: (r["po"] > 0.15) & (r["po"] <= 0.25), False),
    ("Under < 15%", lambda r: r["po"] <= 0.15, False),
]
for name, cond, side in bands:
    sub = res[cond(res)]
    n = len(sub)
    if n == 0:
        print(f"{name:<16} {n:>6}"); continue
    if side is True:    hit = sub["is_over"].mean()
    elif side is False: hit = (~sub["is_over"]).mean()
    else:
        print(f"{name:<16} {n:>6} {'—':>10}"); continue
    print(f"{name:<16} {n:>6} {hit*100:>9.1f}%")

hc = res[(res["po"] > 0.75) | (res["po"] < 0.25)].copy()
if len(hc):
    hc["pred_over"] = hc["po"] > 0.75
    hc["correct"]   = hc["pred_over"] == hc["is_over"]
    hit = hc["correct"].mean()
    print(f"\n{'='*40}")
    print(f"HIGH-CONFIDENCE BETS ONLY")
    print(f"  Matches:   {len(hc)}")
    print(f"  Hit rate:  {hit*100:.1f}%")
    print(f"  Edge:      {(hit-base)*100:+.1f} pts over baseline")
    print(f"{'='*40}")
