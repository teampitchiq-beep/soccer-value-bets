# tune.py
"""Grid-search LAMBDA_BOOST and OVER_THRESHOLD. Run: python tune.py"""
import pandas as pd
from config import MATCHES_XG_CSV
from model import predict_lambdas, over_under_2_5

df = pd.read_csv(MATCHES_XG_CSV)
df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
df = df.sort_values("date").reset_index(drop=True)

START = 200
print(f"Total matches: {len(df)}   Testing from match {START}\n")

records = []
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
    actual_total = row["home_goals"] + row["away_goals"]
    records.append({
        "lh": max(lh, 0.15),
        "la": max(la, 0.15),
        "actual_over": 1 if actual_total > 2.5 else 0,
    })

res = pd.DataFrame(records)
base_over = res["actual_over"].mean()
lam_sum_mean = (res["lh"] + res["la"]).mean()
lam_sum_std  = (res["lh"] + res["la"]).std()

print(f"Test matches:     {len(res)}")
print(f"Base over rate:   {base_over*100:.1f}%")
print(f"Avg λ sum:        {lam_sum_mean:.2f}   (std: {lam_sum_std:.2f})\n")

print(f"{'boost':>6} {'thresh':>7} {'pred_over%':>11} {'acc':>7} {'edge':>7}")
print("-" * 45)

best = {"acc": 0}
for boost in [0.95, 1.00, 1.05, 1.10, 1.15, 1.20]:
    probs = []
    for _, r in res.iterrows():
        po, _, _ = over_under_2_5(r["lh"] * boost, r["la"] * boost)
        probs.append(po)
    res["po"] = probs

    for thresh in [0.40, 0.45, 0.50, 0.55, 0.60]:
        pred = (res["po"] > thresh).astype(int)
        acc  = (pred == res["actual_over"]).mean()
        edge = acc - base_over
        pr   = pred.mean()
        print(f"{boost:>6.2f} {thresh:>7.2f} {pr*100:>10.1f}% "
              f"{acc*100:>6.1f}% {edge*100:>+6.1f}")
        if acc > best["acc"]:
            best = {"acc": acc, "boost": boost, "thresh": thresh,
                    "pred_over": pr}

print()
print(f"BEST: boost={best['boost']:.2f}  threshold={best['thresh']:.2f}  "
      f"accuracy={best['acc']*100:.1f}%  "
      f"edge={((best['acc']-base_over)*100):+.1f} pts")
print()
print("Update config.py with:")
print(f"  LAMBDA_BOOST   = {best['boost']:.2f}")
print(f"  OVER_THRESHOLD = {best['thresh']:.2f}")
