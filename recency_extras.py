# recency_extras.py
"""
Follow-up checks after recency_tests.py (needs the files it saves in data/):
  A. How much weight should a last-N "form" signal get on top of the ridge model? (estimate + bootstrap interval)
  B. Does a faster-reacting model win EARLY in the season, when squads/coaches have just changed?
  C. Are the ridge penalty and the xG weight still sensible? (small sweep, walk-forward)

Run:  python recency_tests.py   then   python recency_extras.py
"""
import itertools
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from model import load_data, fit_model, lambdas_from_fit
from recency_tests import ll_goals, p_over_btts, ll_bin, SCALES

rng = np.random.default_rng(7)
test = pd.read_pickle("data/_recency_test.pkl")
lam = np.load("data/_recency_lambdas.npy", allow_pickle=True).item()
hg, ag = test["home_goals"].values, test["away_goals"].values
y_over = (test["total_goals"] > 2.5).astype(int).values


def goals_ll(L, scale=1.06):
    return ll_goals(L[:, 0], L[:, 1], hg, ag, scale)


# ---------------------------------------------------------------- A. best blend weight
print("\nA. Best weight for form on top of the ridge model (lambda = ridge^(1-w) * form^w), 95% bootstrap interval")
base = lam["ridge_hl365"]
for form in ("form_any20_k5", "form_venue10_k5", "form_any10_k5"):
    F = lam[form]

    def loss(w, idx=slice(None)):
        L = np.exp(w * np.log(F[idx]) + (1 - w) * np.log(base[idx]))
        return ll_goals(L[:, 0], L[:, 1], hg[idx], ag[idx], 1.06).mean()

    w_hat = minimize_scalar(loss, bounds=(-0.2, 0.8), method="bounded").x
    boots = []
    for _ in range(150):
        ix = rng.integers(0, len(test), len(test))
        boots.append(minimize_scalar(lambda w: loss(w, ix), bounds=(-0.2, 0.8), method="bounded").x)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"   {form:<18} best w = {w_hat:+.2f}   (95% interval {lo:+.2f} .. {hi:+.2f})   "
          f"log loss at w=0: {loss(0):.4f}  at best: {loss(w_hat):.4f}")
print("   (w = 0 means form adds nothing beyond the ridge model)")

# ---------------------------------------------------------------- B. early season
EURO = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 15, 83}
d = test["date"]
mask = (d >= "2026-08-01") & (d < "2026-10-01") & test["league_id"].isin(EURO)
print(f"\nB. Early season only (Aug-Sep 2026, European leagues): {int(mask.sum())} matches")
ref = goals_ll(lam["ridge_hl365"])[mask.values]
print(f"{'variant':<34}{'goals ll':>10}{'vs current (±95%)':>24}")
for k in ("ridge_hl45", "ridge_hl90", "ridge_hl180", "ridge_hl365", "ridge_hl730", "win240d", "win365d",
          "form_any20_k5", "blend_hl365+any20_k5_w15", "blend_hl365+venue10_k5_w15"):
    v = goals_ll(lam[k])[mask.values]
    dd = ref - v
    print(f"{k:<34}{v.mean():>10.4f}{dd.mean():>+14.4f} ±{1.96 * dd.std(ddof=1) / np.sqrt(len(dd)):.4f}")

# ---------------------------------------------------------------- C. ridge / xG weight sweep
print("\nC. Ridge penalty x xG weight x half-life (walk-forward, refit every 14 days, scale 1.06)")
df = load_data()
start = pd.Timestamp("2026-01-01", tz="UTC")
t = df[df["date"] >= start].reset_index(drop=True)
hgt, agt = t["home_goals"].values, t["away_goals"].values
yo = (t["total_goals"] > 2.5).astype(int).values
rows = []
for hl, rg, xw in itertools.product((365, 730), (5, 10, 20), (0.5, 0.7, 1.0)):
    fit, last, out = None, None, []
    for _, r in t.iterrows():
        if fit is None or (r["date"] - last).days >= 14:
            fit = fit_model(df[df["date"] < r["date"]], now=r["date"], half_life=hl, ridge=rg, xg_weight=xw)
            last = r["date"]
        out.append(lambdas_from_fit(fit, r["home_team_id"], r["away_team_id"], r["league_id"]))
    L = np.array(out)
    po, _ = p_over_btts(L[:, 0], L[:, 1], 1.06)
    rows.append((hl, rg, xw, ll_goals(L[:, 0], L[:, 1], hgt, agt, 1.06).mean(), ll_bin(po, yo).mean()))
res = pd.DataFrame(rows, columns=["half_life", "ridge", "xg_w", "goals_ll", "over_ll"]).sort_values("goals_ll")
cur = res[(res.half_life == 365) & (res.ridge == 10) & (res.xg_w == 0.7)].iloc[0]
print(res.head(8).to_string(index=False))
print(f"\n   current (365 / 10 / 0.7): goals ll {cur.goals_ll:.4f}, Over ll {cur.over_ll:.4f}   "
      f"| best: {res.iloc[0].goals_ll:.4f} / {res.iloc[0].over_ll:.4f}")
