# tune.py
"""
Grid-search model hyper-parameters by walk-forward LOG LOSS (not accuracy).
Run: python tune.py [--start 2025-10-01]

Searches HALF_LIFE_DAYS x RIDGE x XG_WEIGHT; for each, picks the best
LAMBDA_BOOST (calibration scale). Prints the config lines to paste.

Tip: results within ~0.001 log loss are noise — prefer the simpler/default
setting unless something clearly wins.
"""
import argparse
import itertools
import numpy as np
import pandas as pd

from model import load_data
from backtest import walk_forward, p_over, metrics

ap = argparse.ArgumentParser()
ap.add_argument("--start", default="2025-10-01")
ap.add_argument("--refit-days", type=int, default=14)
args = ap.parse_args()

df = load_data()
start = pd.Timestamp(args.start, tz="UTC")
y_all = None

HALF_LIVES = [180, 365, 730]
RIDGES     = [5, 10, 20]
XG_WEIGHTS = [0.5, 0.7, 1.0]
SCALES     = [1.00, 1.03, 1.06, 1.09, 1.12]

print(f"{'half_life':>9} {'ridge':>6} {'xg_w':>5} {'scale':>6} {'logloss':>9} {'acc':>7}")
print("-" * 50)
best = None
for hl, rg, xw in itertools.product(HALF_LIVES, RIDGES, XG_WEIGHTS):
    res = walk_forward(df, start, args.refit_days, half_life=hl, ridge=rg, xg_weight=xw)
    y = res["over"].values
    scored = [(s, metrics(p_over(res, s), y)) for s in SCALES]
    s, m = min(scored, key=lambda t: t[1]["logloss"])
    print(f"{hl:>9} {rg:>6} {xw:>5.1f} {s:>6.2f} {m['logloss']:>9.4f} {m['acc']*100:>6.1f}%")
    if best is None or m["logloss"] < best[0]:
        best = (m["logloss"], hl, rg, xw, s, m["acc"])

ll, hl, rg, xw, s, acc = best
print(f"\nBEST: logloss={ll:.4f}  accuracy={acc*100:.1f}%\n")
print("Update config.py with:")
print(f"  HALF_LIFE_DAYS = {hl}")
print(f"  RIDGE          = {rg}")
print(f"  XG_WEIGHT      = {xw}")
print(f"  LAMBDA_BOOST   = {s:.2f}")
