# corners_backtest.py
"""
Walk-forward backtest of the corners model (Over/Under the line, default 8.5).
Each match is predicted using ONLY earlier matches.

    python corners_backtest.py                 # full report
    python corners_backtest.py --start 2025-10-01 --refit-days 14
    python corners_backtest.py --tune          # try a few ridge / half-life settings

Needs data/corners.csv  (python corners_data.py).  Output:
  * log loss vs always predicting the base rate   (must be lower to prove an edge)
  * calibration table, confidence-tier hit rates, per-league verdicts
  * recommended CORNERS_SCALE and CORNERS_DISPERSION values for config.py
"""
import argparse
import numpy as np
import pandas as pd

from config import league_name
from scipy.optimize import minimize
from corners_model import (LINE, RIDGE, HALF_LIFE, SCALE, DISPERSION, HIGH_PROB, MEDIUM_PROB, CAL_A, CAL_B,
                           load_training, fit_corners, expected_corners, estimate_phi, prob_over, calibrate)


def walk_forward(df, start, refit_days=14, ridge=None, half_life=None):
    test = df[df["date"] >= start]
    fit, last, phi, rows = None, None, 1.0, []
    for r in test.itertuples():
        if fit is None or (r.date - last).days >= refit_days:
            hist = df[df["date"] < r.date]
            fit = fit_corners(hist, now=r.date, ridge=ridge, half_life=half_life)
            phi = float(DISPERSION) if DISPERSION else estimate_phi(fit, hist, scale=1.0)
            last = r.date
        mh, ma = expected_corners(fit, r.home_team_id, r.away_team_id, r.league_id, scale=1.0)
        rows.append((r.Index, r.league_id, mh, ma, r.home_corners, r.away_corners, phi))
    return pd.DataFrame(rows, columns=["idx", "league_id", "mh", "ma", "hc", "ac", "phi"]).set_index("idx")


def probs(res, scale, phi=None, calibrated=True):
    mu = (res["mh"] + res["ma"]).values * scale
    ph = res["phi"].values if phi is None else np.full(len(res), phi)
    return np.array([prob_over(m, f, calibrated=calibrated) for m, f in zip(mu, ph)])


def fit_platt(p, y):
    """Best (a, b) for p' = sigmoid(a*logit(p)+b) by log loss."""
    p = np.clip(p, 1e-4, 1 - 1e-4)
    z = np.log(p / (1 - p))
    f = lambda w: ll_each(1 / (1 + np.exp(-(w[0] * z + w[1]))), y).mean()
    w = minimize(f, [1.0, 0.0], method="Nelder-Mead").x
    return float(w[0]), float(w[1])


def ll_each(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-10-01")
    ap.add_argument("--refit-days", type=int, default=14)
    ap.add_argument("--min-n", type=int, default=60)
    ap.add_argument("--tune", action="store_true")
    a = ap.parse_args()

    df = load_training()
    if df is None:
        print("No corner data yet. Run:  python corners_data.py")
        return
    start = pd.Timestamp(a.start, tz="UTC")
    if (df["date"] >= start).sum() < 300:                      # not enough history after the default start
        start = df["date"].quantile(0.6)
        print(f"(few matches after {a.start}; using {start.date()} as the test start)")
    train = df[df["date"] < start]
    print(f"Matches with corners: {len(df)}   training before test: {len(train)}   test: {int((df['date']>=start).sum())}")

    if a.tune:
        print(f"\n{'ridge':>7}{'half-life':>11}{'raw':>10}{'recalibrated':>14}")
        best = None
        for rg in (25, 50, 100, 200):
            for hl in (180, 365, 730):
                res = walk_forward(df, start, a.refit_days, ridge=rg, half_life=hl)
                y = ((res["hc"] + res["ac"]) > LINE).astype(int).values
                p = probs(res, SCALE, calibrated=False)
                ca, cb = fit_platt(p, y)
                l_raw, l_cal = ll_each(p, y).mean(), ll_each(calibrate(p, ca, cb), y).mean()
                print(f"{rg:>7}{hl:>11}{l_raw:>10.4f}{l_cal:>14.4f}", flush=True)
                if best is None or l_cal < best[0]:
                    best = (l_cal, rg, hl, ca, cb)
        print("\n'recalibrated' = after the best probability calibration for that setting.")
        print("Lowest = best. Differences under ~0.001 are noise.")
        print(f"\nBest: CORNERS_RIDGE = {best[1]}, CORNERS_HALF_LIFE_DAYS = {best[2]}.0")
        print("Set those in config.py, then run  python corners_backtest.py  again to get CORNERS_CAL_A / CORNERS_CAL_B.")
        return

    res = walk_forward(df, start, a.refit_days)
    T = (res["hc"] + res["ac"]).values
    y = (T > LINE).astype(int)
    p_raw = probs(res, SCALE, calibrated=False)
    p = calibrate(p_raw)                                 # as configured (CORNERS_CAL_A/B; default = unchanged)
    base = ((train["home_corners"] + train["away_corners"]) > LINE).mean()

    ll_m, ll_b = ll_each(p, y).mean(), ll_each(np.full(len(y), base), y).mean()
    mu = (res["mh"] + res["ma"]).values * SCALE
    mae_m, mae_b = np.abs(T - mu).mean(), np.abs(T - (train["home_corners"] + train["away_corners"]).mean()).mean()
    print(f"\nLine {LINE:g}   actual Over rate {y.mean()*100:.1f}%   mean predicted {p.mean()*100:.1f}%   base rate (training) {base*100:.1f}%")
    print(f"{'':<12}{'logloss':>9}{'acc':>8}")
    print(f"{'baseline':<12}{ll_b:>9.4f}{(((base>0.5)==y).mean())*100:>7.1f}%")
    print(f"{'model':<12}{ll_m:>9.4f}{((p>0.5)==y).mean()*100:>7.1f}%")
    print(f"=> logloss improvement vs baseline: {ll_b-ll_m:+.4f} ({'GOOD' if ll_m < ll_b else 'NO EDGE'})")
    print(f"Total-corners error: model MAE {mae_m:.2f} vs constant-average MAE {mae_b:.2f} "
          f"({'model better' if mae_m < mae_b else 'no better than the average'})")

    print("\nCalibration (predicted vs actual over rate)")
    print(f"{'bin':<12}{'n':>6}{'pred':>8}{'actual':>8}")
    cut = pd.cut(p, [0, .4, .5, .6, .65, .7, .75, .8, .85, 1.0])
    t = pd.DataFrame({"p": p, "y": y, "bin": cut}).groupby("bin", observed=True).agg(n=("y", "size"), pred=("p", "mean"), act=("y", "mean"))
    for idx, r in t.iterrows():
        print(f"{str(idx):<12}{int(r.n):>6}{r.pred*100:>7.1f}%{r.act*100:>7.1f}%")

    side_prob = np.where(p >= .5, p, 1 - p)
    hit = np.where(p >= .5, y == 1, y == 0)
    print(f"\nConfidence tiers (probability of the chosen side)")
    for name, lo, hi in [("HIGH", HIGH_PROB, 1.01), ("MEDIUM", MEDIUM_PROB, HIGH_PROB), ("LOW", 0, MEDIUM_PROB)]:
        mk = (side_prob >= lo) & (side_prob < hi)
        if mk.sum():
            print(f"  {name:<7}n={int(mk.sum()):>5}  hit {hit[mk].mean()*100:5.1f}%   avg model prob {side_prob[mk].mean()*100:.1f}%")

    # recommendations (out-of-sample)
    phi_oos = float(max(1.0, np.mean((T - mu / SCALE) ** 2 / (mu / SCALE))))
    best_s, best_l = SCALE, 9
    for s in np.arange(0.90, 1.101, 0.02):
        l = ll_each(probs(res, s, phi_oos, calibrated=False), y).mean()
        if l < best_l:
            best_s, best_l = round(float(s), 2), l
    print(f"\nSpread of total corners (variance/mean): out-of-sample {phi_oos:.2f}, "
          f"in use {'config ' + str(DISPERSION) if DISPERSION else 'auto (' + format(res['phi'].mean(), '.2f') + ' avg)'}")
    print(f"Best scale with that spread: {best_s}  (log loss {best_l:.4f}; small change = scale is not the problem)")

    # probability calibration: honest check = fit on the first half of the test, judge on the second half
    ca, cb = fit_platt(p_raw, y)
    half = len(y) // 2
    a1, b1 = fit_platt(p_raw[:half], y[:half])
    raw2, cal2 = ll_each(p_raw[half:], y[half:]).mean(), ll_each(calibrate(p_raw[half:], a1, b1), y[half:]).mean()
    print(f"\nProbability calibration  p' = sigmoid(A*logit(p)+B):  A={ca:.2f}  B={cb:+.2f}  (A<1 means overconfident)")
    print(f"  honest check (fit on 1st half of test, judged on 2nd half): "
          f"log loss {raw2:.4f} -> {cal2:.4f}  ({'better' if cal2 < raw2 else 'no better'})")
    print(f"  full-test log loss: raw {ll_each(p_raw, y).mean():.4f} -> calibrated {ll_each(calibrate(p_raw, ca, cb), y).mean():.4f}")
    print("Optional config.py lines (recommended - they also fix the confidence tiers):")
    print(f"  CORNERS_CAL_A = {ca:.2f}")
    print(f"  CORNERS_CAL_B = {cb:.2f}")
    print("  (skip CORNERS_SCALE / CORNERS_DISPERSION unless the log-loss gain above is larger than ~0.001)")

    # per league
    lg = res["league_id"].values
    print(f"\nPer-league (log-loss gain vs that league's own base rate; + = model better)")
    print(f"{'league':<20}{'n':>5}{'over%':>7}{'pred%':>7}{'gain':>9}{'±95%':>8}  verdict")
    rows = []
    for lid in pd.unique(lg):
        mk = lg == lid
        n = int(mk.sum())
        tr = train[train["league_id"] == lid]
        b = ((tr["home_corners"] + tr["away_corners"]) > LINE).mean() if len(tr) >= 30 else base
        d = ll_each(np.full(n, b), y[mk]) - ll_each(p[mk], y[mk])
        gain, se = d.mean(), (d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan)
        v = "too few" if n < a.min_n else ("EDGE" if gain - 1.96 * se > 0 else ("WORSE" if gain + 1.96 * se < 0 else "unclear"))
        rows.append((league_name(lid), n, y[mk].mean() * 100, p[mk].mean() * 100, gain, se, v))
    for r in sorted(rows, key=lambda r: -r[4]):
        print(f"{r[0][:19]:<20}{r[1]:>5}{r[2]:>6.1f}%{r[3]:>6.1f}%{r[4]:>+9.4f}{1.96*r[5]:>8.4f}  {r[6]}")
    print("\nNo corner odds are available from the API, so profit can't be backtested - this measures accuracy only.")


if __name__ == "__main__":
    main()
