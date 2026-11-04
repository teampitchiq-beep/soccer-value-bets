# shots_tests.py
"""
Walk-forward tests on SHOTS (needs data/shots_cache.csv from the Shots tab / shots_data.py).

Every match from --start onward is predicted using ONLY earlier matches (refit every --refit-days).
Only matches with known shots are used, and test matches need >= --min-window earlier home matches for the home team
and away matches for the away team (so the "last 10" windows are real).

  1. HOW MUCH HISTORY PREDICTS SHOTS BEST?  Shots are ~12 a match, so they are far less noisy than goals (~1.4).
     Compared: league average, the Shots tab's own number (plain last-10 home / away average), opponent-adjusted
     last-10 / last-20 form, and the ridge-Poisson model (the goals model, fitted on shots) with several half-lives.
     Metric: Poisson log loss and MSE of the predicted shots. Lower = better.
  2. DO SHOTS HELP PREDICT GOALS?  Shots-based expected goals (predicted shots x league goals-per-shot) alone and
     blended with the current goals model. Metrics: goals / Over 2.5 / BTTS log loss, with paired 95% intervals.
  3. IS THE SHOTS TAB'S AVG USEFUL?  Over / BTTS rate by quartile of Avg, and whether Avg adds anything once the
     goals model's probability is known (logistic regression, bootstrap interval).

Run:  python shots_tests.py [--start 2026-01-01] [--refit-days 14] [--min-window 5]
"""
import argparse
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import spearmanr

from model import load_data, fit_model, lambdas_from_fit, XG_WEIGHT
from recency_tests import FormIndex, form_lambdas, ll_goals, ll_bin, p_over_btts, SCALES
import shots_data as SD

rng = np.random.default_rng(11)


def paired(ref, v):
    d = ref - v                                  # > 0 means v is better than ref
    return d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(len(d))


# ------------------------------------------------------------------ data
def load_joined():
    df = load_data()
    cache = SD.load_cache()
    if not cache:
        sys.exit("data/shots_cache.csv is empty: fill it first (python shots_data.py --all).")
    df["event_id"] = df["event_id"].astype(int)
    df["home_shots"] = df["event_id"].map(lambda e: cache.get(e, (np.nan, np.nan))[0])
    df["away_shots"] = df["event_id"].map(lambda e: cache.get(e, (np.nan, np.nan))[1])
    n = int(df["home_shots"].notna().sum())
    print(f"Shots known for {n} of {len(df)} matches ({100 * n / len(df):.0f}%)")
    return df


def wavg(idx, key, t, n, decay, default):
    """Recency-weighted average of a team's own shots over its last n venue matches (newest weight 1, then decay, decay^2 ...)."""
    src = idx.t.get(key)
    if src is None:
        return default
    dates, gf, _ga = src
    i = np.searchsorted(dates, np.datetime64(t.tz_convert("UTC").tz_localize(None)), side="left")
    seg = gf[max(0, i - n):i][::-1]                                   # newest first
    if len(seg) == 0:
        return default
    w = decay ** np.arange(len(seg))
    return float((seg * w).sum() / w.sum())


# ------------------------------------------------------------------ walk-forward
SHOT_RIDGE = {"ridge_hl90": (90, 40), "ridge_hl180": (180, 40), "ridge_hl365": (365, 40), "ridge_hl730": (730, 40),
              "ridge_hl365_r10": (365, 10), "ridge_hl365_r100": (365, 100)}


def run(df, start, refit_days, min_window):
    sdf = df.dropna(subset=["home_shots", "away_shots"]).copy()
    sdf["yh"], sdf["ya"] = sdf["home_shots"], sdf["away_shots"]
    idx = FormIndex(sdf)
    # a copy where the "xG" columns hold shots, so the existing ridge model can be fitted on shots (target = 100% shots)
    shots_view = sdf.copy()
    shots_view["home_xg"], shots_view["away_xg"] = sdf["home_shots"], sdf["away_shots"]
    shots_view["home_goals"], shots_view["away_goals"] = sdf["home_shots"], sdf["away_shots"]

    test = sdf[sdf["date"] >= start].reset_index(drop=True)
    names = (["league_avg", "tab_metric", "tab_w90", "tab_w80", "tab_w90_n15", "form_venue10", "form_venue10_k5", "form_any10_k5", "form_venue20_k5",
              "form_any20_k5"] + list(SHOT_RIDGE))
    S = {n: [] for n in names}                   # predicted shots (home, away)
    G = {"goals_ridge": [], "conv_h": [], "conv_a": []}
    keep, tab = [], []

    gfit, sfits, last = None, None, None
    for k, r in test.iterrows():
        t = r["date"]
        if gfit is None or (t - last).days >= refit_days:
            gfit = fit_model(df[df["date"] < t], now=t)                       # the app's goals model (all matches)
            past_s = shots_view[shots_view["date"] < t]
            sfits = {n: fit_model(past_s, now=t, half_life=hl, ridge=rg, xg_weight=1.0) for n, (hl, rg) in SHOT_RIDGE.items()}
            year = sdf[(sdf["date"] < t) & (sdf["date"] >= t - pd.Timedelta(days=365))]
            lg = year.groupby("league_id").agg(h=("yh", "mean"), a=("ya", "mean"),
                                                gh=("home_goals", "sum"), ga=("away_goals", "sum"),
                                                sh=("home_shots", "sum"), sa=("away_shots", "sum"))
            glob = (year["yh"].mean(), year["ya"].mean(), year["home_goals"].sum() / year["home_shots"].sum(),
                    year["away_goals"].sum() / year["away_shots"].sum())
            last = t
            print(f"  refit {t.date()}", file=sys.stderr)
        lid = r["league_id"]
        if lid in lg.index and lg.loc[lid, "sh"] > 300:
            lgh, lga = lg.loc[lid, "h"], lg.loc[lid, "a"]
            ch, ca = lg.loc[lid, "gh"] / lg.loc[lid, "sh"], lg.loc[lid, "ga"] / lg.loc[lid, "sa"]
        else:
            lgh, lga, ch, ca = glob
        hh = idx.last(("h", int(r["home_team_id"])), t, 10, True)
        aa = idx.last(("a", int(r["away_team_id"])), t, 10, True)
        ok = bool(hh and aa and hh[2] >= min_window and aa[2] >= min_window)
        keep.append(ok)
        S["league_avg"].append((lgh, lga))
        S["tab_metric"].append((hh[0] / hh[2] if hh else lgh, aa[0] / aa[2] if aa else lga))   # exactly the Shots tab numbers
        for nm, n_, dec in (("tab_w90", 10, 0.9), ("tab_w80", 10, 0.8), ("tab_w90_n15", 15, 0.9)):   # recency-weighted Avg
            S[nm].append((wavg(idx, ("h", int(r["home_team_id"])), t, n_, dec, lgh),
                          wavg(idx, ("a", int(r["away_team_id"])), t, n_, dec, lga)))
        S["form_venue10"].append(form_lambdas(idx, r, 10, 0, True, lgh, lga))
        S["form_venue10_k5"].append(form_lambdas(idx, r, 10, 5, True, lgh, lga))
        S["form_any10_k5"].append(form_lambdas(idx, r, 10, 5, False, lgh, lga))
        S["form_venue20_k5"].append(form_lambdas(idx, r, 20, 5, True, lgh, lga))
        S["form_any20_k5"].append(form_lambdas(idx, r, 20, 5, False, lgh, lga))
        for n in SHOT_RIDGE:
            S[n].append(lambdas_from_fit(sfits[n], r["home_team_id"], r["away_team_id"], lid))
        G["goals_ridge"].append(lambdas_from_fit(gfit, r["home_team_id"], r["away_team_id"], lid))
        G["conv_h"].append(ch)
        G["conv_a"].append(ca)
    keep = np.array(keep)
    S = {k: np.array(v)[keep] for k, v in S.items()}
    G = {k: np.array(v)[keep] for k, v in G.items()}
    return test[keep].reset_index(drop=True), S, G


# ------------------------------------------------------------------ 1. predicting shots
def part1(test, S):
    hs, as_ = test["home_shots"].values, test["away_shots"].values
    print(f"\n=== 1. PREDICTING SHOTS: {len(test)} matches | avg shots home {hs.mean():.1f}, away {as_.mean():.1f} "
          f"| spread (sd) home {hs.std():.1f}, away {as_.std():.1f} ===")
    ref = "ridge_hl365"
    ll = {k: ll_goals(v[:, 0], v[:, 1], hs, as_, 1.0) for k, v in S.items()}
    mse = {k: (((v[:, 0] - hs) ** 2 + (v[:, 1] - as_) ** 2) / 2) for k, v in S.items()}
    print(f"{'variant':<20}{'Poisson ll':>12}{'MSE':>8}{'  vs ridge_hl365 (±95%)':>28}")
    for k in sorted(ll, key=lambda k: ll[k].mean()):
        g, ci = paired(ll[ref], ll[k])
        tag = "" if k == ref else ("  <- better" if g - ci > 0 else ("  <- worse" if g + ci < 0 else ""))
        print(f"{k:<20}{ll[k].mean():>12.4f}{mse[k].mean():>8.2f}{g:>+16.4f} ±{ci:.4f}{'  *' if k == ref else ''}{tag}")
    print("(> 0 = better than the ridge model with half-life 365. 'tab_metric' = what the Shots tab shows: plain last-10 average;\n"
          " tab_w90 / tab_w80 = the same with recency weights 0.9 / 0.8 per match back; tab_w90_n15 = last 15 weighted 0.9.)")


# ------------------------------------------------------------------ 2. shots -> goals
def part2(test, S, G):
    hg, ag = test["home_goals"].values, test["away_goals"].values
    yo = (test["total_goals"] > 2.5).astype(int).values
    yb = ((hg > 0) & (ag > 0)).astype(int)
    cand = {"goals_ridge (current)": G["goals_ridge"]}
    for sname in ("ridge_hl365", "form_venue10_k5"):
        sg = np.column_stack([S[sname][:, 0] * G["conv_h"], S[sname][:, 1] * G["conv_a"]])
        cand[f"shots_only[{sname}]"] = sg
        for w in (0.1, 0.2, 0.3, 0.5):
            cand[f"blend[{sname}] w={int(w * 100)}"] = np.exp(w * np.log(sg) + (1 - w) * np.log(G["goals_ridge"]))
    print(f"\n=== 2. DO SHOTS HELP PREDICT GOALS? ({len(test)} matches; each variant uses its own best calibration scale) ===")
    print(f"constant base rate: Over {ll_bin(np.full(len(yo), yo.mean()), yo).mean():.4f}   "
          f"BTTS {ll_bin(np.full(len(yb), yb.mean()), yb).mean():.4f}")
    res = {}
    for k, L in cand.items():
        lh, la = L[:, 0], L[:, 1]
        s_o = min(SCALES, key=lambda s: ll_bin(p_over_btts(lh, la, s)[0], yo).mean())
        s_g = min(SCALES, key=lambda s: ll_goals(lh, la, hg, ag, s).mean())
        s_b = min(SCALES, key=lambda s: ll_bin(p_over_btts(lh, la, s)[1], yb).mean())
        res[k] = {"over": ll_bin(p_over_btts(lh, la, s_o)[0], yo), "btts": ll_bin(p_over_btts(lh, la, s_b)[1], yb),
                  "goals": ll_goals(lh, la, hg, ag, s_g)}
    ref = "goals_ridge (current)"
    print(f"{'variant':<34}{'Over':>8}{'BTTS':>8}{'goals':>8}{'  dGoals (±95%)':>20}{'  dOver (±95%)':>20}")
    for k in sorted(res, key=lambda k: res[k]["goals"].mean()):
        r = res[k]
        gg, gc = paired(res[ref]["goals"], r["goals"])
        og, oc = paired(res[ref]["over"], r["over"])
        mark = "  *" if k == ref else ""
        print(f"{k:<34}{r['over'].mean():>8.4f}{r['btts'].mean():>8.4f}{r['goals'].mean():>8.4f}"
              f"{gg:>+12.4f} ±{gc:.4f}{og:>+12.4f} ±{oc:.4f}{mark}")
    print("(> 0 = better than the current goals model.)")

    # best blend weight with bootstrap
    print("\n   Best weight for shots-based goals on top of the goals model (lambda = goals^(1-w) * shots^w):")
    for sname in ("ridge_hl365", "form_venue10_k5"):
        sg = np.column_stack([S[sname][:, 0] * G["conv_h"], S[sname][:, 1] * G["conv_a"]])
        base = G["goals_ridge"]

        def loss(w, ix=slice(None)):
            L = np.exp(w * np.log(sg[ix]) + (1 - w) * np.log(base[ix]))
            return ll_goals(L[:, 0], L[:, 1], hg[ix], ag[ix], 1.06).mean()

        w = minimize_scalar(loss, bounds=(-0.3, 1.0), method="bounded").x
        bs = [minimize_scalar(lambda x: loss(x, rng.integers(0, len(test), len(test))), bounds=(-0.3, 1.0),
                              method="bounded").x for _ in range(100)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"   shots[{sname}]: best w = {w:+.2f} (95% interval {lo:+.2f} .. {hi:+.2f}); "
              f"log loss {loss(0):.4f} -> {loss(w):.4f}")
    print("   (an interval that excludes 0 means shots add real information; w = 0 means they add nothing)")


# ------------------------------------------------------------------ 3. the tab's Avg metric
def part3(test, S, G):
    yo = (test["total_goals"] > 2.5).astype(int).values
    yb = ((test["home_goals"] > 0) & (test["away_goals"] > 0)).astype(int).values
    tg = test["total_goals"].values
    avg = S["tab_metric"].mean(axis=1)
    comb = S["tab_metric"].sum(axis=1)
    lam_total = G["goals_ridge"].sum(axis=1)
    print(f"\n=== 3. IS THE SHOTS TAB'S AVG USEFUL? ({len(test)} matches) ===")
    rho, p = spearmanr(avg, tg)
    rho2, _ = spearmanr(avg, lam_total)
    print(f"Spearman correlation of Avg with actual total goals: {rho:+.3f} (p={p:.3g}); "
          f"with the goals model's expected total: {rho2:+.3f}")
    q = pd.qcut(avg, 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"])
    t = pd.DataFrame({"q": q, "avg": avg, "goals": tg, "over": yo, "btts": yb}).groupby("q", observed=True).agg(
        n=("over", "size"), avg_shots=("avg", "mean"), goals=("goals", "mean"), over=("over", "mean"), btts=("btts", "mean"))
    print(f"{'Avg shots quartile':<20}{'n':>6}{'avg shots':>11}{'goals/match':>13}{'Over 2.5':>10}{'BTTS':>8}")
    for i, r in t.iterrows():
        print(f"{str(i):<20}{int(r.n):>6}{r.avg_shots:>11.1f}{r.goals:>13.2f}{r.over * 100:>9.1f}%{r.btts * 100:>7.1f}%")

    po, pb = p_over_btts(G["goals_ridge"][:, 0], G["goals_ridge"][:, 1], 1.06)
    z = (avg - avg.mean()) / avg.std()
    for name, y, p_model in (("Over 2.5", yo, po), ("BTTS Yes", yb, pb)):
        lg = np.log(np.clip(p_model, 1e-4, 1 - 1e-4) / (1 - np.clip(p_model, 1e-4, 1 - 1e-4)))

        def fit(ix):
            X = np.column_stack([np.ones(len(ix)), lg[ix], z[ix]])

            def nll(b):
                eta = X @ b
                return np.sum(np.logaddexp(0, eta) - y[ix] * eta)
            return minimize(nll, np.array([0.0, 1.0, 0.0]), method="BFGS").x, nll

        allix = np.arange(len(y))
        b, nll = fit(allix)
        b0 = minimize(lambda c: nll(np.array([c[0], c[1], 0.0])), np.array([0.0, 1.0]), method="BFGS")
        boots = [fit(rng.integers(0, len(y), len(y)))[0][2] for _ in range(150)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        gain = (b0.fun - nll(b)) / len(y)
        print(f"{name}: coefficient on Avg-shots (per 1 sd) = {b[2]:+.3f} (95% interval {lo:+.3f} .. {hi:+.3f}); "
              f"log-loss gain from adding it: {gain:+.5f}")
    print("(An interval that includes 0 means Avg adds nothing the goals model does not already know.)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--refit-days", type=int, default=14)
    ap.add_argument("--min-window", type=int, default=5)
    a = ap.parse_args()
    df = load_joined()
    test, S, G = run(df, pd.Timestamp(a.start, tz="UTC"), a.refit_days, a.min_window)
    if len(test) < 300:
        sys.exit(f"Only {len(test)} usable test matches: fill more shots data or use an earlier --start.")
    part1(test, S)
    part2(test, S, G)
    part3(test, S, G)
