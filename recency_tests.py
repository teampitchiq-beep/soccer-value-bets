# recency_tests.py
"""
Does recency help?  Walk-forward comparison of how much history the goals model should use.

Every match from --start onward is predicted using ONLY earlier matches (refit every --refit-days).
Variants:
  ridge_hl<N>      current ridge-Poisson model with half-life N days (365 = the app's setting)
  win<N>d          same model, flat weights, but only the last N days of data (hard cutoff)
  form_venue10     pure "last 10 matches" model: home attack from the home team's last 10 HOME matches,
                   defence from the away team's last 10 AWAY matches (same idea as the Shots tab)
  form_any10       last 10 matches at any venue
  form_*_k5        same, shrunk toward the league average with 5 pseudo-matches
  blend_*          geometric blend of the ridge model and a form model

Metrics (lower = better):  Over 2.5 log loss, BTTS log loss, goals log loss (Poisson, home+away, most data-efficient).
Each variant gets its own best calibration scale (same treatment for all). Differences vs the current model are
paired per match, with a 95% interval: if the interval includes 0 the difference may be luck.

Run:  python recency_tests.py [--start 2026-01-01] [--refit-days 14]
"""
import argparse
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.special import gammaln

from model import load_data, fit_model, lambdas_from_fit, XG_WEIGHT

SCALES = np.round(np.arange(0.90, 1.31, 0.02), 2)
MAXG = 10
K = np.arange(MAXG + 1)


# ------------------------------------------------------------------ probabilities
def p_over_btts(lh, la, scale):
    """Over 2.5 and BTTS probabilities (vectorised) from expected goals."""
    lh = np.asarray(lh) * scale
    la = np.asarray(la) * scale
    ph = poisson.pmf(K[None, :], lh[:, None])
    pa = poisson.pmf(K[None, :], la[:, None])
    M = ph[:, :, None] * pa[:, None, :]
    M /= M.sum(axis=(1, 2), keepdims=True)
    tot = K[:, None] + K[None, :]
    under = M[:, tot <= 2].sum(axis=1)
    btts = M[:, 1:, 1:].sum(axis=(1, 2))
    return 1 - under, btts


def ll_bin(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def ll_goals(lh, la, hg, ag, scale):
    lh, la = np.maximum(np.asarray(lh) * scale, 1e-6), np.maximum(np.asarray(la) * scale, 1e-6)
    return -(hg * np.log(lh) - lh - gammaln(hg + 1) + ag * np.log(la) - la - gammaln(ag + 1)) / 2.0


# ------------------------------------------------------------------ form (last-N) model
class FormIndex:
    """Per-team chronological lists so 'last N home / away / any matches before date t' is a cheap lookup."""

    def __init__(self, df):
        d = df.sort_values("date")
        self.t = {}
        for side, tid, gf, ga in (("h", "home_team_id", "yh", "ya"), ("a", "away_team_id", "ya", "yh")):
            for team, g in d.groupby(tid):
                self.t[(side, int(team))] = (g["date"].values.astype("datetime64[ns]"), g[gf].values, g[ga].values)
        # any-venue lists
        allrows = {}
        for side, tid, gf, ga in (("h", "home_team_id", "yh", "ya"), ("a", "away_team_id", "ya", "yh")):
            for team, g in d.groupby(tid):
                allrows.setdefault(int(team), []).append(pd.DataFrame(
                    {"date": g["date"].values, "gf": g[gf].values, "ga": g[ga].values}))
        self.any = {}
        for team, parts in allrows.items():
            x = pd.concat(parts).sort_values("date")
            self.any[team] = (x["date"].values.astype("datetime64[ns]"), x["gf"].values, x["ga"].values)

    def last(self, key, t, n, venue=True):
        src = self.t.get(key) if venue else self.any.get(key)
        if src is None:
            return None
        dates, gf, ga = src
        i = np.searchsorted(dates, np.datetime64(t.tz_convert("UTC").tz_localize(None)), side="left")
        lo = max(0, i - n)
        if i - lo == 0:
            return None
        return gf[lo:i].sum(), ga[lo:i].sum(), i - lo


def form_lambdas(idx, r, n, shrink, venue, lg_home, lg_away):
    """Expected goals from last-n form. lg_home/lg_away: league average goals for home/away teams."""
    t = r["date"]
    if venue:
        hh = idx.last(("h", int(r["home_team_id"])), t, n, True)      # home team at home: scored, conceded
        aa = idx.last(("a", int(r["away_team_id"])), t, n, True)      # away team away: scored, conceded
    else:
        hh = idx.last(int(r["home_team_id"]), t, n, False)
        aa = idx.last(int(r["away_team_id"]), t, n, False)
    avg = 0.5 * (lg_home + lg_away)

    def rate(res, base):
        if res is None:
            return base
        gf, ga, m = res
        return gf, ga, m

    def shr(total, m, base):
        return (total + shrink * base) / (m + shrink) if (m + shrink) > 0 else base

    if venue:
        h_att = shr(hh[0], hh[2], lg_home) if hh else lg_home
        a_def = shr(aa[1], aa[2], lg_home) if aa else lg_home        # goals the away team concedes away
        a_att = shr(aa[0], aa[2], lg_away) if aa else lg_away
        h_def = shr(hh[1], hh[2], lg_away) if hh else lg_away        # goals the home team concedes at home
        lh = h_att * a_def / lg_home
        la = a_att * h_def / lg_away
    else:
        h_att = shr(hh[0], hh[2], avg) if hh else avg
        h_def = shr(hh[1], hh[2], avg) if hh else avg
        a_att = shr(aa[0], aa[2], avg) if aa else avg
        a_def = shr(aa[1], aa[2], avg) if aa else avg
        lh = h_att * a_def / avg * (lg_home / avg)
        la = a_att * h_def / avg * (lg_away / avg)
    return max(lh, 0.2), max(la, 0.2)


# ------------------------------------------------------------------ walk-forward driver
def run(df, start, refit_days):
    df = df.copy()
    df["yh"] = XG_WEIGHT * df["home_xg"] + (1 - XG_WEIGHT) * df["home_goals"]
    df["ya"] = XG_WEIGHT * df["away_xg"] + (1 - XG_WEIGHT) * df["away_goals"]
    idx = FormIndex(df)
    test = df[df["date"] >= start].reset_index(drop=True)

    ridge_variants = {f"ridge_hl{h}": dict(half_life=h) for h in (45, 90, 180, 365, 730)}
    window_variants = {f"win{n}d": n for n in (120, 240, 365)}
    out = {name: [] for name in list(ridge_variants) + list(window_variants)
           + ["form_venue10", "form_venue10_k5", "form_any10", "form_any10_k5", "form_venue20_k5", "form_any20_k5"]}

    fits, last = None, None
    t0 = time.time()
    for _, r in test.iterrows():
        if fits is None or (r["date"] - last).days >= refit_days:
            past = df[df["date"] < r["date"]]
            fits = {name: fit_model(past, now=r["date"], **kw) for name, kw in ridge_variants.items()}
            for name, n in window_variants.items():
                w = past[past["date"] >= r["date"] - pd.Timedelta(days=n)]
                fits[name] = fit_model(w, now=r["date"], half_life=1e9)
            year = past[past["date"] >= r["date"] - pd.Timedelta(days=365)]
            lg = year.groupby("league_id").agg(h=("yh", "mean"), a=("ya", "mean"))
            glob = (year["yh"].mean(), year["ya"].mean())
            last = r["date"]
            print(f"  refit {r['date'].date()}  ({time.time() - t0:.0f}s)", file=sys.stderr)
        for name, f in fits.items():
            out[name].append(lambdas_from_fit(f, r["home_team_id"], r["away_team_id"], r["league_id"]))
        lid = r["league_id"]
        lgh, lga = (lg.loc[lid, "h"], lg.loc[lid, "a"]) if lid in lg.index else glob
        out["form_venue10"].append(form_lambdas(idx, r, 10, 0, True, lgh, lga))
        out["form_venue10_k5"].append(form_lambdas(idx, r, 10, 5, True, lgh, lga))
        out["form_any10"].append(form_lambdas(idx, r, 10, 0, False, lgh, lga))
        out["form_any10_k5"].append(form_lambdas(idx, r, 10, 5, False, lgh, lga))
        out["form_venue20_k5"].append(form_lambdas(idx, r, 20, 5, True, lgh, lga))
        out["form_any20_k5"].append(form_lambdas(idx, r, 20, 5, False, lgh, lga))

    lam = {k: np.array(v) for k, v in out.items()}
    for base in ("ridge_hl365", "ridge_hl180"):
        for form in ("form_venue10_k5", "form_any10_k5", "form_any20_k5"):
            for w in (0.15, 0.30, 0.50):
                lam[f"blend_{base[6:]}+{form[5:]}_w{int(w * 100)}"] = np.exp(
                    w * np.log(lam[form]) + (1 - w) * np.log(lam[base]))
    return test, lam


def evaluate(test, lam):
    y_over = (test["total_goals"] > 2.5).astype(int).values
    y_btts = ((test["home_goals"] > 0) & (test["away_goals"] > 0)).astype(int).values
    hg, ag = test["home_goals"].values, test["away_goals"].values
    res = {}
    for name, L in lam.items():
        lh, la = L[:, 0], L[:, 1]
        best = None
        for s in SCALES:
            po, pb = p_over_btts(lh, la, s)
            ov = ll_bin(po, y_over)
            if best is None or ov.mean() < best["ov"].mean():
                best = {"s": s, "ov": ov, "bt": ll_bin(pb, y_btts), "po": po}
        # BTTS / goals scored at their own best scale
        bt_best = min(((ll_bin(p_over_btts(lh, la, s)[1], y_btts), s) for s in SCALES), key=lambda t: t[0].mean())
        gl_best = min(((ll_goals(lh, la, hg, ag, s), s) for s in SCALES), key=lambda t: t[0].mean())
        res[name] = {"scale": best["s"], "over": best["ov"], "btts": bt_best[0], "goals": gl_best[0],
                     "acc": float(((best["po"] > 0.5) == y_over).mean())}
    base_over = ll_bin(np.full(len(y_over), y_over.mean()), y_over)
    base_btts = ll_bin(np.full(len(y_btts), y_btts.mean()), y_btts)
    return res, base_over, base_btts, y_over, y_btts


def paired(a, b):
    d = a - b                                   # positive = b better than a ... here a=reference, b=variant
    return d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(len(d))


def report(test, lam, title="ALL TEST MATCHES"):
    res, bo, bb, yo, yb = evaluate(test, lam)
    ref = "ridge_hl365"
    print(f"\n=== {title}: {len(test)} matches | actual Over2.5 {yo.mean() * 100:.1f}% | BTTS {yb.mean() * 100:.1f}% ===")
    print(f"constant base-rate log loss: Over {bo.mean():.4f}   BTTS {bb.mean():.4f}")
    print(f"{'variant':<34}{'scale':>6}{'Over':>9}{'BTTS':>9}{'goals':>9}  {'dGoals vs current (±95%)':>26}")
    order = sorted(res, key=lambda k: res[k]["goals"].mean())
    for k in order:
        r = res[k]
        gain, ci = paired(res[ref]["goals"], r["goals"])
        tag = "" if k == ref else ("  <- better" if gain - ci > 0 else ("  <- worse" if gain + ci < 0 else ""))
        mark = " *" if k == ref else "  "
        print(f"{k:<34}{r['scale']:>6.2f}{r['over'].mean():>9.4f}{r['btts'].mean():>9.4f}{r['goals'].mean():>9.4f}"
              f"  {gain:>+10.4f} ±{ci:.4f}{mark}{tag}")
    print("(* = current app setting. dGoals > 0 means the variant beats the current model. Differences inside ±95% may be luck.)")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--refit-days", type=int, default=14)
    a = ap.parse_args()
    df = load_data()
    test, lam = run(df, pd.Timestamp(a.start, tz="UTC"), a.refit_days)
    np.save("data/_recency_lambdas.npy", {k: v for k, v in lam.items()}, allow_pickle=True)
    test.to_pickle("data/_recency_test.pkl")
    res = report(test, lam)
    half = len(test) // 2
    for nm, sl in (("FIRST HALF", slice(0, half)), ("SECOND HALF", slice(half, None))):
        report(test.iloc[sl].reset_index(drop=True), {k: v[sl] for k, v in lam.items()}, nm)
