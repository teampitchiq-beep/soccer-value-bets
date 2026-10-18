# model.py
"""
Over/Under 2.5 engine — ridge-regularised Poisson attack/defence model.

Replaces the old "raw home/away xG averages" approach. Differences:
  * One joint fit for ALL teams (each team's attack/defence is opponent-adjusted)
  * Ridge penalty shrinks small-sample teams toward league average
  * Exponential time decay (recent matches count more)
  * Target = blend of xG and actual goals (less noisy than either alone)
  * Matches with missing dates are dropped (they used to sort to the END
    of the data and be treated as the most recent games)

Public API is unchanged, so scan.py / app.py / predictions_history.py /
debug_picks.py keep working:
    load_data, predict_lambdas, over_under_2_5, classify_confidence, full_prediction
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

import config
from config import (
    MATCHES_XG_CSV, MATCHES_CSV,
    LAMBDA_BOOST, OVER_THRESHOLD,
    HIGH_CONF_OVER_PROB, HIGH_CONF_UNDER_PROB,
)

# Optional settings (defaults used if missing from config.py)
HALF_LIFE_DAYS = getattr(config, "HALF_LIFE_DAYS", 365.0)
RIDGE          = getattr(config, "RIDGE", 10.0)
XG_WEIGHT      = getattr(config, "XG_WEIGHT", 0.7)
MEDIUM_OVER_PROB  = getattr(config, "MEDIUM_OVER_PROB", 0.58)
MEDIUM_UNDER_PROB = getattr(config, "MEDIUM_UNDER_PROB", 0.45)

MAX_GOALS = 10


# ---------------------------------------------------------------- data
def load_data():
    if os.path.exists(MATCHES_XG_CSV):
        df = pd.read_csv(MATCHES_XG_CSV)
        print(f"Loaded {len(df)} matches from {MATCHES_XG_CSV}")
    elif os.path.exists(MATCHES_CSV):
        df = pd.read_csv(MATCHES_CSV)
        if "home_xg" not in df.columns:
            df["home_xg"] = df["home_goals"]
        if "away_xg" not in df.columns:
            df["away_xg"] = df["away_goals"]
        print(f"Loaded {len(df)} matches from {MATCHES_CSV} (no xG)")
    else:
        raise FileNotFoundError("No data found. Run python app.py first.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    bad = df["date"].isna().sum()
    if bad:
        print(f"  Dropped {bad} rows with missing dates")
    df = df.dropna(subset=["date"])
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------- fitting
def fit_model(df, now=None, half_life=None, ridge=None, xg_weight=None):
    """Fit ridge Poisson ratings on df. Returns a dict of parameters."""
    half_life = HALF_LIFE_DAYS if half_life is None else half_life
    ridge     = RIDGE if ridge is None else ridge
    xg_weight = XG_WEIGHT if xg_weight is None else xg_weight

    h = df.dropna(subset=["date", "home_xg", "away_xg"])
    if now is None:
        now = h["date"].max()

    teams = pd.unique(np.r_[h["home_team_id"].values, h["away_team_id"].values])
    leagues = pd.unique(h["league_id"].values)
    tidx = {t: i for i, t in enumerate(teams)}
    lidx = {l: i for i, l in enumerate(leagues)}
    T, L = len(teams), len(leagues)

    hi = h["home_team_id"].map(tidx).values
    ai = h["away_team_id"].map(tidx).values
    li = h["league_id"].map(lidx).values
    yh = xg_weight * h["home_xg"].values + (1 - xg_weight) * h["home_goals"].values
    ya = xg_weight * h["away_xg"].values + (1 - xg_weight) * h["away_goals"].values
    age = (now - h["date"]).dt.total_seconds().values / 86400.0
    wt = 0.5 ** (np.clip(age, 0, None) / half_life)

    def unpack(x):
        return x[:T], x[T:2 * T], x[2 * T:2 * T + L], x[2 * T + L:]

    def objective(x):
        att, dfn, hm, bs = unpack(x)
        eta_h = bs[li] + hm[li] + att[hi] - dfn[ai]
        eta_a = bs[li] + att[ai] - dfn[hi]
        mh, ma = np.exp(eta_h), np.exp(eta_a)
        nll = (np.sum(wt * (mh - yh * eta_h)) + np.sum(wt * (ma - ya * eta_a))
               + ridge * (att @ att + dfn @ dfn))
        gh, ga = wt * (mh - yh), wt * (ma - ya)
        g = np.zeros_like(x)
        np.add.at(g[:T], hi, gh)
        np.add.at(g[:T], ai, ga)
        np.add.at(g[T:2 * T], ai, -gh)
        np.add.at(g[T:2 * T], hi, -ga)
        np.add.at(g[2 * T:2 * T + L], li, gh)
        np.add.at(g[2 * T + L:], li, gh + ga)
        g[:T] += 2 * ridge * att
        g[T:2 * T] += 2 * ridge * dfn
        return nll, g

    x0 = np.zeros(2 * T + 2 * L)
    x0[2 * T + L:] = np.log(1.2)
    r = minimize(objective, x0, jac=True, method="L-BFGS-B")
    att, dfn, hm, bs = unpack(r.x)
    return {"tidx": tidx, "lidx": lidx, "att": att, "dfn": dfn,
            "hm": hm, "bs": bs}


def lambdas_from_fit(fit, home_team_id, away_team_id, league_id=None):
    """Raw (uncalibrated) expected goals for a fixture from a fitted model."""
    i = fit["tidx"].get(home_team_id)
    j = fit["tidx"].get(away_team_id)
    l = fit["lidx"].get(league_id) if league_id is not None else None

    if l is None:  # unknown league -> average of known leagues
        base, home = fit["bs"].mean(), fit["hm"].mean()
    else:
        base, home = fit["bs"][l], fit["hm"][l]

    a_i = fit["att"][i] if i is not None else 0.0   # unseen team => league average
    d_i = fit["dfn"][i] if i is not None else 0.0
    a_j = fit["att"][j] if j is not None else 0.0
    d_j = fit["dfn"][j] if j is not None else 0.0

    lam_h = float(np.exp(base + home + a_i - d_j))
    lam_a = float(np.exp(base + a_j - d_i))
    return max(lam_h, 0.20), max(lam_a, 0.20)


# small cache so a page load with 100 fixtures only fits once
_FIT_CACHE = {}

def _get_fit(df):
    key = (len(df), str(df["date"].max()), int(df["event_id"].iloc[-1]) if "event_id" in df else 0)
    if key not in _FIT_CACHE:
        if len(_FIT_CACHE) > 8:
            _FIT_CACHE.clear()
        _FIT_CACHE[key] = fit_model(df)
    return _FIT_CACHE[key]


def predict_lambdas(df, home_team_id, away_team_id, league_id=None):
    """
    Same signature as before. Returns RAW lambdas; callers multiply by
    LAMBDA_BOOST (now a small calibration factor, ~1.06).
    """
    fit = _get_fit(df)
    return lambdas_from_fit(fit, home_team_id, away_team_id, league_id)


# ---------------------------------------------------------------- probabilities
def over_under_2_5(lambda_home, lambda_away, max_goals=MAX_GOALS):
    k = np.arange(max_goals + 1)
    ph = poisson.pmf(k, lambda_home)
    pa = poisson.pmf(k, lambda_away)
    M = np.outer(ph, pa)
    M /= M.sum()

    hh, aa = np.meshgrid(k, k, indexing="ij")
    prob_under = float(M[(hh + aa) <= 2].sum())
    prob_over = 1.0 - prob_under

    flat = [((int(h), int(a)), float(M[h, a]))
            for h in range(max_goals + 1) for a in range(max_goals + 1)]
    top_scores = sorted(flat, key=lambda x: x[1], reverse=True)[:6]
    return prob_over, prob_under, top_scores


def classify_confidence(prob_over):
    if prob_over >= HIGH_CONF_OVER_PROB:
        return "HIGH", "Over 2.5", "green"
    if prob_over <= HIGH_CONF_UNDER_PROB:
        return "HIGH", "Under 2.5", "green"
    if MEDIUM_OVER_PROB <= prob_over < HIGH_CONF_OVER_PROB:
        return "MEDIUM", "Lean Over 2.5", "yellow"
    if HIGH_CONF_UNDER_PROB < prob_over <= MEDIUM_UNDER_PROB:
        return "MEDIUM", "Lean Under 2.5", "yellow"
    return "LOW", "No confident pick", "gray"


def full_prediction(df, home_team_id, away_team_id, league_id=None):
    lh_raw, la_raw = predict_lambdas(df, home_team_id, away_team_id, league_id)

    lh = lh_raw * LAMBDA_BOOST
    la = la_raw * LAMBDA_BOOST

    prob_over, prob_under, top_scores = over_under_2_5(lh, la)
    tier, recommendation, color = classify_confidence(prob_over)

    return {
        "lambda_home":      round(lh, 2),
        "lambda_away":      round(la, 2),
        "prob_over_2_5":    round(prob_over * 100, 1),
        "prob_under_2_5":   round(prob_under * 100, 1),
        "recommendation":   recommendation,
        "confidence":       tier,
        "confidence_color": color,
        "top_scores": [
            {"score": f"{h}-{a}", "prob": round(p * 100, 1)}
            for (h, a), p in top_scores
        ],
    }
