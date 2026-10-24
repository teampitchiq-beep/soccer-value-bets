# ht_model.py
"""
First-half goals model: expected home / away / total goals before half time and
P(Over / Under 1.5 first-half goals).

Same engine as the goals and corners models (ridge-regularised Poisson attack/defence ratings
with time decay, one rating per team across all leagues), fitted on HALF-TIME goals. The total of
two independent Poissons is Poisson, so P(Over 1.5) = P(total >= 2).

Optional settings (add to config.py to override; defaults shown):
    HT_LINE            = 1.5
    HT_HALF_LIFE_DAYS  = 365.0
    HT_RIDGE           = 10.0
    HT_SCALE           = 1.0    # multiply expected HT goals (calibration)
    HT_HIGH_PROB       = 0.75   # confidence tiers (probability of the chosen side)
    HT_MEDIUM_PROB     = 0.65
    HT_MIN_MATCHES     = 8      # each team needs this many matches with a half-time score
    HT_CAL_A           = 1.0    # probability calibration: p' = sigmoid(A * logit(p) + B)
    HT_CAL_B           = 0.0

Run  python ht_backtest.py  to check accuracy and to get recommended scale / calibration.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import poisson

import config
from model import fit_model, lambdas_from_fit

LINE        = getattr(config, "HT_LINE", 1.5)
HALF_LIFE   = getattr(config, "HT_HALF_LIFE_DAYS", 365.0)
RIDGE       = getattr(config, "HT_RIDGE", 10.0)
SCALE       = getattr(config, "HT_SCALE", 1.0)
HIGH_PROB   = getattr(config, "HT_HIGH_PROB", 0.75)
MEDIUM_PROB = getattr(config, "HT_MEDIUM_PROB", 0.65)
MIN_MATCHES = getattr(config, "HT_MIN_MATCHES", 8)
CAL_A       = getattr(config, "HT_CAL_A", 1.0)
CAL_B       = getattr(config, "HT_CAL_B", 0.0)

MATCHES_XG_CSV = config.MATCHES_XG_CSV
HT_CSV = "data/ht_goals.csv"


# ---------------------------------------------------------------- data
def load_training():
    """Finished matches that have a half-time score. None if there is no half-time data yet."""
    if not (os.path.exists(HT_CSV) and os.path.exists(MATCHES_XG_CSV)):
        return None
    c = pd.read_csv(HT_CSV).dropna(subset=["home_ht", "away_ht"])
    if c.empty:
        return None
    m = pd.read_csv(MATCHES_XG_CSV, usecols=["event_id", "date", "league_id", "home_team_id", "away_team_id"])
    m["date"] = pd.to_datetime(m["date"], errors="coerce", utc=True, format="ISO8601")
    df = m.dropna(subset=["date"]).merge(c[["event_id", "home_ht", "away_ht"]], on="event_id")
    return df.sort_values("date").reset_index(drop=True)


def _as_goal_columns(df):
    """fit_model() reads home_xg / away_xg / home_goals / away_goals: point them at HT goals."""
    d = df.copy()
    d["home_xg"] = d["home_goals"] = d["home_ht"].astype(float)
    d["away_xg"] = d["away_goals"] = d["away_ht"].astype(float)
    return d


# ---------------------------------------------------------------- fitting
def fit_ht(df, now=None, half_life=None, ridge=None):
    return fit_model(_as_goal_columns(df), now=now,
                     half_life=HALF_LIFE if half_life is None else half_life,
                     ridge=RIDGE if ridge is None else ridge, xg_weight=1.0)


def expected_ht(fit, home_id, away_id, league_id, scale=None):
    scale = SCALE if scale is None else scale
    mh, ma = lambdas_from_fit(fit, home_id, away_id, league_id)
    return mh * scale, ma * scale


def calibrate(p, a=None, b=None):
    """Platt calibration p' = sigmoid(a*logit(p) + b). a=1, b=0 leaves p unchanged."""
    a = CAL_A if a is None else a
    b = CAL_B if b is None else b
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return 1.0 / (1.0 + np.exp(-(a * np.log(p / (1 - p)) + b)))


def prob_over(mu_total, line=None, calibrated=True):
    """P(first-half goals > line) for a Poisson total (1.5 -> at least 2 goals)."""
    line = LINE if line is None else line
    need = int(np.floor(line)) + 1
    p = float(poisson.sf(need - 1, mu_total))
    return float(calibrate(p)) if calibrated else p


# ---------------------------------------------------------------- cached model for the app
_CACHE = {"key": None, "obj": None}


def _file_key():
    return tuple(os.path.getmtime(p) if os.path.exists(p) else 0 for p in (HT_CSV, MATCHES_XG_CSV))


def get_model():
    """(fit, info, team) built from your stored half-time data, or None if there is not enough."""
    key = _file_key()
    if _CACHE["key"] == key:
        return _CACHE["obj"]
    df = load_training()
    obj = None
    if df is not None and len(df) >= 100:
        fit = fit_ht(df)
        long = pd.concat([
            pd.DataFrame({"team": df["home_team_id"], "date": df["date"], "f": df["home_ht"], "a": df["away_ht"]}),
            pd.DataFrame({"team": df["away_team_id"], "date": df["date"], "f": df["away_ht"], "a": df["home_ht"]}),
        ]).sort_values("date")
        team = {}
        for t, g in long.groupby("team"):
            last = g.tail(10)
            team[int(t)] = {"n": len(g), "for": float(last["f"].mean()), "against": float(last["a"].mean())}
        tot = df["home_ht"] + df["away_ht"]
        info = {"n_matches": len(df), "avg_total": float(tot.mean()),
                "base_over": float((tot > LINE).mean()),
                "first": df["date"].min().strftime("%Y-%m-%d")}
        obj = (fit, info, team)
    _CACHE["key"], _CACHE["obj"] = key, obj
    return obj


def top_scores(mu_h, mu_a, n=2, max_goals=6):
    """The n most likely half-time scorelines (independent Poisson). [{'score': '1-0', 'prob': 21.3}, ...]"""
    k = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(k, mu_h), poisson.pmf(k, mu_a))
    m /= m.sum()
    flat = sorted(((int(h), int(a), float(m[h, a])) for h in k for a in k), key=lambda t: t[2], reverse=True)
    return [{"score": f"{h}-{a}", "prob": round(p * 100, 1)} for h, a, p in flat[:n]]


def _tier(p):
    return "HIGH" if p >= HIGH_PROB else ("MEDIUM" if p >= MEDIUM_PROB else "LOW")


def ht_predictions(matches):
    """
    matches: list of dicts from scan.get_todays_matches (needs home_id, away_id, league_id).
    Returns (rows, info). rows are sorted by date; info has counts / messages for the UI.
    """
    model = get_model()
    if model is None:
        return [], {"ready": False, "line": LINE,
                    "message": "No half-time data yet. Run  python ht_data.py  once (needs internet "
                               "and API quota), then refresh."}
    fit, info, team = model
    rows, skipped = [], 0
    for m in matches:
        hid, aid = m.get("home_id"), m.get("away_id")
        th, ta = team.get(hid), team.get(aid)
        if not th or not ta or th["n"] < MIN_MATCHES or ta["n"] < MIN_MATCHES:
            skipped += 1
            continue
        mu_h, mu_a = expected_ht(fit, hid, aid, m["league_id"])
        mu_t = mu_h + mu_a
        po = prob_over(mu_t)
        over = po >= 0.5
        prob = po if over else 1 - po
        rows.append({
            "event_id": m["event_id"], "date": m["date"], "league_id": m["league_id"],
            "league_name": m["league_name"], "home": m["home"], "away": m["away"],
            "mu_h": round(mu_h, 2), "mu_a": round(mu_a, 2), "mu_t": round(mu_t, 2),
            "top_2_scores": top_scores(mu_h, mu_a),
            "p_over": round(po, 4), "p_under": round(1 - po, 4),
            "pick": f"{'Over' if over else 'Under'} {LINE:g} HT", "is_over": over,
            "prob": round(prob, 4), "tier": _tier(prob), "line": LINE,
            "h_for": round(th["for"], 2), "h_against": round(th["against"], 2),
            "a_for": round(ta["for"], 2), "a_against": round(ta["against"], 2),
            "home_n": th["n"], "away_n": ta["n"],
        })
    rows.sort(key=lambda r: r["date"])
    out = dict(info, ready=True, line=LINE, skipped=skipped, shown=len(rows), message="")
    return rows, out
