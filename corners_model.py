# corners_model.py
"""
Corners model: expected home / away / total corners and P(Over / Under the line, default 8.5).

Same engine as the goals model (ridge-regularised Poisson attack/defence ratings with
time decay, one rating per team across all leagues), fitted on corner counts instead of
goals. Total corners are usually a little more spread out than a pure Poisson, so the
probability uses a Negative Binomial with a spread factor phi (variance = phi x mean),
estimated from the data (override with CORNERS_DISPERSION).

Optional settings (add to config.py to override; defaults shown):
    CORNERS_LINE            = 8.5
    CORNERS_HALF_LIFE_DAYS  = 365.0
    CORNERS_RIDGE           = 25.0   # corner counts are bigger than goals, so shrink a bit more
    CORNERS_SCALE           = 1.0    # multiply expected corners (calibration)
    CORNERS_DISPERSION      = None   # phi >= 1; None = estimate from the data
    CORNERS_HIGH_PROB       = 0.75   # confidence tiers (probability of the chosen side)
    CORNERS_MEDIUM_PROB     = 0.65
    CORNERS_MIN_MATCHES     = 8      # each team needs this many matches with corner data
    CORNERS_CAL_A           = 1.0    # probability calibration: p' = sigmoid(A * logit(p) + B)
    CORNERS_CAL_B           = 0.0    #   (A < 1 pulls overconfident probabilities toward the middle)

Run  python corners_backtest.py  to check accuracy and to get recommended scale / dispersion.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

import config
from model import fit_model, lambdas_from_fit

LINE         = getattr(config, "CORNERS_LINE", 8.5)
HALF_LIFE    = getattr(config, "CORNERS_HALF_LIFE_DAYS", 365.0)
RIDGE        = getattr(config, "CORNERS_RIDGE", 25.0)
SCALE        = getattr(config, "CORNERS_SCALE", 1.0)
DISPERSION   = getattr(config, "CORNERS_DISPERSION", None)
HIGH_PROB    = getattr(config, "CORNERS_HIGH_PROB", 0.75)
MEDIUM_PROB  = getattr(config, "CORNERS_MEDIUM_PROB", 0.65)
MIN_MATCHES  = getattr(config, "CORNERS_MIN_MATCHES", 8)
CAL_A        = getattr(config, "CORNERS_CAL_A", 1.0)
CAL_B        = getattr(config, "CORNERS_CAL_B", 0.0)

MATCHES_XG_CSV = config.MATCHES_XG_CSV
CORNERS_CSV = "data/corners.csv"


# ---------------------------------------------------------------- data
def load_training():
    """Finished matches that have corner counts. None if there is no corner data yet."""
    if not (os.path.exists(CORNERS_CSV) and os.path.exists(MATCHES_XG_CSV)):
        return None
    c = pd.read_csv(CORNERS_CSV).dropna(subset=["home_corners", "away_corners"])
    if c.empty:
        return None
    m = pd.read_csv(MATCHES_XG_CSV, usecols=["event_id", "date", "league_id", "home_team_id", "away_team_id"])
    m["date"] = pd.to_datetime(m["date"], errors="coerce", utc=True, format="ISO8601")
    df = m.dropna(subset=["date"]).merge(c[["event_id", "home_corners", "away_corners"]], on="event_id")
    return df.sort_values("date").reset_index(drop=True)


def _as_goal_columns(df):
    """fit_model() reads home_xg / away_xg / home_goals / away_goals: point them at corners."""
    d = df.copy()
    d["home_xg"] = d["home_goals"] = d["home_corners"].astype(float)
    d["away_xg"] = d["away_goals"] = d["away_corners"].astype(float)
    return d


# ---------------------------------------------------------------- fitting
def fit_corners(df, now=None, half_life=None, ridge=None):
    return fit_model(_as_goal_columns(df), now=now,
                     half_life=HALF_LIFE if half_life is None else half_life,
                     ridge=RIDGE if ridge is None else ridge, xg_weight=1.0)


def expected_corners(fit, home_id, away_id, league_id, scale=None):
    scale = SCALE if scale is None else scale
    mh, ma = lambdas_from_fit(fit, home_id, away_id, league_id)
    return mh * scale, ma * scale


def estimate_phi(fit, df, scale=None):
    """Spread of total corners around the model (variance / mean), ~1 = Poisson."""
    mu = np.array([sum(expected_corners(fit, r.home_team_id, r.away_team_id, r.league_id, scale))
                   for r in df.itertuples()])
    t = (df["home_corners"] + df["away_corners"]).values
    n, k = len(df), len(fit["tidx"])
    pearson = np.mean((t - mu) ** 2 / mu)
    return float(max(1.0, pearson * n / max(n - k, 1)))      # small df correction


def calibrate(p, a=None, b=None):
    """Platt calibration p' = sigmoid(a*logit(p) + b). a=1, b=0 leaves p unchanged."""
    a = CAL_A if a is None else a
    b = CAL_B if b is None else b
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return 1.0 / (1.0 + np.exp(-(a * np.log(p / (1 - p)) + b)))


def prob_over(mu_total, phi, line=None, calibrated=True):
    """P(total corners > line): Poisson (phi ~ 1) or Negative Binomial (phi > 1), then calibrated."""
    line = LINE if line is None else line
    need = int(np.floor(line)) + 1                          # 8.5 -> at least 9
    if phi <= 1.0001:
        p = float(poisson.sf(need - 1, mu_total))
    else:
        r = mu_total / (phi - 1.0)
        p = float(nbinom.sf(need - 1, r, r / (r + mu_total)))
    return float(calibrate(p)) if calibrated else p


# ---------------------------------------------------------------- cached model for the app
_CACHE = {"key": None, "obj": None}


def _file_key():
    return tuple(os.path.getmtime(p) if os.path.exists(p) else 0 for p in (CORNERS_CSV, MATCHES_XG_CSV))


def get_model():
    """(fit, phi, info) built from your stored corner data, or None if there is none."""
    key = _file_key()
    if _CACHE["key"] == key:
        return _CACHE["obj"]
    df = load_training()
    obj = None
    if df is not None and len(df) >= 100:
        fit = fit_corners(df)
        phi = float(DISPERSION) if DISPERSION else estimate_phi(fit, df)
        # per-team match counts and recent corner averages (for display)
        long = pd.concat([
            pd.DataFrame({"team": df["home_team_id"], "date": df["date"], "f": df["home_corners"], "a": df["away_corners"]}),
            pd.DataFrame({"team": df["away_team_id"], "date": df["date"], "f": df["away_corners"], "a": df["home_corners"]}),
        ]).sort_values("date")
        team = {}
        for t, g in long.groupby("team"):
            last = g.tail(10)
            team[int(t)] = {"n": len(g), "for": float(last["f"].mean()), "against": float(last["a"].mean())}
        info = {"n_matches": len(df), "phi": phi, "avg_total": float((df["home_corners"] + df["away_corners"]).mean()),
                "base_over": float(((df["home_corners"] + df["away_corners"]) > LINE).mean()),
                "first": df["date"].min().strftime("%Y-%m-%d")}
        obj = (fit, phi, info, team)
    _CACHE["key"], _CACHE["obj"] = key, obj
    return obj


def _tier(p):
    return "HIGH" if p >= HIGH_PROB else ("MEDIUM" if p >= MEDIUM_PROB else "LOW")


def corner_predictions(matches):
    """
    matches: list of dicts from scan.get_todays_matches (needs home_id, away_id, league_id).
    Returns (rows, info). rows are sorted by date; info has counts / messages for the UI.
    """
    model = get_model()
    if model is None:
        return [], {"ready": False, "line": LINE,
                    "message": "No corner data yet. Run  python corners_data.py  once (needs internet), then refresh."}
    fit, phi, info, team = model
    rows, skipped = [], 0
    for m in matches:
        hid, aid = m.get("home_id"), m.get("away_id")
        th, ta = team.get(hid), team.get(aid)
        if not th or not ta or th["n"] < MIN_MATCHES or ta["n"] < MIN_MATCHES:
            skipped += 1
            continue
        mu_h, mu_a = expected_corners(fit, hid, aid, m["league_id"])
        mu_t = mu_h + mu_a
        po = prob_over(mu_t, phi)
        over = po >= 0.5
        prob = po if over else 1 - po
        rows.append({
            "event_id": m["event_id"], "date": m["date"], "league_id": m["league_id"],
            "league_name": m["league_name"], "home": m["home"], "away": m["away"],
            "mu_h": round(mu_h, 1), "mu_a": round(mu_a, 1), "mu_t": round(mu_t, 1),
            "p_over": round(po, 4), "p_under": round(1 - po, 4),
            "pick": f"{'Over' if over else 'Under'} {LINE:g}", "is_over": over,
            "prob": round(prob, 4), "tier": _tier(prob), "line": LINE,
            "h_for": round(th["for"], 1), "h_against": round(th["against"], 1),
            "a_for": round(ta["for"], 1), "a_against": round(ta["against"], 1),
            "home_n": th["n"], "away_n": ta["n"],
        })
    rows.sort(key=lambda r: r["date"])
    out = dict(info, ready=True, line=LINE, skipped=skipped, shown=len(rows), message="")
    return rows, out
