# btts_model.py
"""
Both Teams To Score (BTTS Yes / No) from the goals model's expected goals.

With independent Poisson goals:  P(BTTS Yes) = (1 - e^-lambda_home) * (1 - e^-lambda_away)

Uses the expected goals already stored with every fixture (lh / la, same ones behind Over/Under 2.5),
so it needs no extra data and no extra API calls. BTTS bookmaker odds come with the fixture odds.

A walk-forward backtest on your data (python btts_backtest.py) showed the probabilities are well
calibrated, so no extra calibration is applied by default. Optional settings (config.py):
    BTTS_SCALE = 1.0        # multiplies the stored expected goals
    BTTS_CAL_A = 1.0        # p' = sigmoid(A * logit(p) + B)
    BTTS_CAL_B = 0.0
    BTTS_HIGH_PROB = 0.65   # confidence tiers (probability of the chosen side)
    BTTS_MEDIUM_PROB = 0.58
"""
import numpy as np

import config

SCALE       = getattr(config, "BTTS_SCALE", 1.0)
CAL_A       = getattr(config, "BTTS_CAL_A", 1.0)
CAL_B       = getattr(config, "BTTS_CAL_B", 0.0)
HIGH_PROB   = getattr(config, "BTTS_HIGH_PROB", 0.65)
MEDIUM_PROB = getattr(config, "BTTS_MEDIUM_PROB", 0.58)


def p_yes(lh, la, scale=None, calibrated=True):
    scale = SCALE if scale is None else scale
    p = float((1 - np.exp(-lh * scale)) * (1 - np.exp(-la * scale)))
    if calibrated and (CAL_A != 1.0 or CAL_B != 0.0):
        p = min(max(p, 1e-4), 1 - 1e-4)
        p = float(1 / (1 + np.exp(-(CAL_A * np.log(p / (1 - p)) + CAL_B))))
    return p


def tier(prob):
    return "HIGH" if prob >= HIGH_PROB else ("MEDIUM" if prob >= MEDIUM_PROB else "LOW")


def single(lh, la, odds_yes=None, odds_no=None):
    """Prediction dict for one fixture (used by Single Match and the list below)."""
    py = p_yes(lh, la)
    yes = py >= 0.5
    prob = py if yes else 1 - py
    odds = odds_yes if yes else odds_no
    return {
        "p_yes": round(py, 4), "p_no": round(1 - py, 4),
        "pick": "BTTS Yes" if yes else "BTTS No", "is_yes": yes,
        "prob": round(prob, 4), "tier": tier(prob),
        "odds_yes": odds_yes, "odds_no": odds_no, "odds": odds,
        "ev": round(prob * odds - 1, 3) if odds else None,
    }


def btts_predictions(matches):
    """matches: list of dicts from scan.get_todays_matches. Returns rows sorted by date."""
    rows = []
    for m in matches:
        if m.get("lh") is None or m.get("la") is None:
            continue
        r = {"event_id": m["event_id"], "date": m["date"], "league_id": m["league_id"],
             "league_name": m["league_name"], "home": m["home"], "away": m["away"],
             "lh": m["lh"], "la": m["la"], "top_2_scores": m.get("top_2_scores", []),
             "home_apps": m.get("home_apps"), "away_apps": m.get("away_apps")}
        r.update(single(m["lh"], m["la"], m.get("btts_yes_odds"), m.get("btts_no_odds")))
        rows.append(r)
    rows.sort(key=lambda r: r["date"])
    return rows
