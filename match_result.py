# match_result.py
"""
Match result (1X2: Home win / Draw / Away win) from the goals model's expected goals.

With independent Poisson goals, the probability of every scoreline is P(h) * P(a); the three outcomes are sums over that grid
(home > away, equal, home < away). Uses the expected goals already stored with every fixture (lh / la, the same ones behind
Over/Under 2.5 and BTTS), so it needs no extra data and no extra API calls. The 1X2 bookmaker odds (home_win / draw / away_win)
come in the same /events/{id}/odds/ response as the other markets, so they cost no extra request either.

Walk-forward backtest on 3,429 of your matches (model probabilities from earlier data only):
  * log loss 1.0088 vs 1.0655 for always predicting the base rate: the probabilities carry real signal
  * plain Poisson under-predicts draws a little (23.3% predicted vs 24.6% actual), so a small draw boost is applied
    (DRAW_BOOST = 0.05 multiplies the probability of every drawn scoreline, then the grid is renormalised). The log-loss gain
    is tiny and inside the noise, but the average predicted draw rate lands much closer to reality
  * well calibrated in the middle, slightly cautious on strong favourites (60-70% predicted -> 72% won) and a little generous to
    big underdogs (under 20% predicted -> 4-10% won)
Optional settings (config.py):
    DRAW_BOOST      = 0.05    # 0 = plain Poisson
    X12_HIGH_PROB   = 0.60    # confidence tiers (probability of the most likely outcome)
    X12_MEDIUM_PROB = 0.50
"""
import numpy as np

import config

DRAW_BOOST = float(getattr(config, "DRAW_BOOST", 0.05))
HIGH_PROB = float(getattr(config, "X12_HIGH_PROB", 0.60))
MEDIUM_PROB = float(getattr(config, "X12_MEDIUM_PROB", 0.50))
MAX_GOALS = 12

OUTCOMES = ("Home", "Draw", "Away")            # sides, in display order


def _pmf(lam, n=MAX_GOALS):
    p = np.empty(n + 1)
    p[0] = np.exp(-lam)
    for k in range(1, n + 1):
        p[k] = p[k - 1] * lam / k
    return p


def probs(lh, la, draw_boost=None):
    """(P(home win), P(draw), P(away win)) as floats that sum to 1."""
    d = DRAW_BOOST if draw_boost is None else float(draw_boost)
    m = np.outer(_pmf(float(lh)), _pmf(float(la)))
    idx = np.arange(MAX_GOALS + 1)
    if d:
        m[idx, idx] *= (1.0 + d)
    m /= m.sum()
    return float(np.tril(m, -1).sum()), float(np.trace(m)), float(np.triu(m, 1).sum())


def tier(prob):
    return "HIGH" if prob >= HIGH_PROB else ("MEDIUM" if prob >= MEDIUM_PROB else "LOW")


def _valid(o):
    try:
        return o is not None and float(o) > 1.0
    except (TypeError, ValueError):
        return False


def single(lh, la, odds_home=None, odds_draw=None, odds_away=None):
    """Prediction dict for one fixture (the 1X2 tab rows)."""
    ph, pd_, pa = probs(lh, la)
    p = {"H": ph, "D": pd_, "A": pa}
    odds = {"H": float(odds_home) if _valid(odds_home) else None,
            "D": float(odds_draw) if _valid(odds_draw) else None,
            "A": float(odds_away) if _valid(odds_away) else None}
    key = max(p, key=p.get)
    names = {"H": "Home win", "D": "Draw", "A": "Away win"}
    ev = {k: (round(p[k] * odds[k] - 1, 3) if odds[k] else None) for k in p}
    book = None
    if all(odds.values()):                                  # bookmaker-implied probabilities with the margin removed
        inv = {k: 1.0 / odds[k] for k in odds}
        tot = sum(inv.values())
        book = {k: round(100 * inv[k] / tot, 1) for k in inv}
    return {
        "p_home": round(ph, 4), "p_draw": round(pd_, 4), "p_away": round(pa, 4),
        "pick": names[key], "pick_key": key, "prob": round(p[key], 4), "tier": tier(p[key]),
        "odds_home": odds["H"], "odds_draw": odds["D"], "odds_away": odds["A"], "odds": odds[key],
        "ev": ev[key], "ev_home": ev["H"], "ev_draw": ev["D"], "ev_away": ev["A"], "book": book,
    }


def match_result_predictions(matches):
    """matches: list of dicts from scan.get_todays_matches. Returns rows sorted by date."""
    rows = []
    for m in matches:
        if m.get("lh") is None or m.get("la") is None:
            continue
        r = {"event_id": m["event_id"], "date": m["date"], "league_id": m["league_id"],
             "league_name": m["league_name"], "home": m["home"], "away": m["away"],
             "lh": m["lh"], "la": m["la"], "top_2_scores": m.get("top_2_scores", []),
             "home_apps": m.get("home_apps"), "away_apps": m.get("away_apps")}
        r.update(single(m["lh"], m["la"], m.get("odds_home"), m.get("odds_draw"), m.get("odds_away")))
        rows.append(r)
    rows.sort(key=lambda r: r["date"])
    return rows


if __name__ == "__main__":
    ph, pd_, pa = probs(1.6, 1.1)
    assert abs(ph + pd_ + pa - 1) < 1e-9 and ph > pa > 0 and pd_ > 0
    p0 = probs(1.3, 1.3, draw_boost=0.0)
    assert abs(p0[0] - p0[2]) < 1e-9, "equal teams: home and away symmetric"
    assert probs(1.3, 1.3, draw_boost=0.1)[1] > p0[1], "a draw boost raises the draw probability"
    s = single(1.6, 1.1, 1.9, 3.6, 4.2)
    assert s["pick"] == "Home win" and s["ev"] == round(s["prob"] * 1.9 - 1, 3) and abs(sum(s["book"].values()) - 100) < 0.2
    assert single(1.6, 1.1)["ev"] is None and single(1.6, 1.1)["book"] is None
    print("match_result self-test passed")
