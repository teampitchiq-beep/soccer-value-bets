# ou15_model.py
"""
Over / Under 1.5 goals, from the goals model's expected goals.

P(Over 1.5) = P(total goals >= 2) from the independent-Poisson scoreline grid built on the expected goals already stored with every fixture
(lh / la, the same ones behind Over/Under 2.5, BTTS and 1X2). No extra data and no extra API calls: the bookmaker prices (over_15_goals,
under_15_goals) arrive in the same /events/{id}/odds/ response as the other markets.

Walk-forward backtest on 3,429 of your matches (probabilities from earlier data only):
  * calibrated as it stands: 79.1% predicted vs 79.2% actual Over 1.5, so NO correction is applied. (The earlier version of this market used a
    hand-tuned logit shift of +0.073; on today's model its measured effect is zero, inside the noise.)
  * 0.60-0.70 predicted -> 75% actual (a little cautious); every other band is within 1-2 points.
  * the edge over "always say Over" is small: log loss 0.5035 vs 0.5108 for the base rate. Over 1.5 happens about four matches in five, so
    there is little room to be better than the crowd, and bookmaker prices on the Over side are short (about 1.2-1.4).
Optional settings (config.py):
    OU15_HIGH_PROB   = 0.85    # confidence tiers on the probability of the model's pick
    OU15_MEDIUM_PROB = 0.78
"""
import numpy as np

import config

HIGH_PROB = float(getattr(config, "OU15_HIGH_PROB", 0.85))
MEDIUM_PROB = float(getattr(config, "OU15_MEDIUM_PROB", 0.78))
LINE = 1.5
MAX_GOALS = 12


def _pmf(lam, n=MAX_GOALS):
    p = np.empty(n + 1)
    p[0] = np.exp(-lam)
    for k in range(1, n + 1):
        p[k] = p[k - 1] * lam / k
    return p


def probs(lh, la):
    """(P(Over 1.5), P(Under 1.5)) as floats that sum to 1."""
    m = np.outer(_pmf(float(lh)), _pmf(float(la)))
    m /= m.sum()
    i = np.arange(MAX_GOALS + 1)
    under = float(m[(i[:, None] + i[None, :]) <= 1].sum())          # 0-0, 1-0, 0-1
    return 1.0 - under, under


def tier(prob):
    return "HIGH" if prob >= HIGH_PROB else ("MEDIUM" if prob >= MEDIUM_PROB else "LOW")


def _valid(o):
    try:
        return o is not None and float(o) > 1.0
    except (TypeError, ValueError):
        return False


def single(lh, la, over_odds=None, under_odds=None):
    """Prediction dict for one fixture."""
    po, pu = probs(lh, la)
    oo = float(over_odds) if _valid(over_odds) else None
    uo = float(under_odds) if _valid(under_odds) else None
    over = po >= pu
    prob = po if over else pu
    odds = oo if over else uo
    book = None
    if oo and uo:                                                    # bookmaker-implied, margin removed
        io, iu = 1.0 / oo, 1.0 / uo
        book = {"Over": round(100 * io / (io + iu), 1), "Under": round(100 * iu / (io + iu), 1)}
    return {
        "p_over": round(po, 4), "p_under": round(pu, 4),
        "pick": "Over 1.5" if over else "Under 1.5", "pick_key": "Over" if over else "Under", "prob": round(prob, 4), "tier": tier(prob),
        "odds_over": oo, "odds_under": uo, "odds": odds,
        "ev": round(prob * odds - 1, 3) if odds else None,
        "ev_over": round(po * oo - 1, 3) if oo else None, "ev_under": round(pu * uo - 1, 3) if uo else None,
        "book": book,
    }


def ou15_predictions(matches):
    """matches: list of dicts from scan.get_todays_matches. Rows sorted by date."""
    rows = []
    for m in matches:
        if m.get("lh") is None or m.get("la") is None:
            continue
        r = {"event_id": m["event_id"], "date": m["date"], "league_id": m["league_id"], "league_name": m["league_name"],
             "home": m["home"], "away": m["away"], "lh": m["lh"], "la": m["la"], "top_2_scores": m.get("top_2_scores", []),
             "home_apps": m.get("home_apps"), "away_apps": m.get("away_apps")}
        r.update(single(m["lh"], m["la"], m.get("over15_odds"), m.get("under15_odds")))
        rows.append(r)
    rows.sort(key=lambda r: r["date"])
    return rows


if __name__ == "__main__":
    po, pu = probs(1.6, 1.1)
    assert abs(po + pu - 1) < 1e-12 and 0.70 < po < 0.80, po
    # exact check against the closed form: P(Under 1.5) = e^-L (1 + L) with L = lh + la (sum of independent Poissons is Poisson)
    L = 1.6 + 1.1
    assert abs(pu - np.exp(-L) * (1 + L)) < 1e-6
    assert probs(0.4, 0.3)[1] > 0.5 > probs(2.2, 1.6)[1], "low-scoring pairing leans Under, high-scoring leans Over"
    s = single(1.6, 1.1, 1.25, 4.0)
    assert s["pick"] == "Over 1.5" and s["ev"] == round(s["prob"] * 1.25 - 1, 3) and abs(sum(s["book"].values()) - 100) < 0.2
    u = single(0.4, 0.3, 1.8, 2.0)
    assert u["pick"] == "Under 1.5" and u["odds"] == 2.0 and u["tier"] in ("LOW", "MEDIUM", "HIGH")
    assert single(1.6, 1.1)["ev"] is None and single(1.6, 1.1)["book"] is None
    assert tier(0.90) == "HIGH" and tier(0.80) == "MEDIUM" and tier(0.70) == "LOW"
    print("ou15_model self-test passed")
