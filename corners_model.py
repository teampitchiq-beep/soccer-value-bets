# corners_model.py
"""
Corners: expected corners per team and the probability of the match going Over / Under the corners line (default 7.5).

  Model     the same ridge for/against engine as the goals, xG and shots models, fitted on corners (corner_kicks from the stats endpoint,
            recency-weighted, opponent-adjusted, with its own baseline and home advantage per league): a team's corners-for strength against
            the opponent's corners-conceded weakness gives its expected corners, and the two add up to the expected match total (the "mean").
            A negative binomial, whose spread is measured out of sample on the last 120 days, turns that mean into P(total corners > line).
            It is the same engine as the Corners cards in the Trends tab (match_stats.stat_model), so the two always agree.
  Line      CORNERS_LINE = 7.5 in config.py: "Over 7.5" means 8 or more corners in the match.
  Odds      NONE. Bzzoiro's free odds feed has no corners prices (they exist only in the paid comparison endpoint), so this is a model-only market:
            it shows the model's fair price, never a bookmaker price or a value figure, and corners cannot be used in the Acca builder,
            Bet of the Day or acca tracking, which all need a real price.
  Data      needs corner stats for at least MIN_FIT_MATCHES (200) finished matches. They come from the stats loader (up to STATS_DAILY_BUDGET a
            day), so a fresh install fills over a few days; until then the tab says how many it has.

Check the calibration on YOUR data before trusting it:   python corners_backtest.py

Optional settings (config.py):
    CORNERS_LINE         = 7.5
    CORNERS_HIGH_PROB    = 0.85    # confidence tiers on the probability of the model's pick
    CORNERS_MEDIUM_PROB  = 0.75
"""
from datetime import datetime, timezone

import pandas as pd

import config
import match_stats as MS
import shots_data as SD

LINE = float(getattr(config, "CORNERS_LINE", 7.5))
HIGH_PROB = float(getattr(config, "CORNERS_HIGH_PROB", 0.85))
MEDIUM_PROB = float(getattr(config, "CORNERS_MEDIUM_PROB", 0.75))
MIN_FIT_MATCHES = MS.MIN_FIT_MATCHES


def tier(prob):
    return "HIGH" if prob >= HIGH_PROB else ("MEDIUM" if prob >= MEDIUM_PROB else "LOW")


def _context(df=None, store=None):
    df = SD._matches_table() if df is None else df
    store = MS.load_store() if store is None else store
    return df, store


def coverage(store=None):
    """How many finished matches have corner counts for both teams."""
    store = MS.load_store() if store is None else store
    if store is None or store.empty or "corners_h" not in store:
        return 0
    return int(store[["corners_h", "corners_a"]].dropna().shape[0])


def _model(df, store, now=None):
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    return MS.stat_model(df, store, "corners", now)


def fixture(mdl, home_id, away_id, league_id, line=None):
    """Prediction dict for one fixture, or None when either team has no corner history in the fit."""
    from model import lambdas_from_fit
    line = LINE if line is None else line
    fit = mdl["fit"]
    if home_id not in fit["tidx"] or away_id not in fit["tidx"]:
        return None
    mh, ma = lambdas_from_fit(fit, home_id, away_id, league_id)
    mu = mh + ma
    p_over = MS.p_line(mu, mdl["k_total"], line, "Over")
    over = p_over >= 0.5
    prob = p_over if over else 1.0 - p_over
    return {
        "line": line, "xc_home": round(mh, 1), "xc_away": round(ma, 1), "xc_total": round(mu, 1),
        "p_over": round(p_over, 4), "p_under": round(1.0 - p_over, 4),
        "fair_over": round(1.0 / p_over, 2) if p_over > 0 else None,
        "fair_under": round(1.0 / (1.0 - p_over), 2) if p_over < 1 else None,
        "pick": f"{'Over' if over else 'Under'} {line:g}", "pick_key": "Over" if over else "Under",
        "prob": round(prob, 4), "tier": tier(prob), "spread_k": round(mdl["k_total"], 1),
    }


def single(home_id, away_id, league_id, df=None, store=None):
    """For the Single Match card: (prediction or None, info)."""
    df, store = _context(df, store)
    n = coverage(store)
    info = {"ready": False, "line": LINE, "matches_with_corners": n, "needed": MIN_FIT_MATCHES, "message": ""}
    if df.empty or n < MIN_FIT_MATCHES:
        info["message"] = (f"Corners need stats for at least {MIN_FIT_MATCHES} finished matches; {n} so far. "
                           "The stats loader fills about 1,000 a day (Automation panel).")
        return None, info
    mdl = _model(df, store)
    if mdl is None:
        info["message"] = "Not enough corner data to fit the model yet."
        return None, info
    info["ready"] = True
    pred = fixture(mdl, int(home_id), int(away_id), int(league_id))
    if pred is None:
        info["message"] = "One of the teams has no corner history yet."
    return pred, info


def corner_predictions(matches, df=None, store=None):
    """Rows for the Corners view (every fixture in `matches`, sorted by date) and an info dict."""
    df, store = _context(df, store)
    n = coverage(store)
    info = {"ready": False, "line": LINE, "matches_with_corners": n, "needed": MIN_FIT_MATCHES, "message": "", "covered": 0, "k": None}
    if df.empty or n < MIN_FIT_MATCHES:
        info["message"] = (f"Corners need stats for at least {MIN_FIT_MATCHES} finished matches; {n} so far. "
                           "The stats loader fills about 1,000 a day, so this fills in over a few days (see the Automation panel).")
        return [], info
    mdl = _model(df, store)
    if mdl is None:
        info["message"] = "Not enough corner data to fit the model yet."
        return [], info
    info.update(ready=True, k=round(mdl["k_total"], 1))
    rows = []
    for m in matches:
        try:
            hid, aid, lid = int(m["home_id"]), int(m["away_id"]), int(m["league_id"])
        except (KeyError, TypeError, ValueError):
            continue
        p = fixture(mdl, hid, aid, lid)
        r = {"event_id": m["event_id"], "date": m["date"], "league_id": m["league_id"], "league_name": m["league_name"],
             "home": m["home"], "away": m["away"], "ready": p is not None}
        if p:
            r.update(p)
            info["covered"] += 1
        rows.append(r)
    rows.sort(key=lambda r: r["date"])
    return rows, info


# ---------------------------------------------------------------- self-test
def _selftest():
    import numpy as np
    from scipy.stats import nbinom
    rng = np.random.default_rng(5)
    # a synthetic league: 12 teams, 600 matches with team-specific corner strengths
    teams = list(range(100, 112))
    att = {t: rng.normal(0, 0.18) for t in teams}
    dfn = {t: rng.normal(0, 0.12) for t in teams}
    rows, store_rows = [], {}
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    for i in range(600):
        h, a = rng.choice(teams, 2, replace=False)
        ch = rng.poisson(5.6 * np.exp(att[h] - dfn[a])); ca = rng.poisson(4.6 * np.exp(att[a] - dfn[h]))
        rows.append({"event_id": 5000 + i, "date": t0 + pd.Timedelta(days=i * 0.5), "league_id": 1, "home_team_id": int(h), "away_team_id": int(a),
                     "home_team": f"T{h}", "away_team": f"T{a}", "home_goals": 1, "away_goals": 1})
        store_rows[5000 + i] = {"corners_h": ch, "corners_a": ca}
    df = pd.DataFrame(rows)
    store = pd.DataFrame.from_dict(store_rows, orient="index").reindex(columns=MS.COLS)
    assert coverage(store) == 600

    # not enough data -> clear message, never a made-up number
    small = store.iloc[:50]
    pr, info = corner_predictions([], df, small)
    assert pr == [] and not info["ready"] and "50 so far" in info["message"]

    mdl = _model(df, store, now=df["date"].max() + pd.Timedelta(days=1))
    assert mdl is not None and 3.0 <= mdl["k_total"] <= 300.0
    p = fixture(mdl, 100, 101, 1)
    assert p["line"] == 7.5 and abs(p["p_over"] + p["p_under"] - 1) < 1e-3
    assert abs(p["xc_total"] - (p["xc_home"] + p["xc_away"])) < 0.11, "the match mean is the two team means added up"
    # the probability is exactly the negative-binomial tail at the line: Over 7.5 = 8 or more corners
    mu = p["xc_total"]
    assert abs(MS.p_line(mu, mdl["k_total"], 7.5, "Over") - (1 - nbinom.cdf(7, mdl["k_total"], mdl["k_total"] / (mdl["k_total"] + mu)))) < 1e-12
    # a team that has never been seen is not predicted at all
    assert fixture(mdl, 100, 9999, 1) is None
    # a higher mean means a higher chance of Over (monotonic)
    assert MS.p_line(11.0, mdl["k_total"], 7.5, "Over") > MS.p_line(8.0, mdl["k_total"], 7.5, "Over")
    # the line is configurable and "Over 8.5" is harder than "Over 7.5"
    assert fixture(mdl, 100, 101, 1, line=8.5)["p_over"] < p["p_over"]
    # rows keep fixtures without data, flagged not ready
    ms = [{"event_id": 1, "date": "2026-12-01T18:00:00+00:00", "league_id": 1, "league_name": "L", "home": "A", "away": "B", "home_id": 100, "away_id": 101},
          {"event_id": 2, "date": "2026-12-01T16:00:00+00:00", "league_id": 1, "league_name": "L", "home": "C", "away": "D", "home_id": 9998, "away_id": 9999}]
    rows_, info = corner_predictions(ms, df, store)
    assert info["ready"] and info["covered"] == 1 and [r["event_id"] for r in rows_] == [2, 1] and rows_[0]["ready"] is False and rows_[1]["ready"] is True
    pred, i2 = single(100, 101, 1, df, store)
    assert pred and pred["pick"] in ("Over 7.5", "Under 7.5") and i2["ready"]
    assert tier(0.9) == "HIGH" and tier(0.8) == "MEDIUM" and tier(0.6) == "LOW"
    print("corners_model self-test passed")


if __name__ == "__main__":
    _selftest()
