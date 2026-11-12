# botd.py
"""
Bet of the Day: the model's most confident single selection, among markets with real bookmaker odds between 1.40 and 1.90.

For one day (Nairobi time) it takes every fixture that has odds, looks at every selection the model prices (Over/Under 2.5,
BTTS Yes/No, 1X2 Home/Draw/Away), keeps those whose REAL odds are inside the range, and picks the one with the highest model
probability ("most confident"). Ties go to the better value (expected value), then the earlier kickoff. The next best selections,
one per match, are listed as runners-up.

Where the fixtures come from, by date:
  today / next ~2 days   the saved picks (live odds). Fixtures the API no longer lists because they have kicked off are rebuilt from
                         the odds log, so the day's pick does not change just because a match has started.
  past days              rebuilt from data/odds_log.csv: the last price snapshot logged before each kickoff plus the model's expected
                         goals at that moment, graded against the final score. Bzzoiro does not give pre-match odds for finished
                         matches, so a day can only be rebuilt if the app was running (and logging) before its kickoffs. Until now the log
                         held Over/Under prices only; BTTS and 1X2 prices are logged from now on, so older days can only offer Over/Under.
  later than the saved picks   no odds exist yet, so there is nothing to pick.

Settings (config.py):  BOTD_MIN_ODDS = 1.40   BOTD_MAX_ODDS = 1.90   BOTD_MIN_PROB = 0.0 (optional floor on the probability)
"""
import os
from datetime import datetime, timezone, timedelta

import pandas as pd

import acca_builder as AB
import btts_model as BM
import config
import match_result as MR
from odds_logger import LOG_CSV

MIN_ODDS = float(getattr(config, "BOTD_MIN_ODDS", 1.40))
MAX_ODDS = float(getattr(config, "BOTD_MAX_ODDS", 1.90))
MIN_PROB = float(getattr(config, "BOTD_MIN_PROB", 0.0))
RUNNERS_UP = 5
EAT = timezone(timedelta(hours=3))
MARKET_LABEL = {"ou25": "Over/Under 2.5", "btts": "Both Teams To Score", "1x2": "Match Result 1X2"}
PRICE_COLS = ["over_odds", "under_odds", "btts_yes_odds", "btts_no_odds", "odds_home", "odds_draw", "odds_away"]


# ---------------------------------------------------------------- dates
def _utc(v):
    ts = pd.Timestamp(v)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def eat_date(v):
    try:
        return _utc(v).astimezone(EAT).strftime("%Y-%m-%d")
    except Exception:
        return None


def eat_time(v):
    try:
        return _utc(v).astimezone(EAT).strftime("%H:%M")
    except Exception:
        return ""


# ---------------------------------------------------------------- fixtures
def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(x) else x


def logged_fixtures(iso):
    """Fixtures kicking off on Nairobi day `iso`, rebuilt from the odds log (last snapshot per event), as match dicts."""
    if not os.path.exists(LOG_CSV):
        return []
    try:
        log = pd.read_csv(LOG_CSV)
    except Exception:
        return []
    for c in PRICE_COLS:
        if c not in log.columns:
            log[c] = pd.NA
    log["ko"] = pd.to_datetime(log["kickoff"], utc=True, errors="coerce")
    log = log.dropna(subset=["ko", "lh", "la"])
    if log.empty:
        return []
    start = pd.Timestamp(iso, tz=EAT).tz_convert("UTC")
    log = log[(log["ko"] >= start) & (log["ko"] < start + pd.Timedelta(days=1))]
    log = log.sort_values("logged_at").drop_duplicates("event_id", keep="last")
    out = []
    for _, r in log.iterrows():
        po = float(r["p_over"]) * 100.0
        out.append({
            "event_id": int(r["event_id"]), "date": r["ko"].isoformat(), "league_id": int(r["league_id"]),
            "league_name": config.league_name(int(r["league_id"])), "home": r["home"], "away": r["away"],
            "lh": float(r["lh"]), "la": float(r["la"]), "po": po, "pu": 100.0 - po,
            "over_odds": _num(r["over_odds"]), "under_odds": _num(r["under_odds"]),
            "btts_yes_odds": _num(r["btts_yes_odds"]), "btts_no_odds": _num(r["btts_no_odds"]),
            "odds_home": _num(r["odds_home"]), "odds_draw": _num(r["odds_draw"]), "odds_away": _num(r["odds_away"]),
            "source": "log",
        })
    return out


def day_pool(iso, live_matches):
    """Live fixtures (saved picks) on that day, plus logged ones the live list no longer has."""
    live = [dict(m, source="live") for m in live_matches if eat_date(m.get("date")) == iso]
    seen = {m["event_id"] for m in live}
    return live + [m for m in logged_fixtures(iso) if m["event_id"] not in seen]


# ---------------------------------------------------------------- selection
def candidates(m):
    """Every selection of this fixture with real odds inside the range (and above the optional probability floor)."""
    out = []
    for market, sels in AB.selections_for(m).items():
        for s in sels:
            o = s["odds"]
            if o is None or not (MIN_ODDS <= o <= MAX_ODDS) or s["prob"] < MIN_PROB:
                continue
            out.append(dict(s, ev=round(s["prob"] * o - 1, 4), fair_odds=round(1.0 / s["prob"], 2),
                            tier=BM.tier(s["prob"]), event_id=m["event_id"], match=m))
    return out


def choose(pool):
    """(best, runners_up, n_candidates, n_matches_with_candidates)."""
    cands = []
    for m in pool:
        cands.extend(candidates(m))
    if not cands:
        return None, [], 0, 0
    cands.sort(key=lambda c: (-c["prob"], -c["ev"], str(c["match"]["date"]), c["event_id"]))
    best = cands[0]
    runners, seen = [], {best["event_id"]}
    for c in cands[1:]:
        if c["event_id"] in seen:
            continue                               # one runner-up per match
        seen.add(c["event_id"])
        runners.append(c)
        if len(runners) >= RUNNERS_UP:
            break
    return best, runners, len(cands), len({c["event_id"] for c in cands})


def outcome(market, side, hg, ag):
    """True / False: did the selection win, given the final score."""
    if market == "ou25":
        return (hg + ag > 2) == (side == "Over")
    if market == "btts":
        return ((hg > 0 and ag > 0) == (side == "Yes"))
    if market == "1x2":
        return {"Home": hg > ag, "Draw": hg == ag, "Away": hg < ag}[side]
    raise ValueError(market)


def finished_scores():
    """{event_id: (home_goals, away_goals)} for every finished match in matches.csv."""
    import shots_data as SD
    df = SD._matches_table()
    if df.empty:
        return {}
    return {int(e): (int(h), int(a)) for e, h, a in zip(df["event_id"], df["home_goals"], df["away_goals"])}


# ---------------------------------------------------------------- presentation
def _breakdown(m):
    """The three markets for the fixture (probabilities from its expected goals, with the real prices), like the Single Match card."""
    import model
    lh, la = m["lh"], m["la"]
    b = BM.single(lh, la, m.get("btts_yes_odds"), m.get("btts_no_odds"))
    x = MR.single(lh, la, m.get("odds_home"), m.get("odds_draw"), m.get("odds_away"))
    po, pu, top = model.over_under_2_5(lh, la)
    return {
        "lh": round(lh, 2), "la": round(la, 2),
        # the same stored probabilities the pick itself was chosen with, so the card never disagrees with its own headline
        "ou": {"p_over": round(m["po"] / 100.0, 4) if m.get("po") is not None else round(po, 4),
               "p_under": round(m["pu"] / 100.0, 4) if m.get("pu") is not None else round(pu, 4),
               "odds_over": m.get("over_odds"), "odds_under": m.get("under_odds")},
        "btts": {"p_yes": b["p_yes"], "p_no": b["p_no"], "odds_yes": m.get("btts_yes_odds"), "odds_no": m.get("btts_no_odds")},
        "x12": {"p_home": x["p_home"], "p_draw": x["p_draw"], "p_away": x["p_away"], "odds_home": x["odds_home"],
                "odds_draw": x["odds_draw"], "odds_away": x["odds_away"], "book": x["book"]},
        "top_2_scores": [{"score": f"{h}-{a}", "prob": round(p * 100, 1)} for (h, a), p in top[:2]],
    }


def _card(c, scores, with_breakdown):
    m = c["match"]
    row = {
        "event_id": c["event_id"], "home": m["home"], "away": m["away"], "league": m.get("league_name") or "",
        "kickoff": eat_time(m["date"]), "kickoff_utc": _utc(m["date"]).isoformat(), "source": m.get("source", "live"),
        "market": c["market"], "market_label": MARKET_LABEL[c["market"]], "side": c["side"], "label": c["label"],
        "prob": round(c["prob"], 4), "odds": c["odds"], "ev": c["ev"], "fair_odds": c["fair_odds"], "tier": c["tier"],
        "result": None, "score": None,
    }
    sc = scores.get(c["event_id"])
    if sc is not None:
        row["score"] = f"{sc[0]}-{sc[1]}"
        row["result"] = "won" if outcome(c["market"], c["side"], sc[0], sc[1]) else "lost"
    if with_breakdown:
        row["breakdown"] = _breakdown(m)
    return row


def day_view(iso, live_matches, now=None, scores=None):
    """Everything the Bet of the Day tab shows for one Nairobi day."""
    now = now or datetime.now(timezone.utc)
    today = now.astimezone(EAT).strftime("%Y-%m-%d")
    when = "past" if iso < today else ("today" if iso == today else "future")
    scores = finished_scores() if scores is None else scores
    pool = day_pool(iso, live_matches)
    view = {"date": iso, "when": when, "criteria": {"min_odds": MIN_ODDS, "max_odds": MAX_ODDS, "min_prob": MIN_PROB},
            "fixtures": len(pool), "candidates": 0, "matches_with_candidates": 0, "pick": None, "runners": [],
            "message": "", "note": ""}
    if not pool:
        if when == "past":
            view["message"] = ("No odds were logged for this day, so no Bet of the Day can be rebuilt. Bzzoiro does not provide pre-match odds "
                               "for finished matches; the app records them while it is running and refreshing.")
        elif when == "today":
            view["message"] = "No fixtures with odds for today."
        else:
            view["message"] = ("No odds yet for this day. Odds are loaded for fixtures in roughly the next two days "
                               "(tap Refresh on All Matches to update them), so later dates stay empty until then.")
        return view
    best, runners, n_c, n_m = choose(pool)
    view.update(candidates=n_c, matches_with_candidates=n_m)
    logged = [m for m in pool if m.get("source") == "log"]
    if logged and not any(m.get("btts_yes_odds") or m.get("odds_home") for m in logged):
        view["note"] = "Only Over/Under prices were logged for this day, so only Over/Under selections could be considered."
    if best is None:
        view["message"] = (f"{len(pool)} fixture{'s' if len(pool) != 1 else ''} with odds, but none has a selection priced between "
                           f"{MIN_ODDS:.2f} and {MAX_ODDS:.2f}.")
        return view
    view["pick"] = _card(best, scores, with_breakdown=True)
    view["runners"] = [_card(c, scores, with_breakdown=False) for c in runners]
    return view


# ---------------------------------------------------------------- self-test
def _selftest():
    now = datetime(2026, 11, 10, 10, 0, tzinfo=timezone.utc)

    def fx(i, hours, lh=1.6, la=1.2, po=55.0, **odds):
        d = {"event_id": i, "date": (now + timedelta(hours=hours)).isoformat(), "league_name": "L", "league_id": 1, "home": f"H{i}",
             "away": f"A{i}", "lh": lh, "la": la, "po": po, "pu": 100 - po, "over_odds": None, "under_odds": None,
             "btts_yes_odds": None, "btts_no_odds": None}
        d.update(odds)
        return d

    # in-range filter is inclusive at both ends, and prices outside it are ignored
    a = fx(1, 3, over_odds=1.40, under_odds=2.9)                     # Over at exactly 1.40 (in range), Under at 2.9 (out)
    assert [c["label"] for c in candidates(a)] == ["Over 2.5"]
    b = fx(2, 3, over_odds=1.39, under_odds=1.91)                    # both just outside
    assert candidates(b) == []
    c_ = fx(3, 3, over_odds=1.90, under_odds=1.20)
    assert [c["label"] for c in candidates(c_)] == ["Over 2.5"]
    # no real 1X2 odds -> never a 1X2 selection from a model price
    assert all(c["market"] != "1x2" for c in candidates(fx(4, 3, over_odds=1.7, under_odds=2.0)))

    # the pick is the HIGHEST probability among in-range selections, not the best price
    p1 = fx(10, 3, po=80.0, over_odds=1.85, under_odds=3.5)         # Over 80% @1.85
    p2 = fx(11, 4, po=62.0, over_odds=1.45, under_odds=3.0)         # Over 62% @1.45
    p3 = fx(12, 5, po=90.0, over_odds=1.30, under_odds=4.0)         # Over 90% @1.30 -> out of range, must be ignored
    v = day_view("2026-11-10", [p1, p2, p3], now=now, scores={})
    assert v["pick"]["event_id"] == 10 and v["pick"]["label"] == "Over 2.5" and abs(v["pick"]["prob"] - 0.80) < 1e-9
    assert v["when"] == "today" and v["fixtures"] == 3 and v["runners"][0]["event_id"] == 11, v
    assert all(r["event_id"] != 10 for r in v["runners"]), "runners-up exclude the pick's match"
    # one runner-up per match even when a match has several in-range selections
    two = fx(20, 3, po=70.0, over_odds=1.60, under_odds=1.85, btts_yes_odds=1.7, btts_no_odds=1.9)
    other = fx(21, 4, po=64.0, over_odds=1.55, under_odds=2.4)
    v = day_view("2026-11-10", [two, other], now=now, scores={})
    assert len({r["event_id"] for r in v["runners"]}) == len(v["runners"])
    # a tie on probability goes to the better value
    t1 = fx(30, 3, po=70.0, over_odds=1.50, under_odds=3.0); t2 = fx(31, 4, po=70.0, over_odds=1.80, under_odds=3.0)
    assert day_view("2026-11-10", [t1, t2], now=now, scores={})["pick"]["event_id"] == 31
    # no candidates / no fixtures / future / past messages
    v = day_view("2026-11-10", [b], now=now, scores={})
    assert v["pick"] is None and "none has a selection priced between 1.40 and 1.90" in v["message"]
    assert "No odds yet" in day_view("2026-11-20", [], now=now, scores={})["message"]
    # grading
    assert outcome("ou25", "Over", 2, 1) and not outcome("ou25", "Under", 2, 1) and outcome("btts", "Yes", 1, 1)
    assert outcome("btts", "No", 2, 0) and outcome("1x2", "Draw", 1, 1) and outcome("1x2", "Away", 0, 1) and not outcome("1x2", "Home", 0, 1)
    v = day_view("2026-11-10", [p1, p2], now=now, scores={10: (2, 1), 11: (0, 0)})
    assert v["pick"]["result"] == "won" and v["pick"]["score"] == "2-1" and v["runners"][0]["result"] == "lost"
    # the card's breakdown shows the same probability as the headline pick
    v = day_view("2026-11-10", [p1], now=now, scores={})
    assert abs(v["pick"]["breakdown"]["ou"]["p_over"] - v["pick"]["prob"]) < 1e-9
    # a 1X2 selection with real odds can win the day
    x = fx(40, 3, lh=2.4, la=0.7, po=60.0, odds_home=1.55, odds_draw=4.5, odds_away=7.0)
    v = day_view("2026-11-10", [x], now=now, scores={})
    assert v["pick"]["market"] == "1x2" and v["pick"]["side"] == "Home" and v["pick"]["label"] == "H40 to win"
    print("botd self-test passed")


if __name__ == "__main__":
    _selftest()
