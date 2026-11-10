# acca_builder.py
"""
Accumulator (acca) builder: pick ONE market to focus on, get the best available selections for a day,
using only REAL bookmaker odds from today's saved fixtures. No API requests are made here.

How a leg qualifies (every one must hold):
  1. the fixture has real bookmaker odds for that selection (model-made prices are never used)
  2. model probability >= min_prob
  3. real odds >= min_odds
  4. the match has not kicked off yet (its odds would be stale)

How the acca is chosen from the qualifying legs (one leg per match, its best side):
  * it has between `min_legs` and `legs` matches (2 or 3 at most: MIN_LEGS = 2, MAX_LEGS = 3)
  * its COMBINED odds must reach min_combined (default 2.0)
  * among all combinations that satisfy both, the one with the HIGHEST combined probability wins
So when the most probable combination falls short of the combined-odds target, the builder scales down to the most
probable combination that does reach it (swapping in lower-probability, higher-paying legs, or using 3 legs instead of 2).
The search is exact (a Pareto dynamic programme), not a heuristic. There is no value / expected-value test.

Markets: ou25 (Over/Under 2.5), btts (Both Teams To Score Yes/No) and 1x2 (Match Result: Home / Draw / Away). Nothing else.
For 1x2 the default minimum probability is lower (45%) because even a strong favourite rarely reaches 58%; the real 1X2 odds
come from the same odds request as the other markets (home_win / draw / away_win).

IMPORTANT: acca odds multiply, and so do the risks. The "all legs win" chance below assumes the legs are
independent and uses the model's probabilities, which in backtests were NOT better than the bookmaker's own.
A higher combined price is not better value. Not betting advice.

Optional defaults (config.py, already present from your older acca settings):
    ACCA_MIN_LEG_PROB = 0.58     # minimum model probability per leg (Over/Under, BTTS)
    ACCA_MIN_LEG_PROB_1X2 = 0.45 # the same for the 1X2 market
    ACCA_MIN_LEG_ODDS = 1.15     # minimum real odds per leg
    ACCA_MIN_ODDS     = 2.0      # minimum COMBINED odds of the acca
    ACCA_MAX_LEGS     = 3        # default maximum number of matches (2 or 3)
"""
from datetime import datetime, timezone, timedelta

import pandas as pd

import btts_model as BM
import config
import match_result as MR

EAT = timezone(timedelta(hours=3))

DEFAULT_MIN_PROB = float(getattr(config, "ACCA_MIN_LEG_PROB", 0.58))
DEFAULT_MIN_ODDS = float(getattr(config, "ACCA_MIN_LEG_ODDS", 1.15))
MIN_LEGS = 2                                      # an acca has at least 2 matches ...
MAX_LEGS = 3                                      # ... and at most 3
DEFAULT_MIN_COMBINED = float(getattr(config, "ACCA_MIN_ODDS", 2.0))
DEFAULT_LEGS = max(MIN_LEGS, min(MAX_LEGS, int(getattr(config, "ACCA_MAX_LEGS", 3))))

# market key -> (title, sides in display order)
MARKETS = {
    "ou25": ("Over/Under 2.5", ("Over", "Under")),
    "btts": ("Both Teams To Score", ("Yes", "No")),
    "1x2": ("Match Result 1X2", ("Home", "Draw", "Away")),
}
# the minimum model probability a leg needs, per market (1X2 outcomes are spread over three results, so the bar is lower)
MARKET_MIN_PROB = {"ou25": DEFAULT_MIN_PROB, "btts": DEFAULT_MIN_PROB,
                   "1x2": float(getattr(config, "ACCA_MIN_LEG_PROB_1X2", 0.45))}


def default_min_prob(market):
    return MARKET_MIN_PROB.get(market, DEFAULT_MIN_PROB)


# ------------------------------------------------------------------ helpers
def _eat(value):
    """Timestamp-like -> aware datetime in EAT (UTC+3), or None."""
    try:
        ts = pd.Timestamp(value)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        return ts.to_pydatetime().astimezone(EAT)
    except Exception:
        return None


def _valid_odds(o):
    try:
        return o is not None and float(o) > 1.0
    except (TypeError, ValueError):
        return False


def _sel(market, side, label, prob, odds):
    odds = float(odds) if _valid_odds(odds) else None
    return {"market": market, "side": side, "label": label, "prob": float(prob), "odds": odds}


def selections_for(m):
    """All selections for one fixture dict (from scan.get_todays_matches), grouped by market key."""
    out = {k: [] for k in MARKETS}

    # Over/Under 2.5 (stored percentages + real odds)
    po, pu = m.get("po"), m.get("pu")
    if po is not None and pu is not None:
        out["ou25"].append(_sel("ou25", "Over", "Over 2.5", po / 100.0, m.get("over_odds")))
        out["ou25"].append(_sel("ou25", "Under", "Under 2.5", pu / 100.0, m.get("under_odds")))

    lh, la = m.get("lh"), m.get("la")
    if lh is None or la is None:
        return out

    # BTTS
    yes_o, no_o = m.get("btts_yes_odds"), m.get("btts_no_odds")
    b = BM.single(lh, la, yes_o, no_o)
    out["btts"].append(_sel("btts", "Yes", "BTTS Yes", b["p_yes"], yes_o))
    out["btts"].append(_sel("btts", "No", "BTTS No", b["p_no"], no_o))

    # Match result 1X2 (model probabilities from the same expected goals + real 1X2 odds when the fixture has them)
    r = MR.single(lh, la, m.get("odds_home"), m.get("odds_draw"), m.get("odds_away"))
    out["1x2"].append(_sel("1x2", "Home", f"{m.get('home', 'Home')} to win", r["p_home"], r["odds_home"]))
    out["1x2"].append(_sel("1x2", "Draw", "Draw", r["p_draw"], r["odds_draw"]))
    out["1x2"].append(_sel("1x2", "Away", f"{m.get('away', 'Away')} to win", r["p_away"], r["odds_away"]))

    return out


def days_available(matches, now=None):
    """Days (EAT dates) that still have not-yet-started fixtures: [{"iso", "label", "count"}, ...]."""
    now = now or datetime.now(timezone.utc)
    today = now.astimezone(EAT).date()
    counts = {}
    for m in matches:
        t = _eat(m.get("date"))
        if t is None or t.astimezone(timezone.utc) <= now:
            continue
        counts[t.date()] = counts.get(t.date(), 0) + 1
    days = []
    for d in sorted(counts):
        name = "Today" if d == today else ("Tomorrow" if d == today + timedelta(days=1) else d.strftime("%a %d %b"))
        days.append({"iso": d.isoformat(), "label": f"{name} · {d.strftime('%d %b')}" if name in ("Today", "Tomorrow") else name,
                     "count": counts[d]})
    return days


# ------------------------------------------------------------------ exact combination search
def _pareto(points):
    """Keep only points that no other point beats on BOTH sum-log-odds and sum-log-probability."""
    points.sort(key=lambda t: (-t[0], -t[1]))
    out, best_b = [], float("-inf")
    for a, b, idx in points:
        if b > best_b + 1e-15:
            out.append((a, b, idx))
            best_b = b
    return out


def best_combination(pool, min_legs, max_legs, min_combined_odds):
    """
    Exact search. pool: list of dicts with "prob" and "odds". Returns the indices of the combination with
    min_legs..max_legs legs, combined odds >= min_combined_odds and the HIGHEST combined probability, or None.
    Works in logs (sums), keeping a Pareto frontier of (odds, probability) per leg count.
    """
    import math
    target = math.log(min_combined_odds)
    front = [[] for _ in range(max_legs + 1)]
    front[0] = [(0.0, 0.0, ())]
    for i, s in enumerate(pool):
        lo, lp = math.log(s["odds"]), math.log(max(s["prob"], 1e-12))
        for k in range(max_legs - 1, -1, -1):              # downwards: each leg is used at most once
            if front[k]:
                front[k + 1] = _pareto(front[k + 1] + [(a + lo, b + lp, idx + (i,)) for a, b, idx in front[k]])
    best = None
    for k in range(min_legs, max_legs + 1):
        for a, b, idx in front[k]:
            if a >= target - 1e-12 and (best is None or (b, a) > (best[1], best[0])):
                best = (a, b, idx)
    return None if best is None else list(best[2])


# ------------------------------------------------------------------ the builder
def build_acca(matches, market, side="best", day=None, legs=None, min_prob=None, min_odds=None,
               min_combined=None, min_legs=None, now=None):
    """
    Returns a dict. Always has "ok", "message", "market_title", "day", "funnel", "params".
    When ok is True it also has "legs" (list), "combined" (odds, prob, implied_prob) and "note".
    `legs` is the MAXIMUM number of matches and `min_legs` the minimum (both kept within 2 to 3).
    legs=3, min_legs=3 means exactly 3; legs=2 means exactly 2; the defaults mean the best of 2 or 3.
    """
    if market not in MARKETS:
        raise ValueError(f"unknown market {market!r}; choose from {', '.join(MARKETS)}")
    title, sides = MARKETS[market]
    side_in = (side or "best").strip().lower()
    if side_in != "best":
        match_side = [s for s in sides if s.lower() == side_in]
        if not match_side:
            raise ValueError(f"unknown side {side!r} for {market}; choose from best, {', '.join(sides)}")
        allowed = tuple(match_side)
    else:
        allowed = sides

    legs = DEFAULT_LEGS if legs is None else max(MIN_LEGS, min(int(legs), MAX_LEGS))
    lo = MIN_LEGS if min_legs is None else max(MIN_LEGS, min(int(min_legs), legs))
    min_prob = default_min_prob(market) if min_prob is None else float(min_prob)
    min_odds = DEFAULT_MIN_ODDS if min_odds is None else float(min_odds)
    min_combined = DEFAULT_MIN_COMBINED if min_combined is None else max(1.01, float(min_combined))
    now = now or datetime.now(timezone.utc)

    days = days_available(matches, now)
    if day is None and days:
        day = days[0]["iso"]
    day_label = next((d["label"] for d in days if d["iso"] == day), day or "the selected day")
    focus = f"{title}" + ("" if side_in == "best" else f" · {allowed[0]}")

    params = {"market": market, "side": side_in, "day": day, "max_legs": legs, "min_legs": lo,
              "min_prob": min_prob, "min_odds": min_odds, "min_combined": min_combined}
    funnel = {"fixtures": 0, "with_odds": 0, "min_prob": 0, "min_odds": 0}
    base = {"market_title": focus, "day": day_label, "funnel": funnel, "params": params,
            "legs": [], "combined": None, "note": ""}
    fail = f"No matches fulfilled the acca criteria for {focus} on {day_label}."

    pool = []
    for m in matches:
        t = _eat(m.get("date"))
        if t is None or t.astimezone(timezone.utc) <= now or (day and t.date().isoformat() != day):
            continue
        funnel["fixtures"] += 1
        sels = [s for s in selections_for(m)[market] if s["side"] in allowed and s["odds"] is not None]
        if not sels:
            continue
        funnel["with_odds"] += 1
        sels = [s for s in sels if s["prob"] >= min_prob]
        if not sels:
            continue
        funnel["min_prob"] += 1
        sels = [s for s in sels if s["odds"] >= min_odds]
        if not sels:
            continue
        funnel["min_odds"] += 1
        best = max(sels, key=lambda s: (s["prob"], s["odds"]))
        pool.append({**best, "event_id": m.get("event_id"), "league": m.get("league_name"),
                     "home": m.get("home"), "away": m.get("away"),
                     "kickoff": t.strftime("%H:%M"), "kickoff_iso": t.isoformat()})

    pool.sort(key=lambda s: (s["prob"], s["odds"]), reverse=True)
    q = len(pool)

    if q < lo:
        if funnel["fixtures"] == 0:
            why = "There are no upcoming fixtures on that day."
        elif funnel["with_odds"] == 0:
            why = "None of the fixtures have real bookmaker odds for this market yet (tap ↻ Refresh once to load them)."
        else:
            why = (f"Only {q} selection{'s' if q != 1 else ''} qualified, and an acca needs at least {lo} matches. "
                   "Try a lower minimum probability or minimum odds, another market or side, or another day.")
        return {**base, "ok": False, "message": f"{fail} {why}"}

    chosen = best_combination(pool, lo, legs, min_combined)
    if chosen is None:
        top_odds = 1.0
        for s in sorted(pool, key=lambda s: s["odds"], reverse=True)[:legs]:
            top_odds *= s["odds"]
        return {**base, "ok": False, "message":
                f"{fail} {q} selections qualified, but no combination of {lo if legs == lo else f'{lo} to {legs}'} of them reaches combined odds "
                f"of {min_combined:.2f} (the best possible is {top_odds:.2f}). Lower the minimum "
                f"probability so higher-paying legs qualify, or lower the combined-odds target."}

    picked = sorted((pool[i] for i in chosen), key=lambda s: (s["prob"], s["odds"]), reverse=True)
    prod_odds = prod_prob = 1.0
    for s in picked:
        prod_odds *= s["odds"]
        prod_prob *= s["prob"]

    ref_odds = 1.0                                   # the plain most-probable combination of the smallest size
    for s in pool[:lo]:
        ref_odds *= s["odds"]
    scaled = ref_odds < min_combined - 1e-12
    note = (f"The most probable {lo}-leg combination only reaches {ref_odds:.2f}, below your {min_combined:.2f} "
            f"minimum, so this is the most probable combination that does reach it." if scaled else "")
    return {**base, "ok": True, "legs": picked, "note": note, "scaled_down": scaled,
            "message": f"{len(picked)}-leg acca on {focus}, {day_label}.",
            "combined": {"legs": len(picked), "odds": round(prod_odds, 2), "prob": prod_prob,
                         "implied_prob": 1.0 / prod_odds}}


# ------------------------------------------------------------------ self-test
def _selftest():
    import itertools
    import random
    now = datetime(2026, 10, 29, 10, 0, tzinfo=timezone.utc)

    def fx(i, hours, po=70.0, over_o=1.60, btts_yes=1.80, lh=1.8, la=1.4):
        return {"event_id": i, "date": (now + timedelta(hours=hours)).isoformat(), "league_name": "Test League",
                "home": f"H{i}", "away": f"A{i}", "lh": lh, "la": la, "po": po, "pu": 100 - po,
                "over_odds": over_o, "under_odds": 2.40, "btts_yes_odds": btts_yes, "btts_no_odds": 2.0}

    assert (MIN_LEGS, MAX_LEGS) == (2, 3)
    ids = lambda r: sorted(l["event_id"] for l in r["legs"])

    # 1. exact search == brute force on random pools, for every allowed leg range
    rnd = random.Random(7)
    for trial in range(400):
        n = rnd.randint(2, 9)
        pool = [{"prob": rnd.uniform(0.3, 0.95), "odds": rnd.uniform(1.05, 2.6)} for _ in range(n)]
        lo = rnd.choice([2, 3]); mx = rnd.randint(lo, 3); target = rnd.uniform(1.4, 6.0)
        got = best_combination(pool, lo, mx, target)
        best = None
        for k in range(lo, mx + 1):
            for idx in itertools.combinations(range(n), k):
                o = p = 1.0
                for i in idx:
                    o *= pool[i]["odds"]; p *= pool[i]["prob"]
                if o >= target - 1e-12 and (best is None or p > best[0]):
                    best = (p, idx)
        if best is None:
            assert got is None, trial
        else:
            gp = 1.0
            for i in got:
                gp *= pool[i]["prob"]
            assert got is not None and abs(gp - best[0]) < 1e-12 and lo <= len(got) <= mx, (trial, gp, best)

    # 2. a hand-checked case. Probabilities/odds: 83/1.15, 82/1.16, 81/1.20, 74/1.55, 72/1.60, 70/1.65
    short = [fx(1, 2, po=83, over_o=1.15), fx(2, 3, po=82, over_o=1.16), fx(3, 4, po=81, over_o=1.20),
             fx(4, 5, po=74, over_o=1.55), fx(5, 6, po=72, over_o=1.60), fx(6, 7, po=70, over_o=1.65)]
    kw = dict(min_prob=0.6, min_odds=1.1, min_combined=2.0, now=now)
    r = build_acca(short, "ou25", "Over", **kw)                         # best of 2 or 3
    assert ids(r) == [4, 5] and r["scaled_down"] and r["combined"]["odds"] >= 2.0, (ids(r), r["message"])
    assert "1.33" in r["note"] and "2-leg" in r["note"] and "most probable combination that does reach it" in r["note"]
    assert ids(build_acca(short, "ou25", "Over", legs=2, **kw)) == [4, 5], "exactly 2 matches"
    r3 = build_acca(short, "ou25", "Over", legs=3, min_legs=3, **kw)    # exactly 3 matches
    assert ids(r3) == [1, 2, 4] and len(r3["legs"]) == 3 and "1.60" in r3["note"], (ids(r3), r3["note"])

    # 3. when the most probable pair already pays enough, it is returned unchanged
    rich = [fx(1, 2, po=80, over_o=1.40), fx(2, 3, po=78, over_o=1.45), fx(3, 4, po=76, over_o=1.50), fx(4, 5, po=60, over_o=2.2)]
    r = build_acca(rich, "ou25", "Over", min_prob=0.5, min_odds=1.1, min_combined=2.0, now=now)
    assert ids(r) == [1, 2] and not r["scaled_down"] and r["note"] == ""
    r = build_acca(rich, "ou25", "Over", legs=3, min_legs=3, min_prob=0.5, min_odds=1.1, min_combined=2.0, now=now)
    assert ids(r) == [1, 2, 3] and not r["scaled_down"]

    # 4. nothing reaches the target within 2-3 matches -> statement with the best possible odds
    low = [fx(i, 2 + i, po=80 - i, over_o=1.20) for i in range(1, 7)]           # 1.2^2 = 1.44, 1.2^3 = 1.73
    r = build_acca(low, "ou25", "Over", min_prob=0.5, min_odds=1.1, min_combined=2.0, now=now)
    assert not r["ok"] and "best possible is 1.73" in r["message"] and "2 to 3 of them" in r["message"], r["message"]
    r = build_acca(low, "ou25", "Over", min_prob=0.5, min_odds=1.1, min_combined=1.7, now=now)
    assert r["ok"] and len(r["legs"]) == 3, "only 3 legs reach 1.7"

    # 5. too few qualifying selections -> statement; two qualifying ones are enough for a 2-match acca
    one = [fx(1, 2), fx(2, 3, over_o=None), fx(3, 4, over_o=None)]
    r = build_acca(one, "ou25", "Over", min_prob=0.5, min_odds=1.1, min_combined=1.5, now=now)
    assert not r["ok"] and r["message"].startswith("No matches fulfilled the acca criteria") and "at least 2 matches" in r["message"]
    assert r["legs"] == []
    two = [fx(1, 2), fx(2, 3), fx(3, 4, over_o=None)]
    r = build_acca(two, "ou25", "Over", min_prob=0.5, min_odds=1.1, min_combined=1.5, now=now)
    assert r["ok"] and ids(r) == [1, 2] and r["combined"]["legs"] == 2
    assert not build_acca(two, "ou25", "Over", legs=3, min_legs=3, min_prob=0.5, min_odds=1.1, min_combined=1.5, now=now)["ok"]

    # 6. filters still apply (started match, probability, real odds) and the leg limits are clamped to 2..3
    ms = [fx(1, 2, po=75), fx(2, 3, po=68), fx(3, 4, po=64), fx(4, 5, po=50, over_o=1.9), fx(5, -1, po=90)]
    r = build_acca(ms, "ou25", "Over", min_prob=0.60, min_odds=1.30, min_combined=1.5, now=now)
    assert ids(r) == [1, 2] and abs(r["combined"]["odds"] - round(1.6 ** 2, 2)) < 0.01, "started match and the 50% fixture excluded"
    r = build_acca(ms, "ou25", "Over", legs=3, min_legs=3, min_prob=0.60, min_odds=1.30, min_combined=1.5, now=now)
    assert ids(r) == [1, 2, 3] and abs(r["combined"]["prob"] - 0.75 * 0.68 * 0.64) < 1e-9
    r = build_acca(ms, "ou25", "Over", min_prob=0.95, now=now)
    assert not r["ok"] and r["funnel"]["fixtures"] == 4 and r["funnel"]["with_odds"] == 4 and r["funnel"]["min_prob"] == 0
    no_odds = [fx(1, 2, over_o=None), fx(2, 3, over_o=None), fx(3, 4, over_o=None)]
    r = build_acca(no_odds, "ou25", "Over", now=now)
    assert not r["ok"] and r["funnel"]["with_odds"] == 0 and "real bookmaker odds" in r["message"]
    p = build_acca(ms, "ou25", "Over", legs=1, min_legs=1, min_prob=0.6, min_odds=1.3, min_combined=1.5, now=now)["params"]
    assert (p["min_legs"], p["max_legs"]) == (2, 2), "1 is raised to the 2-match minimum"
    p = build_acca(ms, "ou25", "Over", legs=9, min_prob=0.6, min_odds=1.3, min_combined=1.5, now=now)["params"]
    assert p["max_legs"] == 3, "9 is lowered to the 3-match maximum"

    # 7. both markets and every side work; anything else is rejected
    ms2 = [fx(i, 2 + i) for i in (1, 2, 3, 4)]
    assert set(MARKETS) == {"ou25", "btts", "1x2"}
    for mk in ("btts", "ou25"):
        assert isinstance(build_acca(ms2, mk, "best", min_prob=0.0, min_odds=1.0, min_combined=1.5, now=now)["ok"], bool)
    assert build_acca(ms2, "btts", "Yes", min_prob=0.0, min_odds=1.0, min_combined=1.5, now=now)["legs"][0]["label"] == "BTTS Yes"
    assert build_acca(ms2, "btts", "No", min_prob=0.0, min_odds=1.0, min_combined=1.5, now=now)["legs"][0]["label"] == "BTTS No"
    # 1X2: three sides, team-named labels, real odds only, the lower default probability, and sides of other markets rejected
    def fx3(i, hours, lh=1.9, la=0.9, oh=1.70, od=3.8, oa=5.0):
        d = fx(i, hours, lh=lh, la=la)
        d.update(odds_home=oh, odds_draw=od, odds_away=oa)
        return d
    x = [fx3(i, 2 + i) for i in (1, 2, 3)]
    r = build_acca(x, "1x2", "Home", min_prob=0.40, min_odds=1.1, min_combined=2.0, now=now)
    assert r["ok"] and all(l["label"] == f"H{l['event_id']} to win" for l in r["legs"]), r["message"]
    assert all(abs(l["prob"] - MR.single(1.9, 0.9)["p_home"]) < 1e-9 for l in r["legs"])
    rd = build_acca(x, "1x2", "Draw", min_prob=0.0, min_odds=1.1, min_combined=2.0, now=now)
    assert rd["ok"] and all(l["label"] == "Draw" and l["odds"] == 3.8 for l in rd["legs"])
    rb = build_acca(x, "1x2", "best", min_prob=0.40, min_odds=1.1, min_combined=2.0, now=now)
    assert rb["ok"] and all(l["side"] == "Home" for l in rb["legs"]), "best side = the most probable outcome"
    assert build_acca(x, "1x2", "Home", now=now)["params"]["min_prob"] == 0.45, "1X2 default minimum probability"
    assert build_acca(x, "btts", "Yes", now=now)["params"]["min_prob"] == DEFAULT_MIN_PROB
    nox = [fx(i, 2 + i) for i in (1, 2, 3)]                           # fixtures without 1X2 odds: never a model-made price
    rn = build_acca(nox, "1x2", "best", min_prob=0.0, min_odds=1.0, min_combined=1.5, now=now)
    assert not rn["ok"] and rn["funnel"]["with_odds"] == 0 and "real bookmaker odds" in rn["message"]
    for bad in (("nope", "best"), ("ou25", "Maybe"), ("btts", "Over"), ("ou25", "Yes"), ("1x2", "Over"), ("1x2", "Yes"), ("btts", "Draw")):
        try:
            build_acca(ms2, *bad, now=now)
            raise AssertionError(f"should have rejected {bad}")
        except ValueError:
            pass
    days = days_available(ms, now)
    assert days and days[0]["count"] == 4
    print("acca_builder self-test passed")


if __name__ == "__main__":
    _selftest()
