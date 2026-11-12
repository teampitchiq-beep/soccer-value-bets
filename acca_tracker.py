# acca_tracker.py
"""
Acca tracking: every acca you choose to track from the Acca builder, kept in data/acca_tracker.json and settled from your finished-match results.

  * One acca = one market (BTTS, Over/Under 2.5 or 1X2), exactly as the builder makes them, so each acca belongs to one of three separate sections.
  * Default stake: 1 unit per acca (ACCA_DEFAULT_STAKE), editable per acca afterwards.
  * Settlement: each leg is graded from the final score (matches.csv). An acca is LOST as soon as any leg has lost, WON when every leg has won,
    and stays PENDING until then. A postponed match can be voided by hand: a voided acca is left out of every statistic.
  * Profit per settled acca: stake x (combined odds - 1) if it won, -stake if it lost. ROI = profit / total stake settled.
  * Streaks are counted in time order (by the last kickoff of each acca). A "streak" is a run of STREAK_MIN or more (default 2) identical results
    in a row; the longest run and the current run are reported too, so single results are never lost.

Settings (config.py):  ACCA_DEFAULT_STAKE = 1.0   ACCA_STREAK_MIN = 2
"""
import json
import os
import threading
from datetime import datetime, timezone

import config
import acca_builder as AB
import botd

DATA_DIR = getattr(config, "DATA_DIR", "data")
STORE = os.path.join(DATA_DIR, "acca_tracker.json")
DEFAULT_STAKE = float(getattr(config, "ACCA_DEFAULT_STAKE", 1.0))
STREAK_MIN = max(2, int(getattr(config, "ACCA_STREAK_MIN", 2)))
SECTIONS = [("btts", "Both Teams To Score", "⚽"), ("ou25", "Over / Under 2.5", "📈"), ("1x2", "Match Result 1X2", "🏆")]
_lock = threading.RLock()


# ---------------------------------------------------------------- storage
def _load():
    if not os.path.exists(STORE):
        return {"next_id": 1, "accas": []}
    try:
        with open(STORE, encoding="utf-8") as f:
            d = json.load(f)
        d.setdefault("next_id", 1 + max([a["id"] for a in d.get("accas", [])] or [0]))
        d.setdefault("accas", [])
        return d
    except Exception:
        # a damaged file must never be overwritten silently: keep a copy and start a fresh one
        try:
            os.replace(STORE, STORE + ".damaged")
        except OSError:
            pass
        return {"next_id": 1, "accas": []}


def _save(d):
    os.makedirs(os.path.dirname(STORE) or ".", exist_ok=True)
    tmp = STORE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    os.replace(tmp, STORE)


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- adding
def _clean(s, n=80):
    return str(s if s is not None else "")[:n]


def add(payload, stake=None):
    """payload: the dict /api/acca returned (an ok build). Returns (ok, message, acca_or_None)."""
    try:
        market = payload["params"]["market"]
        legs_in = payload["legs"]
    except (KeyError, TypeError):
        return False, "Nothing to track: build an acca first.", None
    if market not in AB.MARKETS:
        return False, f"Unknown market '{market}'.", None
    sides = AB.MARKETS[market][1]
    if not isinstance(legs_in, list) or not (AB.MIN_LEGS <= len(legs_in) <= max(AB.MAX_LEGS, AB.MIN_LEGS)):
        return False, f"An acca needs {AB.MIN_LEGS} to {AB.MAX_LEGS} matches.", None
    try:
        stake = DEFAULT_STAKE if stake in (None, "") else float(stake)
    except (TypeError, ValueError):
        return False, "The stake must be a number.", None
    if not (0 < stake <= 100000):
        return False, "The stake must be above 0.", None

    legs, odds, prob, seen = [], 1.0, 1.0, set()
    for l in legs_in:
        try:
            eid, side = int(l["event_id"]), str(l["side"])
            o, p = float(l["odds"]), float(l["prob"])
            ko = botd._utc(l.get("kickoff_iso") or l.get("kickoff_utc")).isoformat()
        except (KeyError, TypeError, ValueError):
            return False, "One of the legs is incomplete, so it was not tracked.", None
        if side not in sides or not (o > 1.0) or not (0 < p <= 1) or eid in seen:
            return False, "One of the legs is not valid, so it was not tracked.", None
        seen.add(eid)
        odds *= o
        prob *= p
        legs.append({"event_id": eid, "home": _clean(l.get("home")), "away": _clean(l.get("away")), "league": _clean(l.get("league")),
                     "kickoff_utc": ko, "label": _clean(l.get("label") or side), "side": side, "prob": round(p, 5), "odds": round(o, 3)})
    legs.sort(key=lambda x: (x["kickoff_utc"], x["event_id"]))
    key = sorted((l["event_id"], l["side"]) for l in legs)
    with _lock:
        d = _load()
        for a in d["accas"]:
            if a["market"] == market and sorted((l["event_id"], l["side"]) for l in a["legs"]) == key:
                return False, "This exact acca is already being tracked.", None
        acca = {"id": d["next_id"], "created_at": _now(), "market": market, "day": _clean(payload.get("day"), 40),
                "legs": legs, "odds": round(odds, 4), "prob": round(prob, 6), "stake": round(stake, 4),
                "status": "pending", "void": False, "leg_results": [None] * len(legs)}
        d["next_id"] += 1
        d["accas"].append(acca)
        _save(d)
    return True, f"Tracking this {len(legs)}-leg acca at {acca['odds']:.2f} with a stake of {stake:g} unit{'s' if stake != 1 else ''}.", acca


def delete(acca_id):
    with _lock:
        d = _load()
        n = len(d["accas"])
        d["accas"] = [a for a in d["accas"] if a["id"] != int(acca_id)]
        if len(d["accas"]) == n:
            return False
        _save(d)
        return True


def set_stake(acca_id, stake):
    try:
        stake = float(stake)
    except (TypeError, ValueError):
        return False, "The stake must be a number."
    if not (0 < stake <= 100000):
        return False, "The stake must be above 0."
    with _lock:
        d = _load()
        for a in d["accas"]:
            if a["id"] == int(acca_id):
                a["stake"] = round(stake, 4)
                _save(d)
                return True, "Stake updated."
    return False, "Acca not found."


def set_void(acca_id, void):
    with _lock:
        d = _load()
        for a in d["accas"]:
            if a["id"] == int(acca_id):
                a["void"] = bool(void)
                _save(d)
                return True
    return False


# ---------------------------------------------------------------- settlement
def settle_one(a, scores):
    """Grade an acca in place from {event_id: (home_goals, away_goals)}. Returns True if anything changed."""
    results = []
    for l in a["legs"]:
        sc = scores.get(l["event_id"])
        results.append(None if sc is None else ("won" if botd.outcome(a["market"], l["side"], sc[0], sc[1]) else "lost"))
    status = "lost" if "lost" in results else ("won" if all(r == "won" for r in results) else "pending")
    changed = results != a.get("leg_results") or status != a.get("status")
    a["leg_results"], a["status"] = results, status
    return changed


def when_key(a):
    """Time order of an acca: the last kickoff among its legs (then its id)."""
    return (max(l["kickoff_utc"] for l in a["legs"]), a["id"])


def effective_status(a):
    return "void" if a.get("void") else a["status"]


def profit(a):
    s = effective_status(a)
    return a["stake"] * (a["odds"] - 1.0) if s == "won" else (-a["stake"] if s == "lost" else 0.0)


# ---------------------------------------------------------------- statistics
def streaks(results):
    """results: list of 'W'/'L' in time order. Runs of identical results, then the counts the dashboard shows."""
    runs = []
    for r in results:
        if runs and runs[-1][0] == r:
            runs[-1][1] += 1
        else:
            runs.append([r, 1])
    def count(kind):
        return sum(1 for k, n in runs if k == kind and n >= STREAK_MIN)
    def longest(kind):
        return max([n for k, n in runs if k == kind] or [0])
    cur = {"type": runs[-1][0], "length": runs[-1][1]} if runs else {"type": None, "length": 0}
    return {"win_streaks": count("W"), "loss_streaks": count("L"), "longest_win": longest("W"), "longest_loss": longest("L"),
            "current": cur, "min_length": STREAK_MIN}


def section_stats(accas):
    """Statistics and chart data for one market's accas (already settled)."""
    live = [a for a in accas if effective_status(a) != "void"]
    settled = sorted([a for a in live if a["status"] in ("won", "lost")], key=when_key)
    pending = [a for a in live if a["status"] == "pending"]
    won = sum(1 for a in settled if a["status"] == "won")
    lost = len(settled) - won
    staked = sum(a["stake"] for a in settled)
    net = sum(profit(a) for a in settled)
    series, cum = [{"n": 0, "cum": 0.0, "id": None, "result": None, "profit": 0.0}], 0.0
    for i, a in enumerate(settled, 1):
        p = profit(a)
        cum += p
        series.append({"n": i, "cum": round(cum, 4), "id": a["id"], "result": "won" if a["status"] == "won" else "lost",
                       "profit": round(p, 4), "odds": a["odds"], "stake": a["stake"], "legs": len(a["legs"]),
                       "when": max(l["kickoff_utc"] for l in a["legs"])})
    peak, max_dd, run_peak = 0.0, 0.0, 0.0
    for pt in series:
        run_peak = max(run_peak, pt["cum"])
        max_dd = max(max_dd, run_peak - pt["cum"])
    return {
        "tracked": len(live), "settled": len(settled), "pending": len(pending), "void": len(accas) - len(live),
        "won": won, "lost": lost,
        "win_rate": round(100.0 * won / len(settled), 1) if settled else None,
        "staked": round(staked, 3), "profit": round(net, 3),
        "roi": round(100.0 * net / staked, 1) if staked else None,
        "avg_odds": round(sum(a["odds"] for a in settled) / len(settled), 2) if settled else None,
        "model_win_rate": round(100.0 * sum(a["prob"] for a in settled) / len(settled), 1) if settled else None,
        "pending_units": round(sum(a["stake"] for a in pending), 3),
        "max_drawdown": round(max_dd, 3),
        "streaks": streaks(["W" if a["status"] == "won" else "L" for a in settled]),
        "series": series,
        "strip": ["W" if a["status"] == "won" else "L" for a in settled][-40:],
    }


def view(scores=None):
    """Settle everything against `scores` (persisting any change) and return all three sections for the dashboard."""
    scores = botd.finished_scores() if scores is None else scores
    with _lock:
        d = _load()
        changed = False
        for a in d["accas"]:
            if settle_one(a, scores):
                changed = True
        if changed:
            _save(d)
        accas = d["accas"]
    sections = []
    for key, title, icon in SECTIONS:
        mine = [a for a in accas if a["market"] == key]
        rows = []
        for a in sorted(mine, key=when_key, reverse=True):
            rows.append(dict(a, status=effective_status(a), profit=round(profit(a), 4), model_status=a["status"]))
        sections.append({"market": key, "title": title, "icon": icon, "stats": section_stats(mine), "accas": rows})
    return {"ok": True, "sections": sections, "default_stake": DEFAULT_STAKE, "streak_min": STREAK_MIN,
            "pending_total": sum(s["stats"]["pending"] for s in sections)}


def pending_count():
    try:
        return sum(1 for a in _load()["accas"] if not a.get("void") and a["status"] == "pending")
    except Exception:
        return 0


# ---------------------------------------------------------------- self-test
def _selftest():
    import tempfile
    global STORE
    old = STORE
    STORE = os.path.join(tempfile.mkdtemp(), "acca_tracker.json")
    try:
        def leg(i, side, odds=1.80, prob=0.6, ko="2026-11-10T17:00:00+03:00"):
            return {"event_id": i, "home": f"H{i}", "away": f"A{i}", "league": "L", "kickoff_iso": ko, "label": f"BTTS {side}",
                    "side": side, "prob": prob, "odds": odds}
        def acca(market, legs):
            return {"ok": True, "day": "Today", "params": {"market": market, "side": "best"}, "legs": legs}

        # --- adding: validation, dedupe, default stake
        ok, msg, a = add(acca("btts", [leg(1, "Yes"), leg(2, "No", 2.0, 0.5)]))
        assert ok and a["stake"] == 1.0 and abs(a["odds"] - 3.6) < 1e-9 and abs(a["prob"] - 0.3) < 1e-9 and a["id"] == 1, msg
        assert not add(acca("btts", [leg(2, "No", 2.0, 0.5), leg(1, "Yes")]))[0], "same acca in another order is a duplicate"
        assert add(acca("btts", [leg(1, "No"), leg(2, "No", 2.0, 0.5)]))[0], "a different side is a different acca"
        for bad in (acca("nope", [leg(1, "Yes"), leg(2, "Yes")]), acca("btts", [leg(1, "Over"), leg(2, "Yes")]),
                    acca("btts", [leg(1, "Yes")]), acca("btts", [leg(1, "Yes"), leg(1, "Yes")]),
                    acca("btts", [leg(1, "Yes", odds=1.0), leg(2, "Yes")]), {"ok": True}, None):
            assert not add(bad)[0], bad
        assert not add(acca("btts", [leg(7, "Yes"), leg(8, "Yes")]), stake=0)[0] and not add(acca("btts", [leg(7, "Yes"), leg(8, "Yes")]), stake="x")[0]
        ok, _, a2 = add(acca("btts", [leg(7, "Yes"), leg(8, "Yes")]), stake=2.5)
        assert ok and a2["stake"] == 2.5

        # --- settlement: pending, early loss, win
        by = lambda i: next(a for a in _load()["accas"] if a["id"] == i)
        d = _load()
        a1 = d["accas"][0]
        assert settle_one(a1, {}) is False and a1["status"] == "pending"
        settle_one(a1, {1: (1, 1)})                                        # leg 1 BTTS Yes won, leg 2 unknown
        assert a1["status"] == "pending" and a1["leg_results"] == ["won", None]
        settle_one(a1, {1: (1, 1), 2: (2, 1)})                             # leg 2 BTTS No lost -> acca lost
        assert a1["status"] == "lost" and effective_status(a1) == "lost" and profit(a1) == -1.0
        a1["leg_results"] = [None, None]; a1["status"] = "pending"
        settle_one(a1, {2: (2, 1)})                                        # one lost leg is enough, even with the other unknown
        assert a1["status"] == "lost"
        a1["leg_results"] = [None, None]; a1["status"] = "pending"
        settle_one(a1, {1: (2, 2), 2: (1, 0)})
        assert a1["status"] == "won" and abs(profit(a1) - 2.6) < 1e-9
        a1["void"] = True
        assert effective_status(a1) == "void" and profit(a1) == 0.0

        # --- streaks (time order), counted from STREAK_MIN
        s = streaks(list("WWWLWLLWWL"))
        assert (s["win_streaks"], s["loss_streaks"], s["longest_win"], s["longest_loss"]) == (2, 1, 3, 2) and s["current"] == {"type": "L", "length": 1}
        s = streaks(list("WLWLWL"))
        assert (s["win_streaks"], s["loss_streaks"], s["longest_win"]) == (0, 0, 1), "alternating results make no streaks"
        assert streaks([])["current"] == {"type": None, "length": 0}

        # --- section statistics, by hand: stake 1 unless stated; odds 2.0 / 3.0 / 1.5
        def mk(i, status, odds, stake=1.0, ko=None):
            return {"id": i, "market": "btts", "legs": [{"kickoff_utc": ko or f"2026-11-{i:02d}T18:00:00+00:00", "event_id": i}], "odds": odds,
                    "stake": stake, "status": status, "void": False, "prob": 0.4, "leg_results": []}
        accs = [mk(1, "won", 2.0), mk(2, "lost", 3.0), mk(3, "won", 3.0, stake=2.0), mk(4, "lost", 1.5), mk(5, "pending", 2.0), mk(6, "lost", 2.0)]
        accs[5]["void"] = True
        st = section_stats(accs)
        # settled: W(2.0, 1u) +1 ; L -1 ; W(3.0, 2u) +4 ; L -1  -> profit +3 on 5 units staked
        assert (st["settled"], st["won"], st["lost"], st["pending"], st["void"], st["tracked"]) == (4, 2, 2, 1, 1, 5)
        assert abs(st["profit"] - 3.0) < 1e-9 and abs(st["staked"] - 5.0) < 1e-9 and st["roi"] == 60.0 and st["win_rate"] == 50.0
        assert [p["cum"] for p in st["series"]] == [0.0, 1.0, 0.0, 4.0, 3.0] and st["max_drawdown"] == 1.0
        assert st["streaks"]["win_streaks"] == 0 and st["strip"] == ["W", "L", "W", "L"] and st["model_win_rate"] == 40.0
        assert st["pending_units"] == 1.0 and st["avg_odds"] == 2.38
        empty = section_stats([])
        assert empty["roi"] is None and empty["win_rate"] is None and empty["series"] == [{"n": 0, "cum": 0.0, "id": None, "result": None, "profit": 0.0}]

        # --- edit / delete / view persistence
        assert set_stake(2, 3)[0] and by(2)["stake"] == 3.0 and not set_stake(2, -1)[0] and not set_stake(99, 1)[0]
        assert set_void(2, True) and by(2)["void"] and delete(2) and not delete(2)
        v = view(scores={7: (1, 1), 8: (0, 2)})                             # acca 3: BTTS Yes @7 won, BTTS Yes @8 lost -> lost
        sec = next(s for s in v["sections"] if s["market"] == "btts")
        assert next(r for r in sec["accas"] if r["id"] == 3)["status"] == "lost"
        assert [s["market"] for s in v["sections"]] == ["btts", "ou25", "1x2"]
        print("acca_tracker self-test passed")
    finally:
        STORE = old


if __name__ == "__main__":
    _selftest()
