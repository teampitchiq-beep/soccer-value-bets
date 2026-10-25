# ht_acca.py
"""
Half-time acca: same rules as the goals and HT accas, built from the first-half goals
predictions, UNDER 1.5 first-half goals legs only.

  * ONE acca per day, today's matches only (your local date).
  * 2-3 legs, combined odds 2.0-3.0, each leg >= 58% model probability.
  * Selection: candidates ranked by probability (highest first); try 2-leg combos, then
    3 legs, in rank order, scaling down to the next-highest only when the odds fall short.
  * A new acca is generated only after the previous one is GRADED (won / lost / void).
    Until then the same acca stays frozen. "Skip" voids it; 48h after the last kickoff
    an ungraded acca is voided automatically.

IMPORTANT - odds: the Bzzoiro API has no half-time odds on the free plan. Leg odds are
MODEL-IMPLIED (1 / probability), so the combined 2.0-3.0 target means "joint probability of
about 33-50%". Enter your bookmaker's real combined odds when logging if you want profit tracked.

Optional settings (config.py; defaults mirror the goals acca):
    HACCA_MIN_ODDS = 2.0   HACCA_MAX_ODDS = 3.0   HACCA_MAX_LEGS = 3   HACCA_MIN_LEGS = 2
    HACCA_MIN_LEG_PROB = 0.58   HACCA_MIN_LEG_ODDS = 1.0   HACCA_POOL = 12
    HACCA_UNDER_ONLY = True     HACCA_MIN_COVERAGE = 0.70
"""
import itertools
import json
import os
from datetime import datetime, timezone

import pandas as pd

import config
from acca_shared import exclude_used
from acca import _now_utc, _tz, local_today

MIN_ODDS     = getattr(config, "HACCA_MIN_ODDS", getattr(config, "ACCA_MIN_ODDS", 2.0))
MAX_ODDS     = getattr(config, "HACCA_MAX_ODDS", getattr(config, "ACCA_MAX_ODDS", 3.0))
MAX_LEGS     = getattr(config, "HACCA_MAX_LEGS", getattr(config, "ACCA_MAX_LEGS", 3))
MIN_LEGS     = getattr(config, "HACCA_MIN_LEGS", getattr(config, "ACCA_MIN_LEGS", 2))
MIN_LEG_PROB = getattr(config, "HACCA_MIN_LEG_PROB", getattr(config, "ACCA_MIN_LEG_PROB", 0.58))
MIN_LEG_ODDS = getattr(config, "HACCA_MIN_LEG_ODDS", 1.0)
POOL         = getattr(config, "HACCA_POOL", getattr(config, "ACCA_POOL", 12))
ONE_PER_DAY  = getattr(config, "HACCA_ONE_PER_DAY", getattr(config, "ACCA_ONE_PER_DAY", True))
UNDER_ONLY   = getattr(config, "HACCA_UNDER_ONLY", True)

EXPIRE_HOURS, CHECK_EVERY_S, FINISH_AFTER_MIN = 48, 300, 100
ACCA_CSV, STATE_JSON = "data/hacca_log.csv", "data/hacca_current.json"
COLUMNS = ["acca_id", "date", "logged_at", "n_legs", "legs", "odds", "model_odds", "odds_source",
           "model_prob", "stake", "result", "pnl", "notes"]


# ---------------------------------------------------------------- selection
def rules_text():
    cap = f"–{MAX_ODDS:g}" if MAX_ODDS else "+"
    side = "Under 1.5 HT only · " if UNDER_ONLY else ""
    return (f"{side}today's matches only · {MIN_LEGS}–{MAX_LEGS} legs · model-implied combined odds "
            f"{MIN_ODDS:g}{cap} · each leg ≥ {MIN_LEG_PROB*100:.0f}% model probability")


MIN_COVERAGE = getattr(config, "HACCA_MIN_COVERAGE", 0.70)
_COV = {"key": None, "ok": None}


def _gradable_leagues():
    """League ids where >= MIN_COVERAGE of finished matches have a half-time score.
    Legs from patchier leagues often can't be graded, which would freeze the acca for 48h.
    Returns None (= no filtering) if the data can't be read."""
    try:
        import os
        mp, cp = config.MATCHES_XG_CSV, "data/ht_goals.csv"
        key = (os.path.getmtime(mp), os.path.getmtime(cp))
        if _COV["key"] != key:
            m = pd.read_csv(mp, usecols=["event_id", "league_id"])
            c = pd.read_csv(cp, usecols=["event_id", "home_ht"])
            x = m.merge(c, on="event_id", how="left")
            cov = x.groupby("league_id")["home_ht"].apply(lambda s: s.notna().mean())
            _COV["key"], _COV["ok"] = key, set(int(l) for l, v in cov.items() if v >= MIN_COVERAGE)
        return _COV["ok"]
    except Exception:
        return None


def _candidates(rows, now):
    tz, out = _tz(), []
    ok_leagues = _gradable_leagues()
    for r in rows:
        if ok_leagues is not None and int(r.get("league_id", -1)) not in ok_leagues:
            continue
        if UNDER_ONLY and r["is_over"]:
            continue
        prob = r["prob"]
        if prob < MIN_LEG_PROB:
            continue
        odds = round(1.0 / prob, 3)
        if odds < MIN_LEG_ODDS:
            continue
        ko = pd.to_datetime(r["date"], utc=True, errors="coerce")
        if pd.isna(ko) or ko <= now:
            continue
        out.append({
            "event_id": int(r["event_id"]), "fixture": f"{r['home']} vs {r['away']}",
            "league": r["league_name"], "kickoff": ko.strftime("%Y-%m-%d %H:%M"),
            "date": ko.tz_convert(tz).strftime("%Y-%m-%d"),
            "pick": r["pick"], "line": r["line"], "exp": r["mu_t"],
            "odds": odds, "prob": round(float(prob), 4),
        })
    out.sort(key=lambda c: (-c["prob"], c["event_id"]))
    return out


def _pick_combo(pool):
    for n in range(MIN_LEGS, MAX_LEGS + 1):
        for combo in itertools.combinations(range(len(pool)), n):
            odds = 1.0
            for i in combo:
                odds *= pool[i]["odds"]
            if odds >= MIN_ODDS and (MAX_ODDS is None or odds <= MAX_ODDS):
                return [pool[i] for i in combo], odds
    return None, None


def build_haccas(rows, only_date=None):
    by_date = {}
    for c in exclude_used(_candidates(rows, _now_utc()), STATE_JSON):
        if only_date and c["date"] != only_date:
            continue
        by_date.setdefault(c["date"], []).append(c)
    out = []
    for date in sorted(by_date):
        legs, odds = _pick_combo(by_date[date][:POOL])
        if not legs:
            continue
        prob = 1.0
        for l in legs:
            prob *= l["prob"]
        out.append({
            "id": f"H{date}-" + "-".join(str(l["event_id"]) for l in sorted(legs, key=lambda x: x["event_id"])),
            "date": date, "legs": [dict(l, result="", score="") for l in legs], "n_legs": len(legs),
            "odds": round(odds, 2), "prob": round(prob, 4), "candidates": len(by_date[date]),
        })
    return out


# ---------------------------------------------------------------- grading
def _leg_ht(event_id):
    """(home, away) half-time goals for a finished match, or None. Local file first, then the API."""
    from ht_data import fetch_event_ht, load_ht
    saved = load_ht()
    row = saved[saved["event_id"] == event_id]
    if len(row) and pd.notna(row.iloc[0]["home_ht"]):
        return int(row.iloc[0]["home_ht"]), int(row.iloc[0]["away_ht"])
    state, val, status = fetch_event_ht(event_id)
    return val if state == "ok" and status == "finished" else None


def _resolve_legs(legs):
    changed = False
    for leg in legs:
        if leg.get("result"):
            continue
        got = _leg_ht(leg["event_id"])
        if not got:
            continue
        h, a = got
        total = h + a
        over = leg["pick"].startswith("Over")
        won = total > leg["line"] if over else total < leg["line"]
        leg["result"], leg["score"] = ("win" if won else "loss"), f"{h}-{a} HT"
        changed = True
    return changed


def _grade(legs):
    if any(l.get("result") == "loss" for l in legs):
        return "loss"
    if legs and all(l.get("result") == "win" for l in legs):
        return "win"
    return ""


# ---------------------------------------------------------------- current acca (frozen state)
def _load_state():
    if not os.path.exists(STATE_JSON):
        return None
    try:
        with open(STATE_JSON, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_state(s):
    os.makedirs("data", exist_ok=True)
    with open(STATE_JSON, "w", encoding="utf-8") as f:
        json.dump(s, f)


def _kick(acca, fn):
    return fn(pd.Timestamp(l["kickoff"], tz="UTC") for l in acca["legs"])


def grade_current():
    s = _load_state()
    if not s or s.get("graded"):
        return s
    try:
        _resolve_legs(s["acca"]["legs"])
    except Exception:
        pass
    s["last_check"] = _now_utc().isoformat()
    result = _grade(s["acca"]["legs"])
    if not result and _now_utc() > _kick(s["acca"], max) + pd.Timedelta(hours=EXPIRE_HOURS):
        result, s["note"] = "void", "expired: no result data 48h after the last kickoff"
    if result:
        s["graded"], s["result"], s["graded_at"] = True, result, _now_utc().isoformat()
    _save_state(s)
    return s


def skip_current():
    s = _load_state()
    if not s or s.get("graded"):
        return False
    s["graded"], s["result"], s["note"] = True, "void", "skipped manually"
    s["graded_at"] = _now_utc().isoformat()
    _save_state(s)
    void_acca(s["acca"]["id"])
    return True


def get_current():
    s = _load_state()
    return s["acca"] if s and not s.get("graded") else None


def current_hacca(rows):
    """{"acca","status": pending|new|none,"prev","logged","reason"} - see acca.current_acca."""
    s = _load_state()
    if s and not s.get("graded"):
        due = _now_utc() >= _kick(s["acca"], min) + pd.Timedelta(minutes=FINISH_AFTER_MIN)
        last = pd.Timestamp(s["last_check"]) if s.get("last_check") else None
        if due and (last is None or (_now_utc() - last).total_seconds() >= CHECK_EVERY_S):
            s = grade_current()
    if s and not s.get("graded"):
        return {"acca": s["acca"], "status": "pending", "prev": None,
                "logged": is_logged(s["acca"]["id"]), "reason": ""}

    today = local_today()
    # One acca per day: once today's acca has been WON or LOST, no replacement until tomorrow.
    # (Skipping / voiding an acca still frees the slot.)  Set ONE_PER_DAY = False in config to disable.
    if ONE_PER_DAY and s and s.get("graded") and s.get("result") in ("win", "loss") \
            and (s.get("acca") or {}).get("date") == today:
        return {"acca": None, "status": "none", "prev": s, "logged": False,
                "reason": f"Today's acca is already settled ({s['result']}). The next one is generated tomorrow."}
    built = build_haccas(rows, only_date=today)
    if built:
        _save_state({"acca": built[0], "created_at": _now_utc().isoformat(),
                     "graded": False, "result": "", "last_check": None})
        return {"acca": built[0], "status": "new", "prev": s, "logged": False, "reason": ""}
    return {"acca": None, "status": "none", "prev": s, "logged": False,
            "reason": f"No HT acca fits today's rules ({today}) right now."}


# ---------------------------------------------------------------- tracking
def load_haccas():
    if not os.path.exists(ACCA_CSV):
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(ACCA_CSV, dtype=str).fillna("")
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df[COLUMNS]


def _save(df):
    os.makedirs("data", exist_ok=True)
    df[COLUMNS].to_csv(ACCA_CSV, index=False)


def is_logged(acca_id):
    return acca_id in load_haccas()["acca_id"].values


def add_acca(acca, stake=1.0, bookmaker_odds=None):
    df = load_haccas()
    if acca["id"] in df["acca_id"].values:
        return False, "already logged"
    real = None
    try:
        real = float(bookmaker_odds) if bookmaker_odds not in (None, "") else None
    except ValueError:
        real = None
    if real is not None and real <= 1.0:
        real = None
    row = {
        "acca_id": acca["id"], "date": acca["date"],
        "logged_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "n_legs": str(acca["n_legs"]), "legs": json.dumps([dict(l, result="", score="") for l in acca["legs"]]),
        "odds": str(real if real else acca["odds"]), "model_odds": str(acca["odds"]),
        "odds_source": "bookmaker" if real else "model", "model_prob": str(acca["prob"]),
        "stake": str(float(stake)), "result": "", "pnl": "", "notes": "",
    }
    _save(pd.concat([df, pd.DataFrame([row])], ignore_index=True))
    return True, "logged"


def void_acca(acca_id):
    s = _load_state()
    if s and not s.get("graded") and s["acca"]["id"] == acca_id:
        s["graded"], s["result"], s["note"] = True, "void", "voided manually"
        s["graded_at"] = _now_utc().isoformat()
        _save_state(s)
    df = load_haccas()
    m = (df["acca_id"] == acca_id) & (df["result"] == "")
    if not m.any():
        return False
    df.loc[m, ["result", "pnl", "notes"]] = ["void", "0.0", "voided manually"]
    _save(df)
    return True


def settle_accas(verbose=False):
    df = load_haccas()
    n, changed = 0, False
    for idx, row in df[df["result"] == ""].iterrows():
        legs = json.loads(row["legs"])
        if _resolve_legs(legs):
            changed = True
        df.at[idx, "legs"] = json.dumps(legs)
        grade = _grade(legs)
        if not grade:
            continue
        stake, odds, real = float(row["stake"]), float(row["odds"]), row["odds_source"] == "bookmaker"
        df.at[idx, "result"] = grade
        if grade == "loss":
            lost = next(l for l in legs if l.get("result") == "loss")
            df.at[idx, "pnl"] = str(round(-stake, 2)) if real else ""
            df.at[idx, "notes"] = f"lost on {lost['fixture']} ({lost['score']})"
        else:
            df.at[idx, "pnl"] = str(round(stake * (odds - 1), 2)) if real else ""
        n += 1
        if verbose:
            print(f"  HT acca {row['acca_id']}: {grade}")
    if changed or n:
        _save(df)
    return n, []


def summary():
    df = load_haccas()
    done = df[df["result"].isin(["win", "loss"])]
    s = {"total": len(df), "pending": int((df["result"] == "").sum()), "settled": len(done),
         "wins": 0, "losses": 0, "hit_rate": 0.0, "expected_hit": 0.0, "avg_odds": 0.0,
         "bk_n": 0, "bk_pnl": 0.0, "bk_roi": 0.0}
    if len(done):
        s["wins"], s["losses"] = int((done["result"] == "win").sum()), int((done["result"] == "loss").sum())
        s["hit_rate"] = round(s["wins"] / len(done) * 100, 1)
        s["expected_hit"] = round(float(done["model_prob"].astype(float).mean()) * 100, 1)
        s["avg_odds"] = round(float(done["model_odds"].astype(float).mean()), 2)
        bk = done[done["odds_source"] == "bookmaker"]
        if len(bk):
            stake = float(bk["stake"].astype(float).sum())
            s["bk_n"], s["bk_pnl"] = len(bk), round(float(bk["pnl"].astype(float).sum()), 2)
            s["bk_roi"] = round(s["bk_pnl"] / stake * 100, 1) if stake else 0.0
    return s


def rows_for_display():
    df = load_haccas()
    out = []
    for r in df.to_dict("records"):
        r["legs"] = json.loads(r["legs"]) if r["legs"] else []
        out.append(r)
    pend = sorted([r for r in out if r["result"] == ""], key=lambda r: r["date"])
    done = sorted([r for r in out if r["result"] != ""], key=lambda r: r["date"], reverse=True)
    return pend, done
