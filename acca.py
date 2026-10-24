# acca.py
"""
2-odds accumulator: ONE acca per present day, tracked and graded.

Rules
  * Only TODAY's matches are used (never future dates). "Today" is your local date.
  * Over 2.5 legs only (set ACCA_OVER_ONLY = False to allow Under).
  * A new acca is generated only when the previous one has been GRADED
    (won / lost / void). Until then the same acca stays on screen, frozen.
  * Selection: candidates ranked by model probability (highest first). Try
    2-leg combinations, then 3 legs, in rank order: (1st+2nd), (1st+3rd) ...
    i.e. highest-probability first, scaling down to the next-highest only when
    the combined odds fall short. First combo with odds in [MIN, MAX] wins.
  * Safety valve: an acca still ungraded 48h after its last kickoff is voided
    automatically; you can also press "Skip" on the card.

Optional settings (add to config.py to override; defaults shown):
    ACCA_MIN_ODDS = 2.0   ACCA_MAX_ODDS = 3.0   (None = no cap)
    ACCA_MAX_LEGS = 3     ACCA_MIN_LEGS = 2
    ACCA_MIN_LEG_PROB = 0.58   ACCA_MIN_LEG_ODDS = 1.15   ACCA_POOL = 12
    ACCA_OVER_ONLY = True
    ACCA_UTC_OFFSET_HOURS = None   # your timezone, e.g. 3 for UTC+3 (None = auto-detect)

CLI:  python acca.py            show what today's acca would be (does not save)
      python acca.py --grade    grade the current acca      python acca.py --settle
      python acca.py --report   tracker summary
"""
import itertools
import json
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

import config

MIN_ODDS     = getattr(config, "ACCA_MIN_ODDS", 2.0)
MAX_ODDS     = getattr(config, "ACCA_MAX_ODDS", 3.0)
MAX_LEGS     = getattr(config, "ACCA_MAX_LEGS", 3)
MIN_LEGS     = getattr(config, "ACCA_MIN_LEGS", 2)
OVER_ONLY    = getattr(config, "ACCA_OVER_ONLY", True)
MIN_LEG_PROB = getattr(config, "ACCA_MIN_LEG_PROB", 0.58)
MIN_LEG_ODDS = getattr(config, "ACCA_MIN_LEG_ODDS", 1.15)
POOL         = getattr(config, "ACCA_POOL", 12)
UTC_OFFSET   = getattr(config, "ACCA_UTC_OFFSET_HOURS", None)

EXPIRE_HOURS  = 48        # void an ungraded acca this long after its last kickoff
CHECK_EVERY_S = 300       # page loads re-check results at most every 5 minutes
FINISH_AFTER_MIN = 100    # don't look for results before kickoff + this

ACCA_CSV = "data/acca_log.csv"
STATE_JSON = "data/acca_current.json"
COLUMNS = ["acca_id", "date", "logged_at", "n_legs", "legs", "odds",
           "model_prob", "stake", "result", "pnl", "notes"]


# ---------------------------------------------------------------- time helpers
def _tz():
    off = UTC_OFFSET
    if off is None:
        off = (time.localtime().tm_gmtoff or 0) / 3600.0
    return timezone(timedelta(hours=off))


def local_today():
    return datetime.now(_tz()).strftime("%Y-%m-%d")


def _now_utc():
    return pd.Timestamp.now(tz="UTC")


# ---------------------------------------------------------------- selection
def rules_text():
    cap = f"–{MAX_ODDS:g}" if MAX_ODDS else "+"
    side = "Over 2.5 only · " if OVER_ONLY else ""
    return (f"{side}today's matches only · {MIN_LEGS}–{MAX_LEGS} legs · "
            f"combined odds {MIN_ODDS:g}{cap} · each leg ≥ {MIN_LEG_PROB*100:.0f}% model probability")


def _candidates(matches, now):
    tz = _tz()
    out = []
    for m in matches:
        pick, prob, odds = m.get("model_pick"), m.get("model_prob"), m.get("book_odds")
        if not pick or prob is None or not odds:
            continue
        if OVER_ONLY and pick != "Over 2.5":
            continue
        if prob < MIN_LEG_PROB or odds < MIN_LEG_ODDS:
            continue
        ko = pd.to_datetime(m.get("date"), utc=True, errors="coerce")
        if pd.isna(ko) or ko <= now:
            continue
        out.append({
            "event_id": int(m["event_id"]),
            "fixture": f"{m['home']} vs {m['away']}",
            "league": m.get("league_name", ""),
            "kickoff": ko.strftime("%Y-%m-%d %H:%M"),               # UTC, as shown in the app
            "date": ko.tz_convert(tz).strftime("%Y-%m-%d"),         # your local day
            "pick": pick,
            "odds": round(float(odds), 3),
            "prob": round(float(prob), 4),
        })
    out.sort(key=lambda c: (-c["prob"], -c["odds"]))
    return out


def _pick_combo(pool):
    """Rank-ordered search: fewest legs first, highest-probability legs first."""
    for n in range(MIN_LEGS, MAX_LEGS + 1):
        for combo in itertools.combinations(range(len(pool)), n):
            odds = 1.0
            for i in combo:
                odds *= pool[i]["odds"]
            if odds >= MIN_ODDS and (MAX_ODDS is None or odds <= MAX_ODDS):
                return [pool[i] for i in combo], odds
    return None, None


def build_accas(matches, only_date=None):
    """One acca per local date where one can be built; only_date restricts to a single day."""
    by_date = {}
    for c in _candidates(matches, _now_utc()):
        if only_date and c["date"] != only_date:
            continue
        by_date.setdefault(c["date"], []).append(c)

    accas = []
    for date in sorted(by_date):
        pool = by_date[date][:POOL]
        legs, odds = _pick_combo(pool)
        if not legs:
            continue
        prob = 1.0
        for l in legs:
            prob *= l["prob"]
        accas.append({
            "id": f"{date}-" + "-".join(str(l["event_id"]) for l in sorted(legs, key=lambda x: x["event_id"])),
            "date": date,
            "legs": [dict(l, result="", score="") for l in legs],
            "n_legs": len(legs),
            "odds": round(odds, 2),
            "prob": round(prob, 4),                 # assumes legs are independent
            "ev": round(prob * odds - 1, 3),
            "candidates": len(by_date[date]),
        })
    return accas


# ---------------------------------------------------------------- grading helpers
def _resolve_legs(legs):
    """Fill in leg results from Bzzoiro. Returns True if any leg changed. Needs internet."""
    from tracker import fetch_event_result
    changed = False
    for leg in legs:
        if leg.get("result"):
            continue
        res = fetch_event_result(leg["event_id"])
        if not res:
            continue
        hg, ag, status = res
        if status != "finished" or hg is None or ag is None:
            continue
        total = hg + ag
        won = total > 2.5 if leg["pick"] == "Over 2.5" else total < 2.5
        leg["result"] = "win" if won else "loss"
        leg["score"] = f"{hg}-{ag}"
        changed = True
    return changed


def _grade(legs):
    """'' (not graded yet) | 'win' | 'loss'. One lost leg sinks the acca."""
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


def _save_state(state):
    os.makedirs("data", exist_ok=True)
    with open(STATE_JSON, "w", encoding="utf-8") as f:
        json.dump(state, f)


def _last_kickoff(acca):
    return max(pd.Timestamp(l["kickoff"], tz="UTC") for l in acca["legs"])


def _first_kickoff(acca):
    return min(pd.Timestamp(l["kickoff"], tz="UTC") for l in acca["legs"])


def grade_current():
    """Try to grade the current acca now (best effort; safe offline). Returns the state."""
    s = _load_state()
    if not s or s.get("graded"):
        return s
    try:
        _resolve_legs(s["acca"]["legs"])
    except Exception:
        pass
    s["last_check"] = _now_utc().isoformat()
    result = _grade(s["acca"]["legs"])
    if not result and _now_utc() > _last_kickoff(s["acca"]) + pd.Timedelta(hours=EXPIRE_HOURS):
        result, s["note"] = "void", "expired: no result data 48h after the last kickoff"
    if result:
        s["graded"], s["result"], s["graded_at"] = True, result, _now_utc().isoformat()
    _save_state(s)
    return s


def skip_current():
    """Void the current acca by hand (e.g. postponed match). Frees the slot for a new one."""
    s = _load_state()
    if not s or s.get("graded"):
        return False
    s["graded"], s["result"], s["note"] = True, "void", "skipped manually"
    s["graded_at"] = _now_utc().isoformat()
    _save_state(s)
    void_acca(s["acca"]["id"])          # also void it in the tracker if it was logged
    return True


def get_current():
    """The frozen current acca dict if it is still ungraded, else None."""
    s = _load_state()
    return s["acca"] if s and not s.get("graded") else None


def current_acca(matches):
    """
    Returns {"acca": dict|None, "status": "pending"|"new"|"none", "prev": state|None,
             "logged": bool, "reason": str}
    - If the previous acca is not graded, it is returned unchanged (nothing new is generated).
    - Once graded, a new acca is generated for today if one fits the rules.
    """
    s = _load_state()

    if s and not s.get("graded"):
        due = _now_utc() >= _first_kickoff(s["acca"]) + pd.Timedelta(minutes=FINISH_AFTER_MIN)
        last = pd.Timestamp(s["last_check"]) if s.get("last_check") else None
        stale = last is None or (_now_utc() - last).total_seconds() >= CHECK_EVERY_S
        if due and stale:
            s = grade_current()

    if s and not s.get("graded"):
        return {"acca": s["acca"], "status": "pending", "prev": None,
                "logged": is_logged(s["acca"]["id"]), "reason": ""}

    today = local_today()
    built = build_accas(matches, only_date=today)
    if built:
        new = built[0]
        _save_state({"acca": new, "created_at": _now_utc().isoformat(), "graded": False,
                     "result": "", "last_check": None})
        return {"acca": new, "status": "new", "prev": s, "logged": False, "reason": ""}

    return {"acca": None, "status": "none", "prev": s, "logged": False,
            "reason": f"No acca fits today's rules ({today}) right now."}


# ---------------------------------------------------------------- tracking (log)
def load_accas():
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
    return acca_id in load_accas()["acca_id"].values


def add_acca(acca, stake=1.0):
    df = load_accas()
    if acca["id"] in df["acca_id"].values:
        return False, "This acca is already logged."
    legs = [dict(l, result="", score="") for l in acca["legs"]]
    row = {
        "acca_id": acca["id"], "date": acca["date"],
        "logged_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "n_legs": str(acca["n_legs"]), "legs": json.dumps(legs),
        "odds": str(acca["odds"]), "model_prob": str(acca["prob"]),
        "stake": str(float(stake)), "result": "", "pnl": "", "notes": "",
    }
    _save(pd.concat([df, pd.DataFrame([row])], ignore_index=True))
    return True, "Acca logged."


def void_acca(acca_id):
    """Void a logged acca (stake returned). Also frees the current-acca slot if it is the same acca."""
    s = _load_state()
    if s and not s.get("graded") and s["acca"]["id"] == acca_id:
        s["graded"], s["result"], s["note"] = True, "void", "voided manually"
        s["graded_at"] = _now_utc().isoformat()
        _save_state(s)
    df = load_accas()
    m = (df["acca_id"] == acca_id) & (df["result"] == "")
    if not m.any():
        return False
    df.loc[m, "result"] = "void"
    df.loc[m, "pnl"] = "0.0"
    df.loc[m, "notes"] = "voided manually"
    _save(df)
    return True


def settle_accas(verbose=False):
    """Check pending LOGGED accas against Bzzoiro results. Returns (n_settled, details)."""
    df = load_accas()
    pending = df[df["result"] == ""]
    n, details, changed = 0, [], False

    for idx, row in pending.iterrows():
        legs = json.loads(row["legs"])
        if _resolve_legs(legs):
            changed = True
        df.at[idx, "legs"] = json.dumps(legs)

        stake, odds = float(row["stake"]), float(row["odds"])
        grade = _grade(legs)
        if grade == "loss":
            lost = next(l for l in legs if l.get("result") == "loss")
            df.at[idx, "result"], df.at[idx, "pnl"] = "loss", str(round(-stake, 2))
            df.at[idx, "notes"] = f"lost on {lost['fixture']} ({lost['score']})"
        elif grade == "win":
            df.at[idx, "result"] = "win"
            df.at[idx, "pnl"] = str(round(stake * (odds - 1), 2))
        else:
            continue
        n += 1
        details.append({"acca_id": row["acca_id"], "result": df.at[idx, "result"],
                        "pnl": float(df.at[idx, "pnl"])})

    if changed or n:
        _save(df)
    if verbose and n:
        for d in details:
            print(f"  acca {d['acca_id']}: {d['result']} ({d['pnl']:+.2f}u)")
    return n, details


def summary():
    df = load_accas()
    done = df[df["result"].isin(["win", "loss"])].copy()
    s = {"total": len(df), "pending": int((df["result"] == "").sum()),
         "settled": len(done), "wins": 0, "losses": 0, "staked": 0.0, "pnl": 0.0,
         "roi": 0.0, "hit_rate": 0.0, "expected_hit": 0.0, "avg_odds": 0.0}
    if len(done):
        stake = done["stake"].astype(float)
        s.update(wins=int((done["result"] == "win").sum()),
                 losses=int((done["result"] == "loss").sum()),
                 staked=round(float(stake.sum()), 2), pnl=round(float(done["pnl"].astype(float).sum()), 2),
                 avg_odds=round(float(done["odds"].astype(float).mean()), 2))
        s["roi"] = round(s["pnl"] / s["staked"] * 100, 1) if s["staked"] else 0.0
        s["hit_rate"] = round(s["wins"] / len(done) * 100, 1)
        s["expected_hit"] = round(float(done["model_prob"].astype(float).mean()) * 100, 1)
    return s


def rows_for_display():
    """(pending, settled) lists of dicts with legs parsed."""
    df = load_accas()
    out = []
    for r in df.to_dict("records"):
        r["legs"] = json.loads(r["legs"]) if r["legs"] else []
        out.append(r)
    pending = sorted([r for r in out if r["result"] == ""], key=lambda r: r["date"])
    settled = sorted([r for r in out if r["result"] != ""], key=lambda r: r["date"], reverse=True)
    return pending, settled


if __name__ == "__main__":
    import sys
    if "--settle" in sys.argv:
        print(settle_accas(verbose=True)[0], "acca(s) settled")
    elif "--grade" in sys.argv:
        s = grade_current()
        print("no current acca" if not s else f"graded={s.get('graded')} result={s.get('result')}")
    elif "--report" in sys.argv:
        for k, v in summary().items():
            print(f"{k:>13}: {v}")
    else:
        from scan import get_todays_matches
        built = build_accas(get_todays_matches(days=2)["matches"], only_date=local_today())
        print(f"Rules: {rules_text()}\n(preview only - nothing saved)\n")
        if not built:
            print("No valid acca for today right now.")
        for a in built:
            print(f"{a['date']}  {a['n_legs']} legs @ {a['odds']}  "
                  f"(model {a['prob']*100:.1f}%, EV {a['ev']*100:+.1f}%)")
            for l in a["legs"]:
                print(f"   {l['kickoff'][11:]}  {l['fixture']:<40} {l['pick']:<10} "
                      f"@{l['odds']}  ({l['prob']*100:.0f}%)")
