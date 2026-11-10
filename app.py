# app.py
"""
Single-entry daily runner with a once-a-day recent-matches fetcher.

The model covers two markets only: Over/Under 2.5 goals and Both Teams To Score (Yes/No).
Tabs: All Matches (Over/Under 2.5), BTTS, 1X2, Trends, Shots, Acca, Results, Date and Single Match.
"""
import json
import os
import sys
import threading
import time
import pandas as pd
from datetime import datetime, timezone, timedelta

from flask import Flask, render_template, request, jsonify, redirect, url_for

from config import (
    DEFAULT_LEAGUE_ID, LEAGUES_TO_FETCH,
    AUTO_FETCH, FORCE_REFRESH,
    MATCHES_CSV, MATCHES_XG_CSV, DATA_DIR,
    league_name,
)

LAST_FETCH_FILE = "data/last_fetch.txt"
REFRESH_INTERVAL_HOURS = 24
RESULTS_DAYS_DEFAULT = 7

RECENT_FETCH_INTERVAL_SECONDS = 60 * 60  # (kept for older code) the automation loop below wakes more often

# Optional settings in config.py (each has a default here):
import config as _config
XG_DAILY_BUDGET = int(getattr(_config, "XG_DAILY_BUDGET", 400))           # real-xG requests per day (0 = off)
XG_ENRICH_AT_STARTUP = bool(getattr(_config, "XG_ENRICH_AT_STARTUP", False))  # True = fetch ALL xG on a first run
AUTOMATION_INTERVAL_SECONDS = int(getattr(_config, "AUTOMATION_INTERVAL_SECONDS", 15 * 60))
STATS_DAILY_BUDGET = int(getattr(_config, "STATS_DAILY_BUDGET", 1000))          # Trends stats requests per UTC day (0 = off)
SHOTS_DAILY_BUDGET = int(getattr(_config, "SHOTS_DAILY_BUDGET", 1000))         # shots requests per UTC day (0 = off)

app = Flask(__name__)

# Display times in Nairobi (EAT, UTC+3, no DST). Stored data stays UTC.
EAT = timezone(timedelta(hours=3))


def _to_eat(value):
    try:
        ts = pd.Timestamp(value)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        return ts.astimezone(EAT)
    except Exception:
        return None


@app.template_filter("eat_date")
def eat_date(value):
    t = _to_eat(value)
    return t.strftime("%Y-%m-%d") if t is not None else str(value)[:10]


@app.template_filter("eat_time")
def eat_time(value):
    t = _to_eat(value)
    return t.strftime("%H:%M") if t is not None else str(value)[11:16]


@app.template_filter("eat_full")
def eat_full(value):
    t = _to_eat(value)
    return t.strftime("%Y-%m-%d %H:%M") if t is not None else str(value)[:16].replace("T", " ")
DF = None

_MATCHES_CACHE = {"at": None, "data": None}
CACHE_MINUTES = 30


def banner(msg):
    print("\n" + "=" * 68)
    print(f"  {msg}")
    print("=" * 68)


# ----------------------------------------------------------------------
# Freshness
# ----------------------------------------------------------------------
def should_refresh():
    if FORCE_REFRESH:
        return True
    if not os.path.exists(MATCHES_CSV):
        return True
    if not os.path.exists(LAST_FETCH_FILE):
        return True
    try:
        with open(LAST_FETCH_FILE) as f:
            last = datetime.fromisoformat(f.read().strip())
        age = datetime.now(timezone.utc) - last
        return age > timedelta(hours=REFRESH_INTERVAL_HOURS)
    except Exception:
        return True


def save_fetch_time():
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(LAST_FETCH_FILE, "w") as f:
        f.write(datetime.now(timezone.utc).isoformat())


def last_fetch_age_str():
    if not os.path.exists(LAST_FETCH_FILE):
        return "never"
    try:
        with open(LAST_FETCH_FILE) as f:
            last = datetime.fromisoformat(f.read().strip())
        delta = datetime.now(timezone.utc) - last
        hours = delta.total_seconds() / 3600
        if hours < 1:
            return f"{int(delta.total_seconds() / 60)} min ago"
        if hours < 48:
            return f"{hours:.1f} h ago"
        return f"{hours / 24:.1f} days ago"
    except Exception:
        return "unknown"


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------
def fetch_all_leagues():
    from fetch_data import fetch_two_seasons
    banner(f"STEP 1 — Fetching matches for {len(LEAGUES_TO_FETCH)} leagues")
    os.makedirs(DATA_DIR, exist_ok=True)

    frames = []
    for i, lid in enumerate(LEAGUES_TO_FETCH, 1):
        print(f"\n[{i}/{len(LEAGUES_TO_FETCH)}] {league_name(lid)} (id={lid})")
        try:
            df = fetch_two_seasons(lid)
            if df is not None and not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"  Failed: {e}")

    if not frames:
        print("\nERROR: No data fetched.")
        return None

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["event_id"]).reset_index(drop=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce", utc=True)
    combined = combined.sort_values("date").reset_index(drop=True)
    if os.path.exists(MATCHES_CSV):
        try:
            old_n = len(pd.read_csv(MATCHES_CSV, usecols=["event_id"]))
        except Exception:
            old_n = 0
        if len(combined) < 0.9 * old_n:
            print(f"\nWARNING: download incomplete ({len(combined)} vs {old_n} matches on disk, "
                  f"probably API quota). Keeping your existing data.")
            return None
    combined.to_csv(MATCHES_CSV, index=False)
    print(f"\n  Total: {len(combined)} matches saved")
    return combined


def enrich_xg(df):
    from fetch_data import enrich_with_xg
    banner("STEP 2 — Enriching xG (parallel; incremental via cache)")
    try:
        return enrich_with_xg(df, progress=True)
    except KeyboardInterrupt:
        print("\n  Interrupted. Cache saved — rerun to resume.")
        raise
    except Exception as e:
        print(f"  Failed: {e}. Using goals as fallback.")
        df["home_xg"] = df["home_goals"]
        df["away_xg"] = df["away_goals"]
        df.to_csv(MATCHES_XG_CSV, index=False)
        return df


def load_dataset():
    from model import load_data
    banner("STEP 3 — Loading dataset")
    df = load_data()
    n_teams = pd.concat([df["home_team"], df["away_team"]]).nunique()
    print(f"  Matches : {len(df)}")
    print(f"  Leagues : {df['league_id'].nunique()}")
    print(f"  Teams   : {n_teams}")
    print(f"  Range   : {df['date'].min().date()} -> {df['date'].max().date()}")
    return df


def _newest_match_date():
    """Date of the newest finished match on disk, e.g. '2026-10-25' (or 'unknown')."""
    try:
        newest = pd.to_datetime(pd.read_csv(MATCHES_CSV, usecols=["date"])["date"],
                                errors="coerce", utc=True).max()
        return newest.strftime("%Y-%m-%d")
    except Exception:
        return "unknown"


def _catchup_days():
    """Days to look back for finished matches: since the newest match on disk (3..30)."""
    try:
        newest = pd.to_datetime(pd.read_csv(MATCHES_CSV, usecols=["date"])["date"],
                                errors="coerce", utc=True).max()
        gap = (datetime.now(timezone.utc) - newest).days + 1
        return int(min(max(gap, 3), 30))
    except Exception:
        return 3


def run_daily_pipeline(force_no_refresh=False):
    global DF
    os.makedirs(DATA_DIR, exist_ok=True)

    if FORCE_REFRESH:
        banner("FORCE_REFRESH — wiping cached data")
        for f in (MATCHES_CSV, MATCHES_XG_CSV):
            if os.path.exists(f):
                os.remove(f)
                print(f"  Removed {f}")

    # A full 2-season, 14-league download costs thousands of API requests. It is only needed
    # when there is no data yet (or FORCE_REFRESH). Otherwise we just catch up on the matches
    # played since the newest one on disk.
    need_full_fetch = (FORCE_REFRESH or not os.path.exists(MATCHES_CSV)) and not force_no_refresh
    
    if need_full_fetch:
        if not AUTO_FETCH:
            print("Stale data but AUTO_FETCH=False. Skipping full fetch.")
        else:
            df = fetch_all_leagues()
            if df is None:
                if os.path.exists(MATCHES_CSV):
                    print("Full fetch failed — continuing with the data already on disk.")
                    need_full_fetch = False
                else:
                    sys.exit(1)
            else:
                save_fetch_time()
    else:
        print(f"✓ Data on disk — newest match: {_newest_match_date()} "
              f"(updated daily; last full download: {last_fetch_age_str()})")
        
        # Even if data is "fresh" (<24h), do a quick fetch to catch 
        # yesterday's matches that finished since the last run.
        if _job_done_today("recent_fetch"):
            print("  Recent matches already fetched today — skipping API call.")
        else:
            print("  Checking for recently finished matches...")
            from fetch_data import fetch_recent_matches
            ok = fetch_recent_matches(LEAGUES_TO_FETCH, days_back=_catchup_days())
            if ok is not False:
                _mark_job_done("recent_fetch")
            else:
                print("  Recent-match fetch was incomplete (API quota or error): not marked done, will retry.")

    if not os.path.exists(MATCHES_XG_CSV) or need_full_fetch:
        raw = pd.read_csv(MATCHES_CSV)
        if XG_ENRICH_AT_STARTUP:
            enrich_xg(raw)                      # one request per match: can use up a whole day's quota
        else:
            from fetch_data import write_proxy_xg
            write_proxy_xg(raw)
            print("  xG: goals stand in for now. Real xG is added every day, newest matches first "
                  f"(up to {XG_DAILY_BUDGET} requests a day; set XG_DAILY_BUDGET in config.py).")
    else:
        print("✓ xG data already present — skipping enrichment")

    DF = load_dataset()

    banner("READY — Web UI is starting")
    print(f"  Matches : http://127.0.0.1:5000/")
    print(f"  Stop    : Ctrl+C  or  pkill -f 'python app.py'")
    print("=" * 68 + "\n")


# ----------------------------------------------------------------------
# Match cache
# ----------------------------------------------------------------------
PICKS_CACHE_FILE = os.path.join(DATA_DIR, "picks_cache.json")


def _json_default(o):
    if hasattr(o, "item"):          # numpy scalars
        return o.item()
    return str(o)


_STALE_PICK_FIELDS = ("odds_all", "markets")          # fields older versions saved; dropped when picks are read


def _read_picks_cache(day):
    """Today's saved picks (same UTC day), or None."""
    try:
        with open(PICKS_CACHE_FILE, encoding="utf-8") as f:
            blob = json.load(f)
        if blob.get("day") == day and blob.get("data", {}).get("matches"):
            for m in blob["data"]["matches"]:                    # picks saved by older versions carry extra fields
                for stale in _STALE_PICK_FIELDS:
                    m.pop(stale, None)
            return blob
    except Exception:
        pass
    return None


def _write_picks_cache(day, stamp, data):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(PICKS_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({"day": day, "stamp": stamp, "data": data}, f, default=_json_default)
    except Exception as e:
        print(f"Could not save picks cache: {e}")


_LOAD_LOCK = threading.RLock()


def load_matches(force=False, days=2):
    """One caller at a time: the daily automation and a page view can never fetch the same fixtures twice."""
    with _LOAD_LOCK:
        return _load_matches_locked(force, days)


def _load_matches_locked(force=False, days=2):
    """Fixtures + picks are fetched ONCE per UTC day (the API quota resets at 00:00 UTC),
    saved to data/picks_cache.json, and reused on every page view and restart.
    Only the Refresh button (force=True) goes back to the API."""
    from scan import get_todays_matches

    now = datetime.now()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if not force:
        c = _MATCHES_CACHE
        if c["data"] is not None and c["at"] is not None:
            has = bool(c["data"].get("matches")) and not c["data"].get("stats", {}).get("partial")
            if has and c.get("day") == today:
                return c["data"]
            if not has and (now - c["at"]).total_seconds() < CACHE_MINUTES * 60:
                return c["data"]          # empty result (e.g. quota hit): don't hammer the API
        saved = _read_picks_cache(today)
        if saved:
            _MATCHES_CACHE.update(at=now, data=saved["data"], day=today, stamp=saved.get("stamp"))
            return saved["data"]

    try:
        data = get_todays_matches(days=days)
        stamp = datetime.now(EAT).strftime("%H:%M")
        _MATCHES_CACHE.update(at=now, data=data, day=today, stamp=stamp)
        if data.get("matches") and not data.get("stats", {}).get("partial"):
            _write_picks_cache(today, stamp, data)      # complete result only (not a half-failed fetch)
        elif data.get("stats", {}).get("fetch_errors"):
            print(f"[FETCH] API problems, not cached for the day: {data['stats']['fetch_errors'][:3]}")
        return data
    except Exception as e:
        print(f"Failed to load matches: {e}")
        return {"matches": [], "stats": {}}


def invalidate_matches_cache():
    """Drops the in-memory copy only. Today's saved picks are re-read from disk,
    so this no longer causes an API re-fetch."""
    _MATCHES_CACHE["at"] = None
    _MATCHES_CACHE["data"] = None


# ----------------------------------------------------------------------
# Results loader
# ----------------------------------------------------------------------
def _results_summary(results):
    try:
        from results_extra import summary
        return summary(results)
    except Exception:
        return {}


def load_recent_results(days=RESULTS_DAYS_DEFAULT):
    if DF is None:
        return []

    df = DF.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

    now = pd.Timestamp.now(tz="UTC")
    cutoff = now - pd.Timedelta(days=days)

    recent = df[(df["date"] >= cutoff) & (df["date"] <= now)].copy()
    if recent.empty:
        return []

    recent = recent.sort_values("date", ascending=False)
    event_ids = recent["event_id"].astype(int).tolist()

    try:
        from predictions_history import ensure_predictions, get_predictions_for
        ensure_predictions(df, event_ids, progress=False)
        pred_map = get_predictions_for(event_ids)
    except Exception as e:
        print(f"Predictions history failed: {e}")
        pred_map = {}

    results = []
    for _, row in recent.iterrows():
        try:
            hg = int(row["home_goals"])
            ag = int(row["away_goals"])
        except (TypeError, ValueError):
            continue

        total = hg + ag
        actual_over = total > 2.5
        eid = int(row["event_id"])

        pred = pred_map.get(eid)
        prediction = None
        if pred:
            pred_pick = pred.get("pred_over_pick", "")
            pred_won = (
                (pred_pick == "Over 2.5" and actual_over) or
                (pred_pick == "Under 2.5" and not actual_over)
            )
            actual_score = f"{hg}-{ag}"
            top1 = pred.get("top1_score", "")
            top2 = pred.get("top2_score", "")

            prediction = {
                "over_pick":   pred_pick,
                "over_prob":   pred.get("po") if pred_pick == "Over 2.5"
                               else pred.get("pu"),
                "over_hit":    pred_won,
                "top1_score":  top1,
                "top1_prob":   pred.get("top1_prob"),
                "top1_hit":    top1 == actual_score,
                "top2_score":  top2,
                "top2_prob":   pred.get("top2_prob"),
                "top2_hit":    top2 == actual_score,
            }

        results.append({
            "event_id":    eid,
            "date":        row["date"].isoformat(),
            "date_label":  row["date"].astimezone(EAT).strftime("%a %d %b"),
            "date_key":    row["date"].astimezone(EAT).strftime("%Y-%m-%d"),
            "time_label":  row["date"].astimezone(EAT).strftime("%H:%M"),
            "league_id":   int(row["league_id"]),
            "league_name": league_name(row["league_id"]),
            "home":        row["home_team"],
            "away":        row["away_team"],
            "home_goals":  hg,
            "away_goals":  ag,
            "total":       total,
            "over_hit":    actual_over,
            "score":       f"{hg}-{ag}",
            "prediction":  prediction,
        })

    try:
        from results_extra import build_extras
        extras = build_extras([r["event_id"] for r in results])
        for r in results:
            r["btts"] = extras.get(r["event_id"], {}).get("btts")
    except Exception as e:
        print(f"Results extras failed: {e}")

    return results


# ----------------------------------------------------------------------
# Background threads
# ----------------------------------------------------------------------
JOBS_FILE = os.path.join(DATA_DIR, "daily_jobs.json")


def _job_done_today(name, now=None):
    """True if background job `name` already ran today (UTC day = API quota day)."""
    try:
        with open(JOBS_FILE, encoding="utf-8") as f:
            return json.load(f).get(name) == (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
    except Exception:
        return False


def _mark_job_done(name, now=None):
    try:
        try:
            with open(JOBS_FILE, encoding="utf-8") as f:
                jobs = json.load(f)
        except Exception:
            jobs = {}
        known = {n for n, _f, _api in STEPS}
        for stale in [k for k in jobs if k not in known]:        # drop entries for jobs that no longer exist
            jobs.pop(stale, None)
        jobs[name] = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(JOBS_FILE, "w", encoding="utf-8") as f:
            json.dump(jobs, f)
    except Exception as e:
        print(f"Could not record job state: {e}")


# ----------------------------------------------------------------------
# Daily automation (everything the model needs, in one ordered job)
# ----------------------------------------------------------------------
# Every UTC day (the API quota day) these run in order, each at most once a day:
#   1. recent_fetch   finished matches since the newest one on disk
#   2. fixtures_scan  fixtures + real odds + picks for every market (saved for the day)
#   3. xg_upgrade     real xG for matches that only have goals as a stand-in (budgeted, newest first)
#   4. results_warm   grade recent predictions so the Results tab opens instantly (local, no API)
# A step that fails retries an hour later. A quota failure (429) holds every API step until the quota resets.
STATUS_FILE = os.path.join(DATA_DIR, "automation_status.json")
STEP_LABELS = {"recent_fetch": "Finished matches", "fixtures_scan": "Fixtures, odds & picks",
               "xg_upgrade": "Real xG upgrade", "shots_upgrade": "Team shots data", "stats_upgrade": "Match stats (Trends)",
               "results_warm": "Results grading"}
_RETRY_AT = {}
_QUOTA_HOLD_UNTIL = [0.0]
_status_lock = threading.Lock()


def _next_utc_midnight(now):
    return (now + timedelta(days=1)).replace(hour=0, minute=2, second=0, microsecond=0)   # +2 min after the reset


def _read_status():
    try:
        with open(STATUS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"steps": {}}


def _status_update(name, now, **fields):
    with _status_lock:
        st = _read_status()
        step = st.setdefault("steps", {}).setdefault(name, {})
        step.update(fields)
        step["last_try"] = now.isoformat()
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(STATUS_FILE, "w", encoding="utf-8") as f:
                json.dump(st, f)
        except Exception as e:
            print(f"Could not save automation status: {e}")


def _step_recent_fetch():
    import fetch_data as FD
    global DF
    ok = FD.fetch_recent_matches(LEAGUES_TO_FETCH, days_back=_catchup_days())
    info = dict(FD.LAST_FETCH_INFO)
    try:
        from model import load_data
        DF = load_data()                       # reload so the model refits on the new matches
    except Exception as e:
        print(f"[AUTOMATION] could not reload data: {e}")
    invalidate_matches_cache()
    return {"ok": ok is not False, "quota": bool(info["quota_hit"]),
            "detail": f"{info['new']} new finished matches" + (" (API quota used up)" if info["quota_hit"] else "")}


def _step_fixtures_scan():
    data = load_matches()
    matches = data.get("matches") or []
    stats = data.get("stats") or {}
    errors = stats.get("fetch_errors") or []
    quota = bool(stats.get("quota_hit")) or any("429" in str(e) for e in errors)
    if matches and not stats.get("partial"):
        return {"ok": True, "quota": False, "detail": f"{len(matches)} fixtures with odds and picks"}
    if quota:
        return {"ok": False, "quota": True, "detail": "API quota used up"}
    if errors:
        return {"ok": False, "quota": False, "detail": f"fetch problems: {str(errors[0])[:80]}"}
    return {"ok": True, "quota": False, "detail": "no upcoming fixtures"}      # a real quiet day: do not keep asking


def _step_xg_upgrade():
    import fetch_data as FD
    global DF
    if XG_DAILY_BUDGET <= 0:
        return {"ok": True, "quota": False, "detail": "off (XG_DAILY_BUDGET = 0)"}
    st = FD.upgrade_xg(budget=XG_DAILY_BUDGET, verbose=True)
    if st.get("error"):
        return {"ok": False, "quota": False, "detail": st["error"]}
    if st["upgraded"]:
        try:
            from model import load_data
            DF = load_data()
        except Exception as e:
            print(f"[AUTOMATION] could not reload data: {e}")
        invalidate_matches_cache()
    cov = FD.xg_coverage()
    detail = (f"{st['upgraded']} upgraded, {st['no_xg']} without xG yet · real xG on {cov['pct']}% of matches, "
              f"{cov['backlog']} to go" + (" · quota used up, continues tomorrow" if st["quota_hit"] else ""))
    all_failed = st["tried"] > 0 and st["errors"] == st["tried"]
    return {"ok": not all_failed, "quota": st["quota_hit"], "detail": detail}


def _step_shots_upgrade():
    """Fetches shots for every finished match that lacks them, up to SHOTS_DAILY_BUDGET requests per UTC day in total
    (restarts and the Shots tab button count too). Runs right after startup, and keeps going on later days until done."""
    import shots_data as SD
    if SHOTS_DAILY_BUDGET <= 0:
        return {"ok": True, "quota": False, "detail": "off (SHOTS_DAILY_BUDGET = 0)"}
    allowed = SHOTS_DAILY_BUDGET - SD.used_today()
    if allowed <= 0:
        return {"ok": True, "quota": False, "detail": f"today's budget of {SHOTS_DAILY_BUDGET} requests is used"}
    upcoming = (_MATCHES_CACHE.get("data") or {}).get("matches") or []
    st = SD.fetch_all_shots(budget=allowed, upcoming=upcoming, verbose=True)
    cov = SD.coverage()
    detail = (f"{st['ok']} fetched, {st['none']} without shots · shots on {cov['pct']}% of {cov['matches']} matches"
              + (f", {st['left']} to go" if st["left"] else ", all done")
              + (" · quota used up, continues tomorrow" if st["quota_hit"] else ""))
    all_failed = st["asked"] > 0 and st["errors"] == st["asked"]
    return {"ok": not all_failed, "quota": st["quota_hit"], "detail": detail}


def _step_stats_upgrade():
    """Full match stats (shots, corners, fouls, cards ...) for the Trends tab. At launch it first fetches the last 10 matches of
    the teams in the upcoming fixtures (so Trends works straight away), then the rest of the history, newest first, up to
    STATS_DAILY_BUDGET requests per UTC day in total. Every request also fills the xG and shots caches."""
    import match_stats as MS
    global DF
    if STATS_DAILY_BUDGET <= 0:
        return {"ok": True, "quota": False, "detail": "off (STATS_DAILY_BUDGET = 0)"}
    allowed = max(STATS_DAILY_BUDGET - MS.used_today(), 0)
    upcoming = (_MATCHES_CACHE.get("data") or {}).get("matches") or []
    st = MS.upgrade_stats(budget=allowed, upcoming=upcoming, verbose=True)
    if st["xg_added"]:                               # real xG arrived with the stats: bring matches_with_xg.csv and the model up to date
        try:
            import fetch_data as FD
            from model import load_data
            FD.upgrade_xg(budget=0, verbose=False)   # budget 0 = no requests, only syncs the csv from the xG cache
            DF = load_data()
            invalidate_matches_cache()
        except Exception as e:
            print(f"[AUTOMATION] could not sync xG: {e}")
    cov = MS.coverage()
    detail = (f"{st['ok']} stored ({st['priority']} for today's teams first), {st['none']} without stats, +{st['xg_added']} real xG "
              f"· full stats on {cov['pct']}% of {cov['matches']} matches"
              + (f", {st['left']} to go" if st["left"] else ", all done")
              + (" · quota used up, continues tomorrow" if st["quota_hit"] else ""))
    all_failed = st["asked"] > 0 and st["errors"] == st["asked"]
    return {"ok": not all_failed, "quota": st["quota_hit"], "detail": detail}


def _step_results_warm():
    if DF is None:
        return {"ok": False, "quota": False, "detail": "data not loaded yet"}
    results = load_recent_results(RESULTS_DAYS_DEFAULT)
    return {"ok": True, "quota": False, "detail": f"{len(results)} finished matches graded"}


STEPS = [("recent_fetch", _step_recent_fetch, True), ("fixtures_scan", _step_fixtures_scan, True),
         ("xg_upgrade", _step_xg_upgrade, True), ("shots_upgrade", _step_shots_upgrade, True), ("stats_upgrade", _step_stats_upgrade, True),
         ("results_warm", _step_results_warm, False)]


def run_automation_cycle(now=None):
    """Runs every step that is due. Safe to call often; each step runs at most once per UTC day."""
    now = now or datetime.now(timezone.utc)
    ts = now.timestamp()
    ran = []
    for name, fn, uses_api in STEPS:
        if _job_done_today(name, now) or ts < _RETRY_AT.get(name, 0):
            continue
        if uses_api and ts < _QUOTA_HOLD_UNTIL[0]:
            _status_update(name, now, result="waiting", detail="waiting for the API quota to reset",
                           next_try=datetime.fromtimestamp(_QUOTA_HOLD_UNTIL[0], timezone.utc).isoformat())
            continue
        print(f"\n[AUTOMATION] {STEP_LABELS[name]}...")
        try:
            res = fn()
        except Exception as e:
            res = {"ok": False, "quota": False, "detail": f"error: {e}"}
        quota = bool(res.get("quota"))
        if res["ok"]:
            _mark_job_done(name, now)
            _RETRY_AT.pop(name, None)
            result, nxt = "ok", None
        elif quota:
            _RETRY_AT[name] = _next_utc_midnight(now).timestamp()
            result, nxt = "quota", _next_utc_midnight(now)
        else:
            _RETRY_AT[name] = ts + 3600
            result, nxt = "error", now + timedelta(hours=1)
        if quota and uses_api:
            _QUOTA_HOLD_UNTIL[0] = _next_utc_midnight(now).timestamp()        # no more API steps today
            if res["ok"]:
                nxt = None
        fields = dict(result=result, detail=res.get("detail", ""), next_try=nxt.isoformat() if nxt else None)
        if res["ok"]:
            fields["last_ok"] = now.isoformat()
        _status_update(name, now, **fields)
        print(f"[AUTOMATION] {STEP_LABELS[name]}: {result} · {res.get('detail', '')}")
        ran.append((name, result))
    return ran


def automation_loop():
    time.sleep(5)                                   # let the web server come up first
    while True:
        try:
            run_automation_cycle()
        except Exception as e:
            print(f"[AUTOMATION] cycle error: {e}")
        time.sleep(AUTOMATION_INTERVAL_SECONDS)


def automation_status():
    """Everything the UI, /health and --status show about the automation. Local files only: no API calls."""
    st = _read_status().get("steps", {})
    steps = []
    for name, _fn, _api in STEPS:
        s = st.get(name, {})
        if _job_done_today(name):
            state, when = "done today", s.get("last_ok")
        elif s.get("result") == "quota":
            state, when = "waiting for the quota reset", s.get("next_try")
        elif s.get("result") == "error":
            state, when = "will retry", s.get("next_try")
        elif s.get("result") == "waiting":
            state, when = "waiting for the quota reset", s.get("next_try")
        else:
            state, when = "not run yet today", None
        steps.append({"name": name, "label": STEP_LABELS[name], "state": state, "detail": s.get("detail", ""),
                      "when": eat_full(when) + " EAT" if when else ""})
    try:
        from fetch_data import xg_coverage
        xg = xg_coverage()
    except Exception:
        xg = {"matches": 0, "real": 0, "pct": 0, "backlog": 0, "gave_up": 0}
    return {"steps": steps, "xg": xg, "xg_daily_budget": XG_DAILY_BUDGET}


def start_background_threads():
    t = threading.Thread(target=automation_loop, daemon=True)
    t.start()
    print("✓ Daily automation running: finished matches → fixtures & picks → real xG → results (each once a day)")


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Local team lookup (works offline; falls back to the API only if needed)
# ----------------------------------------------------------------------
def _norm(text):
    import unicodedata
    t = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in t if not unicodedata.combining(c)).lower().strip()


def _rank_selections(pred):
    """Over/Under 2.5, BTTS and 1X2 on the Single Match card as picks, ranked by the model's probability for the
    chosen side. Each: market, pick, prob (%), tier, real odds (or None), model odds, EV (if real)."""
    real = pred.get("real") or {}
    items = []

    def add(market, pick, prob, tier, real_odds):
        if prob is None:
            return
        items.append({"market": market, "pick": pick, "prob": round(float(prob), 1), "tier": tier,
                      "real": real_odds, "model": round(100 / float(prob), 2),
                      "ev": round(float(prob) / 100 * real_odds - 1, 3) if real_odds else None})

    po, pu = pred.get("prob_over_2_5"), pred.get("prob_under_2_5")
    if po is not None and pu is not None:
        over = po >= pu
        add("Over/Under 2.5", "Over 2.5" if over else "Under 2.5", po if over else pu,
            pred.get("confidence"), real.get("over25") if over else real.get("under25"))

    b = pred.get("btts")
    if b:
        add("BTTS", b["pick"], b["prob"] * 100, b["tier"], b.get("odds"))

    x = pred.get("x12")
    if x:
        label = {"H": f"{pred.get('home_team', 'Home')} to win", "D": "Draw",
                 "A": f"{pred.get('away_team', 'Away')} to win"}[x["pick_key"]]
        add("Match Result 1X2", label, x["prob"] * 100, x["tier"], x.get("odds"))

    items.sort(key=lambda i: i["prob"], reverse=True)
    return items


def _cached_fixture(hid, aid):
    """This pairing's upcoming fixture from today's SAVED picks (never calls the API), or None.
    Used to show real bookmaker odds on the Single Match card."""
    data = _MATCHES_CACHE.get("data")
    if not data:
        saved = _read_picks_cache(datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        data = saved["data"] if saved else None
    for m in (data or {}).get("matches", []):
        if m.get("home_id") == hid and m.get("away_id") == aid:
            return m
    return None


def find_team_local(df, name, league_id=None):
    """Return (team_id, team_name) from the local dataset, or (None, None)."""
    import difflib
    if df is None or df.empty or not name:
        return None, None
    home = df[["league_id", "home_team_id", "home_team"]]
    away = df[["league_id", "away_team_id", "away_team"]]
    home.columns = away.columns = ["league_id", "team_id", "team"]
    teams = pd.concat([home, away], ignore_index=True)
    teams["n"] = teams["team"].map(_norm)

    q = _norm(name)
    pools = []
    if league_id is not None:
        pools.append(teams[teams["league_id"] == league_id])
    pools.append(teams)                     # then any league (e.g. a CL team from another league)

    for pool in pools:
        if pool.empty:
            continue
        counts = pool.groupby(["team_id", "team", "n"]).size().reset_index(name="apps")
        counts = counts.sort_values("apps", ascending=False)
        exact = counts[counts["n"] == q]
        if len(exact):
            r = exact.iloc[0]
            return int(r.team_id), r.team
        contains = counts[counts["n"].str.contains(q, regex=False)]
        if len(contains):
            r = contains.iloc[0]
            return int(r.team_id), r.team
        close = difflib.get_close_matches(q, counts["n"].tolist(), n=1, cutoff=0.75)
        if close:
            r = counts[counts["n"] == close[0]].iloc[0]
            return int(r.team_id), r.team
    return None, None


def _shots_window():
    try:
        import shots_data
        return shots_data.SHOTS_WINDOW
    except Exception:
        return 10


@app.route("/", methods=["GET", "POST"])
def index():
    from model import full_prediction
    from fetch_data import find_team

    result = None
    home_input, away_input = "", ""
    league_id = DEFAULT_LEAGUE_ID

    force_refresh = request.args.get("refresh") == "1"
    data = load_matches(force=force_refresh) if DF is not None else \
           {"matches": [], "stats": {}}

    all_matches = data["matches"]
    stats = data["stats"]
    last_updated = _MATCHES_CACHE.get("stamp") or "—"

    try:
        results_days = int(request.args.get("results_days",
                                            RESULTS_DAYS_DEFAULT))
        if results_days not in (3, 7, 14, 30):
            results_days = RESULTS_DAYS_DEFAULT
    except (TypeError, ValueError):
        results_days = RESULTS_DAYS_DEFAULT

    results = load_recent_results(days=results_days)

    if request.method == "POST":
        home_input = request.form.get("home_team", "").strip()
        away_input = request.form.get("away_team", "").strip()
        try:
            league_id = int(request.form.get("league_id", DEFAULT_LEAGUE_ID))
        except ValueError:
            league_id = DEFAULT_LEAGUE_ID

        if DF is None or DF.empty:
            result = {"error": "No dataset loaded."}
        elif not home_input or not away_input:
            result = {"error": "Select both teams."}
        else:
            hid, hname = find_team_local(DF, home_input, league_id)
            if not hid:
                hid, hname = find_team(home_input, league_id)      # API fallback (needs internet)
            aid, aname = find_team_local(DF, away_input, league_id)
            if not aid:
                aid, aname = find_team(away_input, league_id)
            if hid and aid and hid == aid:
                result = {"error": "Pick two different teams."}
            elif not hid or not aid:
                result = {"error": f"Team not found. "
                                   f"Home: {hname or 'x'} | Away: {aname or 'x'}"}
            else:
                try:
                    pred = full_prediction(DF, hid, aid, league_id)
                    pred["home_team"] = hname
                    pred["away_team"] = aname
                    pred["league_name"] = league_name(league_id)
                    fx = _cached_fixture(hid, aid)
                    pred["real"] = ({"over25": fx.get("over_odds"), "under25": fx.get("under_odds"),
                                     "btts_yes": fx.get("btts_yes_odds"), "btts_no": fx.get("btts_no_odds"),
                                     "odds_home": fx.get("odds_home"), "odds_draw": fx.get("odds_draw"),
                                     "odds_away": fx.get("odds_away"), "kickoff": fx.get("date")} if fx else None)
                    try:
                        from btts_model import single as btts_single
                        pred["btts"] = btts_single(pred["lambda_home"], pred["lambda_away"],
                                                   (pred["real"] or {}).get("btts_yes"),
                                                   (pred["real"] or {}).get("btts_no"))
                    except Exception as e:
                        pred["btts"] = None
                    try:
                        from match_result import single as x12_single
                        rl = pred["real"] or {}
                        pred["x12"] = x12_single(pred["lambda_home"], pred["lambda_away"],
                                                 rl.get("odds_home"), rl.get("odds_draw"), rl.get("odds_away"))
                    except Exception as e:
                        print(f"1X2 for single match failed: {e}")
                        pred["x12"] = None
                    pred["ranked"] = _rank_selections(pred)
                    result = pred
                except Exception as e:
                    result = {"error": f"Prediction failed: {e}"}

    try:
        import acca_builder as AB
        acca_days = AB.days_available(all_matches)
        acca_cfg = {"markets": [{"key": k, "title": t, "sides": list(sd), "min_prob": round(AB.default_min_prob(k) * 100)}
                                for k, (t, sd) in AB.MARKETS.items()],
                    "min_prob": round(AB.DEFAULT_MIN_PROB * 100), "min_odds": AB.DEFAULT_MIN_ODDS,
                    "max_legs": AB.MAX_LEGS, "min_legs": AB.MIN_LEGS,
                    "min_combined": AB.DEFAULT_MIN_COMBINED}
    except Exception as e:
        print(f"acca setup failed: {e}")
        acca_days, acca_cfg = [], {"markets": [], "min_prob": 58, "min_odds": 1.15, "max_legs": 3, "min_legs": 2, "min_combined": 2.0}

    try:
        from btts_model import btts_predictions
        btts_rows = btts_predictions(all_matches)
        _btts_by_id = {r["event_id"]: r for r in btts_rows}
        for _m in all_matches:                       # so All Matches cards can show BTTS too
            _m["btts"] = _btts_by_id.get(_m["event_id"])
    except Exception as e:
        print(f"BTTS build failed: {e}")
        btts_rows = []

    try:
        from match_result import match_result_predictions
        x12_rows = match_result_predictions(all_matches)
    except Exception as e:
        print(f"1X2 build failed: {e}")
        x12_rows = []

    teams_by_league = {}
    if DF is not None and not DF.empty:
        for lid, g in DF.groupby("league_id"):
            names = pd.concat([g["home_team"], g["away_team"]]).dropna().astype(str).unique().tolist()
            teams_by_league[str(int(lid))] = sorted(names, key=str.lower)

    leagues_list = sorted(DF["league_id"].unique().tolist()) if DF is not None else []
    leagues_named = [(lid, league_name(lid)) for lid in leagues_list]

    return render_template(
        "index.html",
        result=result,
        home_input=home_input,
        away_input=away_input,
        league_id=league_id,
        leagues=leagues_named,
        all_matches=all_matches,
        stats=stats,
        picks_last_updated=last_updated,
        last_fetch=last_fetch_age_str(),
        automation=automation_status(),
        results=results,
        rx=_results_summary(results),
        results_days=results_days,
        acca_days=acca_days,                  # [{"iso", "label", "count"}]: days that still have upcoming fixtures
        acca_cfg=acca_cfg,                    # market list + default settings for the Acca tab
        teams_by_league=teams_by_league,
        shots_window=_shots_window(),
        btts_rows=sorted(btts_rows, key=lambda r: r["prob"], reverse=True),   # highest probability first
        x12_rows=sorted(x12_rows, key=lambda r: r["prob"], reverse=True),     # most likely outcome first
    )


@app.route("/api/matches")
def api_matches():
    return jsonify(load_matches())


@app.route("/api/acca")
def api_acca():
    """Acca for ONE market from today's saved fixtures and real bookmaker odds. Makes no API requests of its own.
    Query: market (ou25|btts|1x2), side (best or a side), day (YYYY-MM-DD, EAT), legs (auto = best of 2 or 3 matches,
    or exactly 2, or exactly 3), min_prob (%), min_odds (per leg), min_combined (combined odds target).
    By probability only."""
    import acca_builder as AB
    a = request.args

    def num(name, default, lo, hi):
        try:
            return max(lo, min(hi, float(a.get(name, default))))
        except (TypeError, ValueError):
            return default

    raw_legs = (a.get("legs") or "auto").strip().lower()
    if raw_legs in ("2", "3"):
        legs_lo = legs_hi = int(raw_legs)                      # exactly 2 or exactly 3 matches
    else:
        legs_lo, legs_hi = AB.MIN_LEGS, AB.DEFAULT_LEGS        # auto: the best of 2 or 3
    data = load_matches() if DF is not None else {"matches": []}
    try:
        res = AB.build_acca(
            data.get("matches", []), a.get("market", "ou25"), a.get("side", "best"), day=a.get("day") or None,
            legs=legs_hi, min_legs=legs_lo,
            min_prob=num("min_prob", AB.DEFAULT_MIN_PROB * 100, 0, 99) / 100.0,
            min_odds=num("min_odds", AB.DEFAULT_MIN_ODDS, 1.0, 50.0),
            min_combined=num("min_combined", AB.DEFAULT_MIN_COMBINED, 1.01, 1000.0))
    except ValueError as e:
        return jsonify({"ok": False, "message": str(e)}), 400
    return jsonify(res)


def _shots_fixtures(iso, allow_api):
    """Fixtures for one Nairobi day: finished matches on disk + today's saved picks. If the day is today or later
    and nothing upcoming is known, the API is asked only when allow_api is True (one request per league)."""
    import shots_data as SD
    data = load_matches() if DF is not None else {"matches": []}
    fixtures = SD.day_fixtures(iso, data.get("matches", []))
    needs_load, err = False, None
    if iso >= datetime.now(EAT).strftime("%Y-%m-%d") and not any(f["status"] == "upcoming" for f in fixtures):
        if allow_api:
            extra, err = SD.fetch_fixtures_for_day(iso)
            seen = {f["event_id"] for f in fixtures}
            fixtures += [f for f in extra if f["event_id"] not in seen]
        else:
            needs_load = True
    return fixtures, needs_load, err


@app.route("/api/shots")
def api_shots():
    """Shots per team for one Nairobi day (?date=YYYY-MM-DD; past or future). Reads local files only, unless
    load=1 is given for a day with no known fixtures (then one request per league fetches them)."""
    import shots_data as SD
    iso = (request.args.get("date") or datetime.now(EAT).strftime("%Y-%m-%d")).strip()
    try:
        datetime.strptime(iso, "%Y-%m-%d")
    except ValueError:
        return jsonify({"ok": False, "message": "Bad date."}), 400
    fixtures, needs_load, err = _shots_fixtures(iso, request.args.get("load") == "1")
    rows, cov = SD.build_rows(fixtures)
    cov.pop("fetchable_ids", None)
    return jsonify({"ok": True, "date": iso, "rows": rows, "coverage": cov, "needs_load": needs_load,
                    "error": err, "window": SD.SHOTS_WINDOW, "overall": SD.coverage()})


@app.route("/api/shots/fill", methods=["POST"])
def api_shots_fill():
    """Fetch the shots still missing behind one day's fixtures (spends API requests, capped)."""
    import shots_data as SD
    body = request.get_json(silent=True) or {}
    iso = str(body.get("date") or "")
    try:
        datetime.strptime(iso, "%Y-%m-%d")
        limit = max(1, min(int(body.get("limit", 200)), 400))
    except (ValueError, TypeError):
        return jsonify({"ok": False, "message": "Bad request."}), 400
    fixtures, _nl, _err = _shots_fixtures(iso, False)
    ids = SD.missing_ids_for(fixtures)
    res = SD.fetch_shots(ids, budget=limit, verbose=True)
    res.update(success=True, remaining=max(len(ids) - res["ok"] - res["none"], 0))
    return jsonify(res)


@app.route("/api/trends")
def api_trends():
    """Last-10 trend cards for the teams playing on one Nairobi day (?date=YYYY-MM-DD&min_hits=10). Local files only,
    unless load=1 is given for a day with no known fixtures (then one request per league fetches them)."""
    import match_stats as MS
    iso = (request.args.get("date") or datetime.now(EAT).strftime("%Y-%m-%d")).strip()
    try:
        datetime.strptime(iso, "%Y-%m-%d")
        min_hits = max(7, min(int(request.args.get("min_hits", 10)), 10))
    except (ValueError, TypeError):
        return jsonify({"ok": False, "message": "Bad request."}), 400
    fixtures, needs_load, err = _shots_fixtures(iso, request.args.get("load") == "1")
    all_cards, info, top = MS.build_cards(fixtures)
    cards = MS.cards_for_level(all_cards, min_hits)
    return jsonify({"ok": True, "date": iso, "cards": cards[:1500], "total_cards": len(cards), "fixtures": len(fixtures),
                    "top": top, "info": info, "needs_load": needs_load, "error": err, "overall": MS.coverage(),
                    "markets": [{"key": k, "label": v[0]} for k, v in MS.MARKETS.items()] + [{"key": k, "label": v} for k, v in MS.GOAL_MARKETS.items()],
                    "variants": [{"key": k, "label": v} for k, v in MS.VARIANTS.items()]})


@app.route("/api/trends/fill", methods=["POST"])
def api_trends_fill():
    """Fetch the match stats still missing behind one day's fixtures (each team's last 10 matches); spends API requests, capped."""
    import match_stats as MS
    body = request.get_json(silent=True) or {}
    iso = str(body.get("date") or "")
    try:
        datetime.strptime(iso, "%Y-%m-%d")
        limit = max(1, min(int(body.get("limit", 300)), 500))
    except (ValueError, TypeError):
        return jsonify({"ok": False, "message": "Bad request."}), 400
    fixtures, _nl, _err = _shots_fixtures(iso, False)
    ids = MS.needed_ids(fixtures)
    res = MS.fetch_stats(ids, budget=limit, verbose=True)
    res.update(success=True, remaining=max(len(ids) - res["ok"] - res["none"], 0))
    return jsonify(res)


@app.route("/api/results")
def api_results():
    days = request.args.get("days", RESULTS_DAYS_DEFAULT)
    try:
        days = int(days)
    except ValueError:
        days = RESULTS_DAYS_DEFAULT
    return jsonify(load_recent_results(days=days))


@app.route("/health")
def health():
    return {
        "status": "ok",
        "matches_loaded": 0 if DF is None else len(DF),
        "last_fetch": last_fetch_age_str(),
        "automation": automation_status(),
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def cmd_status():
    banner("DATA STATUS")
    print(f"  matches.csv          : "
          f"{'exists' if os.path.exists(MATCHES_CSV) else 'missing'}")
    print(f"  matches_with_xg.csv  : "
          f"{'exists' if os.path.exists(MATCHES_XG_CSV) else 'missing'}")
    print(f"  xg_cache.csv         : "
          f"{'exists' if os.path.exists('data/xg_cache.csv') else 'missing'}")
    print(f"  predictions_history  : "
          f"{'exists' if os.path.exists('data/predictions_history.csv') else 'missing'}")
    print(f"  last full download   : {last_fetch_age_str()}")
    print(f"  newest match on disk : {_newest_match_date()}")
    a = automation_status()
    print(f"  real xG coverage     : {a['xg']['pct']}% of {a['xg']['matches']} matches "
          f"({a['xg']['backlog']} to upgrade, {XG_DAILY_BUDGET}/day budget)")
    for step in a["steps"]:
        print(f"  {step['label']:<22}: {step['state']}" + (f" ({step['when']})" if step["when"] else "")
              + (f" · {step['detail']}" if step["detail"] else ""))



def cmd_scan():
    from scan import get_todays_matches
    result = get_todays_matches(days=2)
    s = result["stats"]

    print("=" * 78)
    print("  DAILY SCAN")
    print("=" * 78)
    print(f"  Upcoming: {s['total_upcoming']}  |  "
          f"With history: {s['with_history']}\n")

    for m in result["matches"]:
        icon = "🟢" if m["po"] >= 50 else "🔵"
        print(f"{icon}  {m['date'][:16]}  [{m['league_name']}]")
        print(f"     {m['home']} vs {m['away']}")
        print(f"     Over 2.5: {m['po']}%   Under: {m['pu']}%   {m['tier']}")
        print(f"     xG: {m['lh']} – {m['la']}")
        top = m["top_2_scores"]
        if top:
            print(f"     Top: {top[0]['score']} ({top[0]['prob']}%)"
                  + (f", {top[1]['score']} ({top[1]['prob']}%)"
                     if len(top) > 1 else ""))
        print()


if __name__ == "__main__":
    if "--status" in sys.argv:
        cmd_status()
        sys.exit(0)

    if "--scan" in sys.argv:
        if os.path.exists(MATCHES_XG_CSV):
            from model import load_data
            try:
                DF = load_data()
            except Exception as e:
                print(f"Failed to load: {e}")
                sys.exit(1)
        cmd_scan()
        sys.exit(0)

    if "--all-matches" in sys.argv:
        if os.path.exists(MATCHES_XG_CSV):
            from model import load_data
            try:
                DF = load_data()
            except Exception as e:
                print(f"Failed to load: {e}")
                sys.exit(1)
        cmd_scan()
        sys.exit(0)

    force_no_refresh = "--no-refresh" in sys.argv

    try:
        run_daily_pipeline(force_no_refresh=force_no_refresh)
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(0)

    start_background_threads()

    app.run(host="0.0.0.0", port=5000, debug=False)
