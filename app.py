# app.py
"""
Single-entry daily runner with a once-a-day recent-matches fetcher.
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

RECENT_FETCH_INTERVAL_SECONDS = 60 * 60  # 1 hour

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
            fetch_recent_matches(LEAGUES_TO_FETCH, days_back=_catchup_days())
            _mark_job_done("recent_fetch")

    if not os.path.exists(MATCHES_XG_CSV) or need_full_fetch:
        raw = pd.read_csv(MATCHES_CSV)
        enrich_xg(raw)
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


def _read_picks_cache(day):
    """Today's saved picks (same UTC day), or None."""
    try:
        with open(PICKS_CACHE_FILE, encoding="utf-8") as f:
            blob = json.load(f)
        if blob.get("day") == day and blob.get("data", {}).get("matches"):
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


def load_matches(force=False, days=2):
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
            r["ht"] = extras.get(r["event_id"], {}).get("ht")
            r["corners"] = extras.get(r["event_id"], {}).get("corners")
            r["btts"] = extras.get(r["event_id"], {}).get("btts")
    except Exception as e:
        print(f"Results extras failed: {e}")

    return results


# ----------------------------------------------------------------------
# Background threads
# ----------------------------------------------------------------------
JOBS_FILE = os.path.join(DATA_DIR, "daily_jobs.json")


def _job_done_today(name):
    """True if background job `name` already ran today (UTC day = API quota day)."""
    try:
        with open(JOBS_FILE, encoding="utf-8") as f:
            return json.load(f).get(name) == datetime.now(timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return False


def _mark_job_done(name):
    try:
        try:
            with open(JOBS_FILE, encoding="utf-8") as f:
                jobs = json.load(f)
        except Exception:
            jobs = {}
        jobs[name] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(JOBS_FILE, "w", encoding="utf-8") as f:
            json.dump(jobs, f)
    except Exception as e:
        print(f"Could not record job state: {e}")


def recent_fetch_loop():
    """Once per UTC day: pull newly finished matches. The loop only wakes hourly to check."""
    while True:
        try:
            if not _job_done_today("recent_fetch"):
                print("\n[RECENT FETCH] Checking for new finished matches...")
                from fetch_data import fetch_recent_matches
                fetch_recent_matches(LEAGUES_TO_FETCH, days_back=_catchup_days())

                # Reload DF into memory so the UI updates
                global DF
                from model import load_data
                DF = load_data()
                _mark_job_done("recent_fetch")
                print("[RECENT FETCH] Dataset reloaded.")
                invalidate_matches_cache()          # memory only; picks re-read from today's file
        except Exception as e:
            print(f"[RECENT FETCH] error: {e}")
        time.sleep(RECENT_FETCH_INTERVAL_SECONDS)


def corners_update_loop():
    """Once per UTC day: fetch corner counts for newly finished matches (max 400 requests)."""
    time.sleep(120)                      # let the app finish starting first
    while True:
        try:
            if not _job_done_today("corners_update"):
                from corners_data import update_corners
                added = update_corners(max_new=400, verbose=False)
                _mark_job_done("corners_update")
                if added:
                    print(f"\n[CORNERS] {added} match(es) with corner data added.")
                    invalidate_matches_cache()
            if not _job_done_today("ht_update"):
                from ht_data import update_ht
                added = update_ht(max_new=getattr(__import__("config"), "HT_DAILY_MAX", 500), verbose=False)
                _mark_job_done("ht_update")
                if added:
                    print(f"\n[HALF-TIME] {added} match(es) with half-time score added.")
                    invalidate_matches_cache()
        except Exception as e:
            print(f"[CORNERS] error: {e}")
        time.sleep(RECENT_FETCH_INTERVAL_SECONDS)


def start_background_threads():
    t3 = threading.Thread(target=corners_update_loop, daemon=True)
    t3.start()
    print("✓ Corners updater running (once a day)")

    t2 = threading.Thread(target=recent_fetch_loop, daemon=True)
    t2.start()
    print("✓ Recent match fetcher running (once a day)")


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


def _single_extras(hid, aid, league_id):
    """Half-time and corners predictions for the Single Match card."""
    out = {"ht": None, "corners": None}

    try:
        import ht_model as H
        model = H.get_model()
        if model is None:
            out["ht"] = {"ok": False, "message": "No half-time data yet (run python ht_data.py)."}
        else:
            fit, info, team = model
            th, ta = team.get(int(hid)), team.get(int(aid))
            if not th or not ta or th["n"] < H.MIN_MATCHES or ta["n"] < H.MIN_MATCHES:
                out["ht"] = {"ok": False, "message": "Not enough half-time history for one of these teams."}
            else:
                mh, ma = H.expected_ht(fit, int(hid), int(aid), league_id)
                po = H.prob_over(mh + ma)
                over = po >= 0.5
                prob = po if over else 1 - po
                out["ht"] = {"ok": True, "line": H.LINE, "mu_h": round(mh, 2), "mu_a": round(ma, 2),
                             "mu_t": round(mh + ma, 2), "p_over": round(po * 100, 1),
                             "p_under": round((1 - po) * 100, 1),
                             "pick": f"{'Over' if over else 'Under'} {H.LINE:g} HT", "is_over": over,
                             "prob": round(prob * 100, 1), "tier": H._tier(prob),
                             "top_scores": H.top_scores(mh, ma, n=2)}
    except Exception as e:
        out["ht"] = {"ok": False, "message": f"Half-time prediction failed: {e}"}

    try:
        import corners_model as C
        model = C.get_model()
        if model is None:
            out["corners"] = {"ok": False, "message": "No corner data yet (run python corners_data.py)."}
        else:
            fit, phi, info, team = model
            th, ta = team.get(int(hid)), team.get(int(aid))
            if not th or not ta or th["n"] < C.MIN_MATCHES or ta["n"] < C.MIN_MATCHES:
                out["corners"] = {"ok": False, "message": "Not enough corner history for one of these teams."}
            else:
                mh, ma = C.expected_corners(fit, int(hid), int(aid), league_id)
                po = C.prob_over(mh + ma, phi)
                over = po >= 0.5
                prob = po if over else 1 - po
                out["corners"] = {"ok": True, "line": C.LINE, "mu_h": round(mh, 1), "mu_a": round(ma, 1),
                                  "mu_t": round(mh + ma, 1), "p_over": round(po * 100, 1),
                                  "p_under": round((1 - po) * 100, 1),
                                  "pick": f"{'Over' if over else 'Under'} {C.LINE:g}", "is_over": over,
                                  "prob": round(prob * 100, 1), "tier": C._tier(prob)}
    except Exception as e:
        out["corners"] = {"ok": False, "message": f"Corners prediction failed: {e}"}
    return out


def _rank_selections(pred):
    """All markets on the Single Match card as picks, ranked by the model's probability for the
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

    h = pred.get("ht")
    if h and h.get("ok"):
        add("First half O/U 1.5", h["pick"], h["prob"], h["tier"], None)

    c = pred.get("corners")
    if c and c.get("ok"):
        add(f"Corners O/U {c['line']:g}", f"{c['pick']} corners", c["prob"], c["tier"], None)

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
                                     "kickoff": fx.get("date")} if fx else None)
                    pred.update(_single_extras(hid, aid, league_id))     # "ht" and "corners"
                    try:
                        from btts_model import single as btts_single
                        pred["btts"] = btts_single(pred["lambda_home"], pred["lambda_away"],
                                                   (pred["real"] or {}).get("btts_yes"),
                                                   (pred["real"] or {}).get("btts_no"))
                    except Exception as e:
                        pred["btts"] = None
                    pred["ranked"] = _rank_selections(pred)
                    result = pred
                except Exception as e:
                    result = {"error": f"Prediction failed: {e}"}

    try:
        from corners_model import corner_predictions
        corner_rows, corner_info = corner_predictions(all_matches)
    except Exception as e:
        print(f"corners build failed: {e}")
        corner_rows, corner_info = [], {"ready": False, "line": 8.5, "message": f"Corners unavailable: {e}"}

    try:
        from ht_model import ht_predictions
        ht_rows, ht_info = ht_predictions(all_matches)
    except Exception as e:
        print(f"half-time build failed: {e}")
        ht_rows, ht_info = [], {"ready": False, "line": 1.5, "message": f"Half-time unavailable: {e}"}

    try:
        from btts_model import btts_predictions
        btts_rows = btts_predictions(all_matches)
        _btts_by_id = {r["event_id"]: r for r in btts_rows}
        for _m in all_matches:                       # so All Matches cards can show BTTS too
            _m["btts"] = _btts_by_id.get(_m["event_id"])
    except Exception as e:
        print(f"BTTS build failed: {e}")
        btts_rows = []

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
        results=results,
        rx=_results_summary(results),
        results_days=results_days,
        corner_rows=sorted(corner_rows, key=lambda r: r["prob"], reverse=True),   # highest probability first
        corner_info=corner_info,
        teams_by_league=teams_by_league,
        btts_rows=sorted(btts_rows, key=lambda r: r["prob"], reverse=True),   # highest probability first
        ht_rows=sorted(ht_rows, key=lambda r: r["prob"], reverse=True),   # highest probability first
        ht_info=ht_info,
    )


@app.route("/api/matches")
def api_matches():
    return jsonify(load_matches())


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
