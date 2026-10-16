# app.py
"""
Single-entry daily runner with auto-settle.

Usage:
    python app.py                  # daily run: fetch if stale, serve UI
    python app.py --force-refresh  # wipe cache, refetch everything
    python app.py --no-refresh     # skip fetch, just serve
    python app.py --scan           # print value bets and exit
    python app.py --all-matches    # print every upcoming match and exit
    python app.py --status         # show data freshness and exit
    python app.py --auto-settle    # settle pending bets and exit
"""
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

AUTO_SETTLE_INTERVAL_SECONDS = 15 * 60   # every 15 minutes

app = Flask(__name__)
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


def run_daily_pipeline(force_no_refresh=False):
    global DF
    os.makedirs(DATA_DIR, exist_ok=True)

    if FORCE_REFRESH:
        banner("FORCE_REFRESH — wiping cached data")
        for f in (MATCHES_CSV, MATCHES_XG_CSV):
            if os.path.exists(f):
                os.remove(f)
                print(f"  Removed {f}")

    need_fetch = should_refresh() and not force_no_refresh
    if need_fetch:
        if not AUTO_FETCH:
            print("Stale data but AUTO_FETCH=False. Skipping fetch.")
        else:
            df = fetch_all_leagues()
            if df is None:
                sys.exit(1)
            save_fetch_time()
    else:
        print(f"✓ Data is fresh (last fetch: {last_fetch_age_str()})")

    if not os.path.exists(MATCHES_XG_CSV) or need_fetch:
        raw = pd.read_csv(MATCHES_CSV)
        enrich_xg(raw)
    else:
        print("✓ xG data already present — skipping enrichment")

    DF = load_dataset()

    banner("READY — Web UI is starting")
    print(f"  Matches : http://127.0.0.1:5000/")
    print(f"  Tracker : http://127.0.0.1:5000/tracker")
    print(f"  Stop    : Ctrl+C  or  pkill -f 'python app.py'")
    print("=" * 68 + "\n")


# ----------------------------------------------------------------------
# Match cache
# ----------------------------------------------------------------------
def load_matches(force=False, days=2):
    from scan import get_todays_matches

    now = datetime.now()
    if (not force and _MATCHES_CACHE["at"] is not None
            and (now - _MATCHES_CACHE["at"]).total_seconds() < CACHE_MINUTES * 60):
        return _MATCHES_CACHE["data"]

    try:
        data = get_todays_matches(days=days)
        _MATCHES_CACHE["at"] = now
        _MATCHES_CACHE["data"] = data
        return data
    except Exception as e:
        print(f"Failed to load matches: {e}")
        return {"matches": [], "value_bets": [], "stats": {}}


def invalidate_matches_cache():
    _MATCHES_CACHE["at"] = None
    _MATCHES_CACHE["data"] = None


# ----------------------------------------------------------------------
# Background auto-settle thread
# ----------------------------------------------------------------------
def auto_settle_loop():
    """Runs forever in a daemon thread. Settles every N minutes."""
    while True:
        try:
            time.sleep(AUTO_SETTLE_INTERVAL_SECONDS)
            from tracker import auto_settle
            n, details = auto_settle(verbose=False)
            if n > 0:
                print(f"\n[AUTO-SETTLE] {n} bet(s) settled:")
                for d in details:
                    icon = "🟢" if d["result"] == "win" else "🔴"
                    print(f"  {icon} {d['fixture'][:40]:<42} "
                          f"{d['score']:>5}  {d['result']:<5} {d['pnl']:+.2f}u")
                # Invalidate picks cache so the homepage reflects updates
                invalidate_matches_cache()
        except Exception as e:
            print(f"[AUTO-SETTLE] error: {e}")


def start_auto_settle_thread():
    t = threading.Thread(target=auto_settle_loop, daemon=True)
    t.start()
    print(f"✓ Auto-settle running (every "
          f"{AUTO_SETTLE_INTERVAL_SECONDS // 60} min)")


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    from model import full_prediction
    from fetch_data import find_team

    result = None
    home_input, away_input = "", ""
    league_id = DEFAULT_LEAGUE_ID

    force_refresh = request.args.get("refresh") == "1"
    data = load_matches(force=force_refresh) if DF is not None else \
           {"matches": [], "value_bets": [], "stats": {}}

    all_matches = data["matches"]
    value_bets = data["value_bets"]
    stats = data["stats"]
    last_updated = _MATCHES_CACHE["at"].strftime("%H:%M") \
        if _MATCHES_CACHE["at"] else "—"

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
            result = {"error": "Enter both team names."}
        else:
            hid, hname = find_team(home_input, league_id)
            aid, aname = find_team(away_input, league_id)
            if not hid or not aid:
                result = {"error": f"Team not found. "
                                   f"Home: {hname or 'x'} | Away: {aname or 'x'}"}
            else:
                try:
                    pred = full_prediction(DF, hid, aid, league_id)
                    pred["home_team"] = hname
                    pred["away_team"] = aname
                    pred["league_name"] = league_name(league_id)
                    result = pred
                except Exception as e:
                    result = {"error": f"Prediction failed: {e}"}

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
        value_bets=value_bets,
        stats=stats,
        picks_last_updated=last_updated,
        last_fetch=last_fetch_age_str(),
    )


@app.route("/log_bet/<int:event_id>", methods=["POST"])
def log_bet(event_id):
    """One-tap logging from the value bet card."""
    from tracker import add_bet_from_pick
    from scan import get_todays_matches

    # Get the pick data — from cache if available
    data = load_matches()
    pick = None
    for p in data.get("value_bets", []) + data.get("matches", []):
        if int(p.get("event_id", 0)) == event_id:
            pick = p
            break

    if not pick:
        return redirect(url_for("index", msg="pick_not_found"))

    stake = request.form.get("stake", "1.0")
    try:
        stake = float(stake)
    except ValueError:
        stake = 1.0

    ok, msg = add_bet_from_pick(pick, stake=stake)
    return redirect(url_for("index", msg=("logged" if ok else "duplicate")))


@app.route("/tracker")
def tracker_view():
    from tracker import load_bets, auto_pnl, auto_settle

    # Auto-settle on page load too — instant feedback
    try:
        auto_settle(verbose=False)
    except Exception:
        pass

    try:
        df = load_bets()

        if len(df) == 0:
            return render_template(
                "tracker.html",
                summary={
                    "total": 0, "settled": 0, "pending": 0,
                    "wins": 0, "losses": 0,
                    "total_stake": 0, "total_pnl": 0,
                    "roi": 0, "win_rate": 0, "avg_odds": 0,
                },
                settled=[], pending=[],
            )

        df["pnl"] = df.apply(
            lambda r: auto_pnl(r) if pd.isna(r["pnl"]) else r["pnl"], axis=1
        )
        settled = df[df["result"].isin(["win", "loss", "void"])].copy()
        pending = df[~df["result"].isin(["win", "loss", "void"])].copy()

        if len(settled) > 0:
            wins = (settled["result"] == "win").sum()
            losses = (settled["result"] == "loss").sum()
            total_stake = settled["stake"].sum()
            total_pnl = settled["pnl"].sum()
            roi = total_pnl / total_stake * 100 if total_stake else 0
            win_rate = wins / (wins + losses) * 100 if (wins + losses) else 0
            avg_odds = settled["odds"].mean()
        else:
            wins = losses = 0
            total_stake = total_pnl = roi = win_rate = avg_odds = 0

        summary = {
            "total": len(df),
            "settled": len(settled),
            "pending": len(pending),
            "wins": int(wins),
            "losses": int(losses),
            "total_stake": round(float(total_stake), 2),
            "total_pnl": round(float(total_pnl), 2),
            "roi": round(float(roi), 1),
            "win_rate": round(float(win_rate), 1),
            "avg_odds": round(float(avg_odds), 2) if avg_odds else 0,
        }

        settled_rows = settled.sort_values("date", ascending=False).to_dict("records")
        pending_rows = pending.sort_values("date").to_dict("records")

        return render_template(
            "tracker.html",
            summary=summary,
            settled=settled_rows,
            pending=pending_rows,
        )
    except Exception as e:
        return f"<pre>Tracker error: {e}</pre>"


@app.route("/api/matches")
def api_matches():
    return jsonify(load_matches())


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
    print(f"  bets_log.csv         : "
          f"{'exists' if os.path.exists('bets_log.csv') else 'missing'}")
    print(f"  last fetch           : {last_fetch_age_str()}")

    if os.path.exists("bets_log.csv"):
        try:
            df = pd.read_csv("bets_log.csv")
            pending = df[~df["result"].isin(["win", "loss", "void"])]
            print(f"  bets logged          : {len(df)}")
            print(f"  pending             : {len(pending)}")
        except Exception:
            pass


def cmd_scan(show_all=False):
    from scan import get_todays_matches
    result = get_todays_matches(days=2)
    s = result["stats"]

    print("=" * 78)
    print("  DAILY SCAN")
    print("=" * 78)
    print(f"  Upcoming: {s['total_upcoming']}  |  "
          f"With history: {s['with_history']}  |  "
          f"Value bets: {s['value_bets_count']}\n")

    if show_all:
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
            if m["ev"] is not None:
                print(f"     EV: {m['ev']*100:+.1f}%")
            print()
    else:
        if not result["value_bets"]:
            print("No value bets found.")
            return
        for p in result["value_bets"]:
            icon = "🟢" if p["model_pick"] == "Over 2.5" else "🔵"
            print(f"{icon}  {p['date'][:16]}  [{p['league_name']}]")
            print(f"     {p['home']} vs {p['away']}")
            print(f"     PICK: {p['model_pick']}  @ {p['book_odds']:.2f}")
            print(f"     Edge: {p['ev']*100:+.1f}%")
            print(f"     Event ID: {p['event_id']}")


if __name__ == "__main__":
    if "--status" in sys.argv:
        cmd_status()
        sys.exit(0)

    if "--auto-settle" in sys.argv:
        from tracker import auto_settle
        n, details = auto_settle(verbose=True)
        print(f"Settled {n} bet(s).")
        sys.exit(0)

    if "--scan" in sys.argv:
        if os.path.exists(MATCHES_XG_CSV):
            from model import load_data
            try:
                DF = load_data()
            except Exception as e:
                print(f"Failed to load: {e}")
                sys.exit(1)
        cmd_scan(show_all=False)
        sys.exit(0)

    if "--all-matches" in sys.argv:
        if os.path.exists(MATCHES_XG_CSV):
            from model import load_data
            try:
                DF = load_data()
            except Exception as e:
                print(f"Failed to load: {e}")
                sys.exit(1)
        cmd_scan(show_all=True)
        sys.exit(0)

    force_no_refresh = "--no-refresh" in sys.argv

    try:
        run_daily_pipeline(force_no_refresh=force_no_refresh)
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(0)

    # Auto-settle on startup (catches yesterday's matches)
    try:
        from tracker import auto_settle
        auto_settle(verbose=False)
    except Exception as e:
        print(f"Startup auto-settle failed: {e}")

    # Start the background settle loop
    start_auto_settle_thread()

    app.run(host="0.0.0.0", port=5000, debug=False)
