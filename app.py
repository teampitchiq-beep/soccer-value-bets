# app.py
"""
Single-entry daily runner with auto-settle and hourly recent fetcher.
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
RESULTS_DAYS_DEFAULT = 7

AUTO_SETTLE_INTERVAL_SECONDS = 15 * 60
RECENT_FETCH_INTERVAL_SECONDS = 60 * 60  # 1 hour

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

    need_full_fetch = should_refresh() and not force_no_refresh
    
    if need_full_fetch:
        if not AUTO_FETCH:
            print("Stale data but AUTO_FETCH=False. Skipping full fetch.")
        else:
            df = fetch_all_leagues()
            if df is None:
                sys.exit(1)
            save_fetch_time()
    else:
        print(f"✓ Data is fresh (last fetch: {last_fetch_age_str()})")
        
        # Even if data is "fresh" (<24h), do a quick fetch to catch 
        # yesterday's matches that finished since the last run.
        print("  Checking for recently finished matches...")
        from fetch_data import fetch_recent_matches
        fetch_recent_matches(LEAGUES_TO_FETCH, days_back=3)

    if not os.path.exists(MATCHES_XG_CSV) or need_full_fetch:
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
# Results loader
# ----------------------------------------------------------------------
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

    bets_by_event = {}
    if os.path.exists("bets_log.csv"):
        try:
            bets = pd.read_csv("bets_log.csv")
            for _, b in bets.iterrows():
                try:
                    eid = int(b["event_id"])
                except (TypeError, ValueError):
                    continue
                bets_by_event[eid] = {
                    "pick":   str(b.get("pick", "")),
                    "result": str(b.get("result", "")).lower(),
                    "pnl":    float(b["pnl"]) if pd.notna(b.get("pnl")) else None,
                }
        except Exception:
            pass

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
            "date_label":  row["date"].strftime("%a %d %b"),
            "date_key":    row["date"].strftime("%Y-%m-%d"),
            "time_label":  row["date"].strftime("%H:%M"),
            "league_id":   int(row["league_id"]),
            "league_name": league_name(row["league_id"]),
            "home":        row["home_team"],
            "away":        row["away_team"],
            "home_goals":  hg,
            "away_goals":  ag,
            "total":       total,
            "over_hit":    actual_over,
            "score":       f"{hg}-{ag}",
            "bet":         bets_by_event.get(eid),
            "prediction":  prediction,
        })

    return results


# ----------------------------------------------------------------------
# Background threads
# ----------------------------------------------------------------------
def auto_settle_loop():
    while True:
        try:
            time.sleep(AUTO_SETTLE_INTERVAL_SECONDS)
            from tracker import auto_settle
            n, details = auto_settle(verbose=False)
            if n > 0:
                print(f"\n[AUTO-SETTLE] {n} bet(s) settled.")
                invalidate_matches_cache()
        except Exception as e:
            print(f"[AUTO-SETTLE] error: {e}")


def recent_fetch_loop():
    while True:
        try:
            time.sleep(RECENT_FETCH_INTERVAL_SECONDS)
            print("\n[RECENT FETCH] Checking for new finished matches...")
            from fetch_data import fetch_recent_matches
            fetch_recent_matches(LEAGUES_TO_FETCH, days_back=3)
            
            # Reload DF into memory so the UI updates
            global DF
            from model import load_data
            DF = load_data()
            print("[RECENT FETCH] Dataset reloaded.")
            invalidate_matches_cache()
        except Exception as e:
            print(f"[RECENT FETCH] error: {e}")


def start_background_threads():
    t1 = threading.Thread(target=auto_settle_loop, daemon=True)
    t1.start()
    print(f"✓ Auto-settle thread running (every "
          f"{AUTO_SETTLE_INTERVAL_SECONDS // 60} min)")

    t2 = threading.Thread(target=recent_fetch_loop, daemon=True)
    t2.start()
    print(f"✓ Recent match fetcher running (every "
          f"{RECENT_FETCH_INTERVAL_SECONDS // 60} min)")


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
        results=results,
        results_days=results_days,
    )


@app.route("/log_bet/<int:event_id>", methods=["POST"])
def log_bet(event_id):
    from tracker import add_bet_from_pick

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
    print(f"  bets_log.csv         : "
          f"{'exists' if os.path.exists('bets_log.csv') else 'missing'}")
    print(f"  predictions_history  : "
          f"{'exists' if os.path.exists('data/predictions_history.csv') else 'missing'}")
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

    try:
        from tracker import auto_settle
        auto_settle(verbose=False)
    except Exception as e:
        print(f"Startup auto-settle failed: {e}")

    start_background_threads()

    app.run(host="0.0.0.0", port=5000, debug=False)
