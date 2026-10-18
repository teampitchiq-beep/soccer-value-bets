# odds_logger.py
"""
Saves a snapshot of pre-match Over/Under 2.5 odds + the model's probability
for every upcoming fixture, so you can later backtest against REAL pre-kickoff
prices (Bzzoiro doesn't provide them for finished matches).

  python odds_logger.py                 # one snapshot now
  python odds_logger.py --watch         # snapshot every 60 min until Ctrl-C
  python odds_logger.py --watch --every 30 --days 1
  python odds_logger.py --status        # what's been logged so far

scan.py also calls log_snapshot() automatically whenever it runs, so simply
using the app / --scan logs odds too. Identical odds are not logged twice.

Termux tip for --watch: run `termux-wake-lock` first so Android doesn't
kill the process when the screen is off.

Offline use: if the API can't be reached, nothing is logged and nothing
breaks; --watch retries every few minutes until you're back online. Snapshots
simply have gaps, and the analysis uses the last one taken before each kickoff
(see --max-hours in odds_analysis.py to keep only near-kickoff snapshots).

Later: python odds_analysis.py   (uses these logs once matches have finished)
"""
import argparse
import os
import time
from datetime import datetime, timezone

import pandas as pd

LOG_CSV = "data/odds_log.csv"
COLUMNS = ["logged_at", "event_id", "kickoff", "hours_to_kickoff", "league_id",
           "home", "away", "lh", "la", "p_over", "over_odds", "under_odds", "model"]
MODEL_TAG = "ridge_v1"
RETRY_MIN = 10          # --watch retry interval when offline / failing


def _last_seen():
    """event_id -> (over_odds, under_odds) from the most recent logged snapshot."""
    if not os.path.exists(LOG_CSV):
        return {}
    try:
        old = pd.read_csv(LOG_CSV, usecols=["event_id", "over_odds", "under_odds"])
    except Exception:
        return {}
    last = old.drop_duplicates("event_id", keep="last")
    return {int(r.event_id): (r.over_odds, r.under_odds) for r in last.itertuples()}


def log_snapshot(matches, verbose=False):
    """
    matches: list of dicts as returned by scan.get_todays_matches()["matches"].
    Appends one row per fixture that has odds, hasn't kicked off, and whose
    odds changed since the last snapshot. Returns number of rows written.
    """
    now = datetime.now(timezone.utc)
    seen = _last_seen()
    rows = []
    for m in matches:
        over_o, under_o = m.get("over_odds"), m.get("under_odds")
        if not over_o or not under_o:
            continue
        kickoff = pd.to_datetime(m.get("date"), utc=True, errors="coerce")
        if pd.isna(kickoff) or kickoff <= now:
            continue                      # only genuine pre-kickoff prices
        eid = int(m["event_id"])
        if eid in seen and seen[eid] == (over_o, under_o):
            continue                      # nothing changed
        rows.append({
            "logged_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "event_id": eid,
            "kickoff": kickoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hours_to_kickoff": round((kickoff - now).total_seconds() / 3600, 2),
            "league_id": m.get("league_id"),
            "home": m.get("home"),
            "away": m.get("away"),
            "lh": m.get("lh"),
            "la": m.get("la"),
            "p_over": round(m["po"] / 100.0, 4),
            "over_odds": over_o,
            "under_odds": under_o,
            "model": MODEL_TAG,
        })
    if rows:
        os.makedirs("data", exist_ok=True)
        pd.DataFrame(rows, columns=COLUMNS).to_csv(
            LOG_CSV, mode="a", index=False, header=not os.path.exists(LOG_CSV))
    if verbose:
        print(f"  odds_logger: {len(rows)} new snapshot rows "
              f"({len(matches)} fixtures scanned)")
    return len(rows)


def run_once(days):
    """Returns (rows_logged, fixtures_scanned). fixtures_scanned == 0 usually means offline."""
    from scan import get_todays_matches      # imported lazily (scan imports this module)
    result = get_todays_matches(days=days)
    matches = result["matches"]
    if result["stats"]["total_upcoming"] == 0:
        print("  No fixtures returned - you may be offline or the API is unreachable. Nothing logged.")
        return 0, 0
    n = log_snapshot(matches, verbose=True)
    return n, len(matches)


def status():
    if not os.path.exists(LOG_CSV):
        print("No odds log yet. Run: python odds_logger.py")
        return
    log = pd.read_csv(LOG_CSV)
    log["kickoff"] = pd.to_datetime(log["kickoff"], utc=True, errors="coerce")
    ev = log.drop_duplicates("event_id", keep="last")
    now = pd.Timestamp.now(tz="UTC")
    past = ev[ev["kickoff"] < now]
    print(f"Snapshot rows:      {len(log)}")
    print(f"Distinct fixtures:  {len(ev)}  (kicked off already: {len(past)}, upcoming: {len(ev) - len(past)})")
    print(f"First / last log:   {log['logged_at'].min()}  ->  {log['logged_at'].max()}")
    if len(past):
        print(f"\nOf the {len(past)} fixtures already kicked off, results arrive after your next "
              f"data refresh.\nRun `python odds_analysis.py` once you have ~100+ finished fixtures "
              f"(ideally 300+).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="keep logging on a timer")
    ap.add_argument("--every", type=int, default=60, help="minutes between snapshots (watch mode)")
    ap.add_argument("--days", type=int, default=2, help="how many days ahead to scan")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()

    if a.status:
        status()
    elif a.watch:
        print(f"Logging every {a.every} min (retries every {min(a.every, RETRY_MIN)} min "
              f"when offline). Ctrl-C to stop.")
        while True:
            wait = a.every
            try:
                print(datetime.now().strftime("[%Y-%m-%d %H:%M]"))
                _, scanned = run_once(a.days)
                if scanned == 0:
                    wait = min(a.every, RETRY_MIN)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"  snapshot failed: {e}")
                wait = min(a.every, RETRY_MIN)
            time.sleep(wait * 60)
    else:
        run_once(a.days)
