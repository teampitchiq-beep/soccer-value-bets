# odds_probe.py
"""
Check / download Bzzoiro odds for FINISHED matches (for backtesting).

  python odds_probe.py              # sample 60 finished matches, report O/U 2.5 coverage
  python odds_probe.py --backfill   # fetch odds for all finished matches -> data/odds_history.csv
                                    # (resumable: re-run to continue / pick up new matches)
  python odds_probe.py --refresh    # re-fetch rows missing last_change_at (needed for timing
                                    # checks) + retry likely-failed rows from May 2026 onward

The API returns a dict of nulls for matches without odds, so "has O/U 2.5"
means BOTH over_25_goals and under_25_goals are non-null.
"""
import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from config import BZZOIRO_API_KEY, BZZOIRO_BASE_URL, MATCHES_XG_CSV

HEADERS = {"Authorization": f"Token {BZZOIRO_API_KEY}"}
OUT_CSV = "data/odds_history.csv"
COLUMNS = ["event_id", "over_25", "under_25", "last_update_at", "last_change_at", "raw"]
WORKERS = 6
DEFAULT_LIMIT = 500          # max requests per --backfill run
SAMPLE = 60
RETRY_FROM = "2026-04-25"   # odds coverage is ~100% from May 2026, so empties after this are suspect


# Quota guard: after this many 429s in a row everything stops. Without it, one run keeps
# sending thousands of requests that can only fail while the daily quota is used up.
STOP_AFTER_429 = 5
QUOTA_STOP = threading.Event()
_lock = threading.Lock()
_streak = [0]


def fetch(event_id):
    if QUOTA_STOP.is_set():
        return event_id, None, "skipped (quota)"
    try:
        r = requests.get(f"{BZZOIRO_BASE_URL}/events/{event_id}/odds/",
                         headers=HEADERS, timeout=20)
    except Exception as e:
        return event_id, None, f"request failed: {e}"
    with _lock:
        if r.status_code == 429:
            _streak[0] += 1
            if _streak[0] >= STOP_AFTER_429:
                QUOTA_STOP.set()
        else:
            _streak[0] = 0
    if r.status_code != 200:
        return event_id, None, f"HTTP {r.status_code}"
    try:
        return event_id, r.json(), None
    except Exception:
        return event_id, None, "bad JSON"


def real_odds(data):
    odds = (data or {}).get("odds") or {}
    return {k: v for k, v in odds.items() if v is not None}


def has_ou25(odds):
    return odds.get("over_25_goals") is not None and odds.get("under_25_goals") is not None


def make_row(eid, data):
    odds = real_odds(data)
    return {
        "event_id": eid,
        "over_25": odds.get("over_25_goals"),
        "under_25": odds.get("under_25_goals"),
        "last_update_at": (data or {}).get("last_update_at"),
        "last_change_at": (data or {}).get("last_change_at"),
        "raw": json.dumps(odds) if odds else "",
    }


def finished_events():
    df = pd.read_csv(MATCHES_XG_CSV)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    return df.dropna(subset=["date"]).sort_values("date")


def load_saved():
    if not os.path.exists(OUT_CSV):
        return pd.DataFrame(columns=COLUMNS)
    saved = pd.read_csv(OUT_CSV)
    for c in COLUMNS:
        if c not in saved.columns:
            saved[c] = None
    return saved[COLUMNS].drop_duplicates("event_id", keep="last")


def save_all(saved):
    os.makedirs("data", exist_ok=True)
    saved[COLUMNS].to_csv(OUT_CSV, index=False)


def fetch_many(event_ids):
    """Fetch in parallel. Returns (rows, n_failed). Failures are NOT returned as rows."""
    rows, failed, n = [], 0, 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(fetch, e) for e in event_ids]
        for fut in as_completed(futs):
            eid, data, err = fut.result()
            if err:
                failed += 1
            else:
                rows.append(make_row(eid, data))
            n += 1
            if n % 200 == 0 or n == len(event_ids):
                print(f"  {n}/{len(event_ids)}")
    if QUOTA_STOP.is_set():
        print("\nSTOPPED EARLY: the API's daily quota is used up (HTTP 429). It resets at "
              "00:00 UTC (03:00 Nairobi).\nRows fetched before it stopped were saved; "
              "re-run tomorrow to continue.")
    return rows, failed


def merge_rows(saved, rows):
    if not rows:
        return saved
    new = pd.DataFrame(rows, columns=COLUMNS)
    keep = saved[~saved["event_id"].isin(new["event_id"])]
    return pd.concat([keep, new], ignore_index=True)


def report(saved):
    ok = saved["over_25"].notna() & saved["under_25"].notna()
    print(f"\n{len(saved)} rows in {OUT_CSV}; {int(ok.sum())} have Over/Under 2.5 odds; "
          f"{int(saved['last_change_at'].notna().sum())} have last_change_at.")


def probe():
    df = finished_events()
    step = max(1, len(df) // SAMPLE)
    sample = df.iloc[::step].head(SAMPLE)
    print(f"Probing {len(sample)} finished matches "
          f"({sample['date'].min().date()} -> {sample['date'].max().date()})\n")
    ou, errors, by_month, by_league, example = 0, {}, {}, {}, None
    for _, r in sample.iterrows():
        eid, data, err = fetch(int(r["event_id"]))
        month, lg = r["date"].strftime("%Y-%m"), int(r["league_id"])
        by_month.setdefault(month, [0, 0]); by_month[month][1] += 1
        by_league.setdefault(lg, [0, 0]);   by_league[lg][1] += 1
        if err:
            errors[err] = errors.get(err, 0) + 1
            continue
        if has_ou25(real_odds(data)):
            ou += 1
            by_month[month][0] += 1; by_league[lg][0] += 1
            example = example or data
        time.sleep(0.05)
    print(f"Over/Under 2.5 (both sides): {ou}/{len(sample)}  ({ou/len(sample)*100:.0f}%)")
    if errors:
        print(f"Errors: {errors}")
    print("\nO/U 2.5 coverage by month (with / sampled):")
    for m in sorted(by_month):
        print(f"  {m}: {by_month[m][0]}/{by_month[m][1]}")
    print("\nO/U 2.5 coverage by league_id (with / sampled):")
    for l in sorted(by_league):
        print(f"  {l}: {by_league[l][0]}/{by_league[l][1]}")
    if example:
        print("\nExample response WITH O/U 2.5:")
        print(json.dumps(example, indent=2, default=str)[:1500])


def backfill(limit=DEFAULT_LIMIT):
    df = finished_events().sort_values("date", ascending=False)      # newest first: only recent matches have odds
    saved = load_saved()
    have = set(saved["event_id"].astype(int))
    todo = [int(e) for e in df["event_id"] if int(e) not in have]
    print(f"{len(saved)} already saved, {len(todo)} to fetch")
    if limit and len(todo) > limit:
        print(f"Fetching the newest {limit} only (--limit {limit}) so one run cannot use up the daily quota.\n"
              f"Raise it with --limit N if you have requests to spare.")
        todo = todo[:limit]
    if todo:
        rows, failed = fetch_many(todo)
        saved = merge_rows(saved, rows)
        save_all(saved)
        if failed:
            print(f"{failed} requests failed (not saved) - just re-run to retry them")
    report(saved)


def refresh():
    df = finished_events().set_index("event_id")
    saved = load_saved()
    has_odds = saved["over_25"].notna() & saved["under_25"].notna()
    need_timing = has_odds & saved["last_change_at"].isna()
    dates = saved["event_id"].map(df["date"])
    suspect_empty = (~has_odds) & (dates >= pd.Timestamp(RETRY_FROM, tz="UTC"))
    todo = [int(e) for e in saved.loc[need_timing | suspect_empty, "event_id"]]
    print(f"Refreshing {len(todo)} rows "
          f"({int(need_timing.sum())} missing timing, {int(suspect_empty.sum())} suspect empties)")
    if todo:
        rows, failed = fetch_many(todo)
        saved = merge_rows(saved, rows)
        save_all(saved)
        if failed:
            print(f"{failed} requests failed - re-run --refresh to retry")
    report(saved)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                    help="max requests for --backfill (default %(default)s; 0 = no limit)")
    a = ap.parse_args()
    if a.refresh:
        refresh()
    elif a.backfill:
        backfill(a.limit)
    else:
        probe()
