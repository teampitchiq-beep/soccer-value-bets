# corners_data.py
"""
Downloads and stores corner counts (home / away) for finished matches.

    python corners_data.py             # fetch corners for ALL finished matches that lack them
    python corners_data.py --limit 300 # only the 300 most recent missing ones
    python corners_data.py --status    # what has been stored so far

Source: Bzzoiro  /events/{id}/stats/  ->  stats.home.corner_kicks / stats.away.corner_kicks
Saved to data/corners.csv (resumable: re-run any time to continue or to pick up new matches).
Matches without a stats block (e.g. qualifying rounds) are recorded as "missing" and only
retried while they are less than 3 days old.

The app also calls update_corners() in the background every hour.
Needs internet.
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd
import requests

from config import BZZOIRO_API_KEY, BZZOIRO_BASE_URL, MATCHES_XG_CSV

HEADERS = {"Authorization": f"Token {BZZOIRO_API_KEY}"}
CORNERS_CSV = "data/corners.csv"
COLUMNS = ["event_id", "home_corners", "away_corners", "fetched_at"]
WORKERS = 4                 # kept modest: Bzzoiro answers HTTP 429 when hit too fast
RETRIES_ON_429 = 3
RETRY_RECENT_DAYS = 3
RETRY_AFTER_HOURS = 2


def _value(v):
    """Stat values are plain numbers, or {"value": n, "total": ..} for some stats."""
    if isinstance(v, dict):
        v = v.get("value")
    try:
        f = float(v)
        return None if pd.isna(f) else int(round(f))
    except (TypeError, ValueError):
        return None


def fetch_event_corners(event_id):
    """-> ("ok", (home, away)) | ("missing", None) | ("error", None)"""
    r = None
    for attempt in range(RETRIES_ON_429 + 1):
        try:
            r = requests.get(f"{BZZOIRO_BASE_URL}/events/{event_id}/stats/", headers=HEADERS, timeout=20)
        except Exception:
            return "error", None
        if r.status_code != 429:
            break
        time.sleep(min(2 ** (attempt + 1), 15))        # rate limited: back off, then retry
    if r is None or r.status_code == 429:
        return "error", None                            # still limited: not stored, retried next run
    if r.status_code == 404:
        return "missing", None
    if r.status_code != 200:
        return "error", None
    try:
        stats = (r.json() or {}).get("stats") or {}
    except Exception:
        return "error", None
    home, away = stats.get("home") or {}, stats.get("away") or {}
    h = _value(home.get("corner_kicks", home.get("corners")))
    a = _value(away.get("corner_kicks", away.get("corners")))
    if h is None or a is None:
        return "missing", None
    return "ok", (h, a)


def load_corners():
    if not os.path.exists(CORNERS_CSV):
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(CORNERS_CSV)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[COLUMNS].drop_duplicates("event_id", keep="last")


def _save(df):
    os.makedirs("data", exist_ok=True)
    df[COLUMNS].to_csv(CORNERS_CSV, index=False)


def _finished_matches():
    m = pd.read_csv(MATCHES_XG_CSV, usecols=["event_id", "date"])
    m["date"] = pd.to_datetime(m["date"], errors="coerce", utc=True, format="ISO8601")
    return m.dropna(subset=["date"]).sort_values("date", ascending=False)   # newest first


def _todo(matches, saved):
    now = pd.Timestamp.now(tz="UTC")
    info = saved.set_index("event_id")
    todo = []
    for r in matches.itertuples():
        eid = int(r.event_id)
        if eid not in info.index:
            todo.append(eid)
            continue
        row = info.loc[eid]
        if pd.notna(row["home_corners"]):
            continue                                   # have it
        if (now - r.date).days >= RETRY_RECENT_DAYS:
            continue                                   # old and still missing: give up
        fa = pd.to_datetime(row["fetched_at"], errors="coerce", utc=True)
        if pd.isna(fa) or (now - fa).total_seconds() >= RETRY_AFTER_HOURS * 3600:
            todo.append(eid)                           # recent match: stats may have arrived since
    return todo


def update_corners(max_new=None, verbose=True):
    """Fetch corners for finished matches that lack them. Returns number of matches with corners added."""
    matches = _finished_matches()
    saved = load_corners()
    todo = _todo(matches, saved)
    if max_new:
        todo = todo[:max_new]
    if verbose:
        print(f"corners: {len(saved)} rows stored, {len(todo)} to fetch")
    if not todo:
        return 0

    added, batch = 0, []
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def flush(batch):
        nonlocal saved
        if not batch:
            return
        new = pd.DataFrame(batch, columns=COLUMNS)
        saved = pd.concat([saved[~saved["event_id"].isin(new["event_id"])], new], ignore_index=True)
        _save(saved)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for n, (eid, (status, val)) in enumerate(zip(todo, pool.map(fetch_event_corners, todo)), 1):
            if status == "error":
                continue                                # transient: not stored, retried next run
            if status == "ok":
                batch.append((eid, val[0], val[1], stamp))
                added += 1
            else:
                batch.append((eid, None, None, stamp))
            if len(batch) >= 200:
                flush(batch); batch = []
            if verbose and n % 200 == 0:
                print(f"  {n}/{len(todo)}")
    flush(batch)
    if verbose:
        print(f"corners: added {added} matches with corner data")
    return added


def status():
    saved = load_corners()
    ok = saved["home_corners"].notna()
    total = len(_finished_matches())
    print(f"Finished matches in your data: {total}")
    print(f"With corners stored:           {int(ok.sum())}")
    print(f"Marked missing (no stats):     {int((~ok).sum())}")
    print(f"Not fetched yet:               {max(total - len(saved), 0)}")
    if ok.any():
        tot = saved.loc[ok, "home_corners"] + saved.loc[ok, "away_corners"]
        print(f"Average corners per match:     home {saved.loc[ok,'home_corners'].mean():.1f}, "
              f"away {saved.loc[ok,'away_corners'].mean():.1f}, total {tot.mean():.1f}")
        print(f"Share of matches with 9+ corners (Over 8.5): {(tot >= 9).mean()*100:.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    status() if a.status else (update_corners(max_new=a.limit), status())
