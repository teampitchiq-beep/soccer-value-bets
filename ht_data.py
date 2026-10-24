# ht_data.py
"""
Downloads and stores HALF-TIME goals (home / away) for finished matches.

    python ht_data.py              # fetch half-time scores for ALL finished matches that lack them
    python ht_data.py --limit 300  # only the 300 most recent missing ones
    python ht_data.py --status     # what has been stored so far

Source: Bzzoiro  /events/{id}/  ->  home_score_ht / away_score_ht
Saved to data/ht_goals.csv (resumable: re-run any time to continue or pick up new matches).

To save API requests it first tries the league LIST endpoint (200 matches per request). If the
list rows carry the half-time score, nearly everything arrives in ~40 requests. If they do not,
it falls back to one request per match. It stops immediately when the daily quota is used up.

The app also calls update_ht() once a day in the background.   Needs internet.
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd
import requests

from config import BZZOIRO_API_KEY, BZZOIRO_BASE_URL, MATCHES_XG_CSV, LEAGUES_TO_FETCH

HEADERS = {"Authorization": f"Token {BZZOIRO_API_KEY}"}
HT_CSV = "data/ht_goals.csv"
COLUMNS = ["event_id", "home_ht", "away_ht", "fetched_at"]
WORKERS = 4
RETRIES_ON_429 = 3
RETRY_RECENT_DAYS = 3
RETRY_AFTER_HOURS = 2
BULK_MIN_TODO = 30          # use the cheap list endpoint when at least this many are missing


class QuotaExhausted(Exception):
    """The account's daily request quota is used up (resets at 00:00 UTC)."""


def _get(url, params=None, timeout=25):
    """GET with back-off on short rate limits; raises QuotaExhausted on the daily quota."""
    r = None
    for attempt in range(RETRIES_ON_429 + 1):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=timeout)
        except Exception:
            return None
        if r.status_code != 429:
            return r
        try:
            code = (r.json() or {}).get("code", "")
        except Exception:
            code = ""
        if code == "taster_exhausted":
            raise QuotaExhausted()
        time.sleep(min(2 ** (attempt + 1), 15))          # short burst limit: back off and retry
    return r


def _int(v):
    try:
        f = float(v)
        return None if pd.isna(f) else int(round(f))
    except (TypeError, ValueError):
        return None


def parse_ht(d):
    """(home_ht, away_ht) from an event dict, or None when the half-time score is not there."""
    h, a = _int(d.get("home_score_ht")), _int(d.get("away_score_ht"))
    return (h, a) if h is not None and a is not None else None


def fetch_event_ht(event_id):
    """-> (state, (home, away) | None, status); state is ok | missing | error.
    Raises QuotaExhausted when the daily quota is gone."""
    r = _get(f"{BZZOIRO_BASE_URL}/events/{event_id}/")
    if r is None:
        return "error", None, None
    if r.status_code == 404:
        return "missing", None, None
    if r.status_code != 200:
        return "error", None, None
    try:
        d = r.json() or {}
    except Exception:
        return "error", None, None
    ht = parse_ht(d)
    return ("ok", ht, d.get("status")) if ht else ("missing", None, d.get("status"))


# ---------------------------------------------------------------- storage
def load_ht():
    if not os.path.exists(HT_CSV):
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(HT_CSV)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[COLUMNS].drop_duplicates("event_id", keep="last")


def _save(df):
    os.makedirs("data", exist_ok=True)
    df[COLUMNS].to_csv(HT_CSV, index=False)


def _finished_matches():
    m = pd.read_csv(MATCHES_XG_CSV, usecols=["event_id", "date"])
    m["date"] = pd.to_datetime(m["date"], errors="coerce", utc=True, format="ISO8601")
    return m.dropna(subset=["date"]).sort_values("date", ascending=False)      # newest first


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
        if pd.notna(row["home_ht"]):
            continue
        if (now - r.date).days >= RETRY_RECENT_DAYS:
            continue                                   # old and still missing: give up
        fa = pd.to_datetime(row["fetched_at"], errors="coerce", utc=True)
        if pd.isna(fa) or (now - fa).total_seconds() >= RETRY_AFTER_HOURS * 3600:
            todo.append(eid)
    return todo


# ---------------------------------------------------------------- cheap bulk pass
def _bulk_pass(needed, date_from, date_to, verbose=True):
    """Half-time scores from the league LIST endpoint. Returns {event_id: (h, a)}, or None if
    the list rows do not carry half-time scores (then the caller falls back to per-match calls)."""
    found, probed = {}, False
    for lid in LEAGUES_TO_FETCH:
        offset = 0
        while True:
            r = _get(f"{BZZOIRO_BASE_URL}/events/", params={
                "league_id": lid, "status": "finished", "date_from": date_from,
                "date_to": date_to, "limit": 200, "offset": offset})
            if r is None or r.status_code != 200:
                break
            data = r.json() or {}
            results = data.get("results") or []
            if not results:
                break
            if not probed:
                probed = True
                if "home_score_ht" not in results[0]:
                    if verbose:
                        print("  list endpoint has no half-time scores: using one request per match")
                    return None
            for m in results:
                eid = m.get("id")
                if eid in needed:
                    ht = parse_ht(m)
                    if ht:
                        found[eid] = ht
            if not data.get("next"):
                break
            offset += 200
    return found if probed else None


# ---------------------------------------------------------------- update
def update_ht(max_new=None, verbose=True):
    """Fetch half-time goals for finished matches that lack them. Returns number added."""
    matches = _finished_matches()
    saved = load_ht()
    todo = _todo(matches, saved)
    if verbose:
        print(f"half-time: {len(saved)} rows stored, {len(todo)} to fetch")
    if not todo:
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    added = 0

    def flush(rows):
        nonlocal saved
        if not rows:
            return
        new = pd.DataFrame(rows, columns=COLUMNS)
        saved = pd.concat([saved[~saved["event_id"].isin(new["event_id"])], new], ignore_index=True)
        _save(saved)

    try:
        # 1) cheap: whole league pages
        if len(todo) >= BULK_MIN_TODO:
            dates = matches[matches["event_id"].isin(todo)]["date"]
            found = _bulk_pass(set(todo), dates.min().strftime("%Y-%m-%d"),
                               dates.max().strftime("%Y-%m-%d"), verbose)
            if found:
                flush([(eid, h, a, stamp) for eid, (h, a) in found.items()])
                added += len(found)
                todo = [e for e in todo if e not in found]
                if verbose:
                    print(f"  bulk pass stored {len(found)} matches")

        # 2) one request per remaining match (capped)
        if max_new:
            todo = todo[:max_new]
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for i in range(0, len(todo), 40):
                chunk = todo[i:i + 40]
                batch = []
                try:
                    for eid, (state, val, _st) in zip(chunk, pool.map(fetch_event_ht, chunk)):
                        if state == "error":
                            continue
                        if state == "ok":
                            batch.append((eid, val[0], val[1], stamp))
                            added += 1
                        else:
                            batch.append((eid, None, None, stamp))
                finally:
                    flush(batch)                       # keep whatever arrived, even if the quota ran out
                if verbose and (i + 40) % 200 == 0:
                    print(f"  {i + 40}/{len(todo)}")
    except QuotaExhausted:
        print("half-time: API daily quota used up — stopped, will resume after 00:00 UTC.")
    if verbose:
        print(f"half-time: added {added} matches")
    return added


def status():
    saved = load_ht()
    ok = saved["home_ht"].notna()
    total = len(_finished_matches())
    print(f"Finished matches in your data: {total}")
    print(f"With half-time score stored:   {int(ok.sum())}")
    print(f"Marked missing:                {int((~ok).sum())}")
    print(f"Not fetched yet:               {max(total - len(saved), 0)}")
    if ok.any():
        tot = saved.loc[ok, "home_ht"] + saved.loc[ok, "away_ht"]
        print(f"Average HT goals per match:    {tot.mean():.2f}")
        print(f"Share with 2+ HT goals (Over 1.5): {(tot >= 2).mean()*100:.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    status() if a.status else (update_ht(max_new=a.limit), status())
