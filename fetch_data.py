# fetch_data.py
import os
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta

from config import (
    BZZOIRO_API_KEY, BZZOIRO_BASE_URL, DATA_DIR,
    MATCHES_CSV, MATCHES_XG_CSV, XG_CACHE_CSV,
    XG_WORKERS,
)

HEADERS = {"Authorization": f"Token {BZZOIRO_API_KEY}"}


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


# Outcome of the most recent fetch_recent_matches() call, read by app.py's automation loop.
LAST_FETCH_INFO = {"new": 0, "quota_hit": False}


def _get(url, params=None):
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if r.status_code == 401:
            print("ERROR: Invalid Bzzoiro API key.")
            return None
        if r.status_code == 402:
            print("ERROR: Paid add-on required.")
            return None
        if r.status_code != 200:
            print(f"HTTP {r.status_code} for {url}")
            return None
        return r.json()
    except Exception as e:
        print(f"Request failed: {e}")
        return None


# ----------------------------------------------------------------------
# Leagues & seasons
# ----------------------------------------------------------------------
def get_leagues(country=None):
    url = f"{BZZOIRO_BASE_URL}/leagues/"
    out, offset, limit = [], 0, 100
    while True:
        data = _get(url, {"limit": limit, "offset": offset})
        if not data:
            break
        results = data.get("results", [])
        if not results:
            break
        out.extend(results)
        if not data.get("next"):
            break
        offset += limit
    if country:
        out = [lg for lg in out
               if lg.get("country", "").lower() == country.lower()]
    return out


def get_current_season(league_id):
    data = _get(f"{BZZOIRO_BASE_URL}/leagues/{league_id}/season/")
    if not data:
        return None, None, None
    season = data.get("season") or {}
    return season.get("id"), season.get("start_date"), season.get("end_date")


def get_all_seasons(league_id):
    data = _get(f"{BZZOIRO_BASE_URL}/leagues/{league_id}/seasons/")
    if not data:
        return []
    seasons = data.get("seasons", [])
    return sorted(seasons, key=lambda s: s.get("start_date", ""), reverse=True)


# ----------------------------------------------------------------------
# Events
# ----------------------------------------------------------------------
def fetch_matches(league_id, season_id):
    url = f"{BZZOIRO_BASE_URL}/events/"
    rows, offset, limit = [], 0, 200

    while True:
        params = {
            "league_id": league_id,
            "season_id": season_id,
            "limit": limit,
            "offset": offset,
        }
        data = _get(url, params)
        if not data:
            break

        results = data.get("results", [])
        if not results:
            break

        for m in results:
            if m.get("status") != "finished":
                continue
            if m.get("home_score") is None or m.get("away_score") is None:
                continue

            rows.append({
                "event_id":     m.get("id"),
                "date":         m.get("event_date"),
                "league_id":    m.get("league_id"),
                "season_id":    m.get("season_id"),
                "home_team":    m.get("home_team"),
                "home_team_id": m.get("home_team_id"),
                "away_team":    m.get("away_team"),
                "away_team_id": m.get("away_team_id"),
                "home_goals":   m.get("home_score"),
                "away_goals":   m.get("away_score"),
                "total_goals":  m.get("home_score") + m.get("away_score"),
            })

        if not data.get("next"):
            break
        offset += limit

    return rows


def fetch_two_seasons(league_id):
    ensure_data_dir()

    season_id, start_date, end_date = get_current_season(league_id)
    if not season_id:
        print(f"  Could not resolve current season for league {league_id}.")
        return pd.DataFrame()

    print(f"  Current season: id={season_id} ({start_date} -> {end_date})")
    current = fetch_matches(league_id, season_id)
    print(f"    {len(current)} finished matches")

    prev = []
    seasons = get_all_seasons(league_id)
    prev_season = None
    for s in seasons:
        if s.get("id") != season_id and s.get("start_date", "") < start_date:
            prev_season = s
            break

    if prev_season:
        print(f"  Previous season: id={prev_season['id']} "
              f"({prev_season.get('start_date')} -> {prev_season.get('end_date')})")
        prev = fetch_matches(league_id, prev_season["id"])
        print(f"    {len(prev)} finished matches")

    df = pd.DataFrame(current + prev)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True, format="ISO8601")
        df = df.sort_values("date").reset_index(drop=True)
    return df


# ----------------------------------------------------------------------
# Lightweight recent fetch (for hourly updates)
# ----------------------------------------------------------------------
def fetch_recent_matches(leagues, days_back=3):
    """
    Fetch matches from the last N days that we don't already have.
    Appends them to matches.csv and matches_with_xg.csv (using goals as xG fallback).
    """
    LAST_FETCH_INFO.update(new=0, quota_hit=False)
    if not os.path.exists(MATCHES_CSV):
        return

    existing = pd.read_csv(MATCHES_CSV)
    existing_ids = set(existing["event_id"].astype(int).tolist())

    now = datetime.now(timezone.utc)
    date_from = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")
    date_to = now.strftime("%Y-%m-%d")

    new_rows = []
    quota_hit = False
    complete = True          # False if any request failed (quota 429, network, HTTP error)
    for lid in leagues:
        offset, limit = 0, 200
        while True:
            try:
                r = requests.get(
                    f"{BZZOIRO_BASE_URL}/events/",
                    headers=HEADERS,
                    params={
                        "league_id": lid,
                        "date_from": date_from,
                        "date_to": date_to,
                        "status": "finished",
                        "limit": limit,
                        "offset": offset,
                    },
                    timeout=15,
                )
            except Exception:
                complete = False
                break

            if r.status_code != 200:
                complete = False
                quota_hit = quota_hit or r.status_code == 429
                break

            data = r.json()
            res = data.get("results", [])
            if not res:
                break

            for m in res:
                eid = m.get("id")
                if eid in existing_ids:
                    continue
                if m.get("home_score") is None or m.get("away_score") is None:
                    continue
                if not m.get("event_date"):
                    print(f"  [FETCH RECENT] skipping event {eid}: API returned no date (will retry)")
                    continue

                new_rows.append({
                    "event_id":     eid,
                    "date":         m.get("event_date"),
                    "league_id":    m.get("league_id"),
                    "season_id":    m.get("season_id"),
                    "home_team":    m.get("home_team"),
                    "home_team_id": m.get("home_team_id"),
                    "away_team":    m.get("away_team"),
                    "away_team_id": m.get("away_team_id"),
                    "home_goals":   m.get("home_score"),
                    "away_goals":   m.get("away_score"),
                    "total_goals":  m.get("home_score") + m.get("away_score"),
                })

            if not data.get("next"):
                break
            offset += limit

        if quota_hit:
            LAST_FETCH_INFO["quota_hit"] = True
            print("  [FETCH RECENT] API quota used up (HTTP 429): stopping early.")
            break

    if new_rows:
        LAST_FETCH_INFO["new"] = len(new_rows)
        print(f"  [FETCH RECENT] Found {len(new_rows)} new finished matches.")
        new_df = pd.DataFrame(new_rows)
        
        # 1. Update matches.csv
        combined = pd.concat([existing, new_df], ignore_index=True)
        # format="ISO8601": the CSV holds "2026-10-16 19:00:00+00:00" but the API sends
        # "2026-10-17T19:00:00Z". Without it pandas coerces the new rows to NaT (blank dates).
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce", utc=True, format="ISO8601")
        combined = combined.sort_values("date").reset_index(drop=True)
        combined.to_csv(MATCHES_CSV, index=False)

        # 2. Update matches_with_xg.csv (use goals as xG proxy to keep it fast)
        if os.path.exists(MATCHES_XG_CSV):
            xg_df = pd.read_csv(MATCHES_XG_CSV)
            new_df_xg = new_df.copy()
            new_df_xg["home_xg"] = new_df_xg["home_goals"]
            new_df_xg["away_xg"] = new_df_xg["away_goals"]
            
            xg_combined = pd.concat([xg_df, new_df_xg], ignore_index=True)
            xg_combined["date"] = pd.to_datetime(xg_combined["date"], errors="coerce", utc=True, format="ISO8601")
            xg_combined = xg_combined.sort_values("date").reset_index(drop=True)
            xg_combined.to_csv(MATCHES_XG_CSV, index=False)

    if not complete:
        print("  [FETCH RECENT] Incomplete: some requests failed, so this will be retried later.")
    return complete


# ----------------------------------------------------------------------
# Real xG enrichment
# ----------------------------------------------------------------------
def fetch_match_xg(event_id):
    url = f"{BZZOIRO_BASE_URL}/events/{event_id}/stats/"
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
    except Exception:
        return None
    if r.status_code != 200:
        return None

    try:
        data = r.json()
    except Exception:
        return None

    stats = data.get("stats") or {}
    home_block = stats.get("home") or {}
    away_block = stats.get("away") or {}

    hxg = home_block.get("expected_goals")
    axg = away_block.get("expected_goals")

    if hxg is None or axg is None:
        return None

    try:
        hxg = float(hxg)
        axg = float(axg)
    except (TypeError, ValueError):
        return None

    if pd.isna(hxg) or pd.isna(axg):
        return None

    return {"home_xg": hxg, "away_xg": axg}


def _load_xg_cache():
    if not os.path.exists(XG_CACHE_CSV):
        return {}
    try:
        cache = pd.read_csv(XG_CACHE_CSV)
    except Exception:
        return {}

    out = {}
    for _, r in cache.iterrows():
        try:
            eid = int(r["event_id"])
        except (TypeError, ValueError):
            continue
        hxg, axg = r["home_xg"], r["away_xg"]
        if pd.isna(hxg) or pd.isna(axg):
            continue
        out[eid] = (float(hxg), float(axg))
    return out


def _save_xg_cache(cache):
    ensure_data_dir()
    rows = [{"event_id": eid, "home_xg": h, "away_xg": a}
            for eid, (h, a) in cache.items()
            if h is not None and a is not None]
    pd.DataFrame(rows).to_csv(XG_CACHE_CSV, index=False)


def enrich_with_xg(df, progress=True, save_every=100):
    cache = _load_xg_cache()
    total = len(df)

    to_fetch = [int(r["event_id"]) for _, r in df.iterrows()
                if int(r["event_id"]) not in cache]

    print(f"  Cache hits:     {total - len(to_fetch)} / {total}")
    print(f"  Need to fetch:  {len(to_fetch)} events")
    print(f"  Workers:        {XG_WORKERS}\n")

    if to_fetch:
        completed = 0
        fetched_ok = 0

        with ThreadPoolExecutor(max_workers=XG_WORKERS) as pool:
            futures = {pool.submit(fetch_match_xg, eid): eid
                       for eid in to_fetch}

            for fut in as_completed(futures):
                eid = futures[fut]
                try:
                    xg = fut.result()
                except Exception:
                    xg = None

                if xg:
                    cache[eid] = (xg["home_xg"], xg["away_xg"])
                    fetched_ok += 1

                completed += 1
                if progress and completed % 100 == 0:
                    pct = completed / len(to_fetch) * 100
                    print(f"  {completed}/{len(to_fetch)} ({pct:.0f}%)  "
                          f"ok={fetched_ok}")

                if save_every and completed % save_every == 0:
                    _save_xg_cache(cache)

        _save_xg_cache(cache)

    home_xg_col, away_xg_col = [], []
    fallbacks = 0
    for _, row in df.iterrows():
        eid = int(row["event_id"])
        hxg, axg = cache.get(eid, (None, None))

        if hxg is None or pd.isna(hxg) or axg is None or pd.isna(axg):
            hxg = row["home_goals"]
            axg = row["away_goals"]
            fallbacks += 1

        home_xg_col.append(hxg)
        away_xg_col.append(axg)

    df["home_xg"] = home_xg_col
    df["away_xg"] = away_xg_col
    df.to_csv(MATCHES_XG_CSV, index=False)

    print(f"\n  Enrichment complete.")
    print(f"  Missing xG (fell back to goals): {fallbacks}")
    print(f"  Saved to {MATCHES_XG_CSV}")
    return df


# ----------------------------------------------------------------------
# Budgeted xG upgrade (runs from app.py every day)
# ----------------------------------------------------------------------
# Matches added by the daily update, and matches whose xG request failed during a big download, carry
# goals as a stand-in for xG. upgrade_xg() replaces those with real xG a few hundred at a time, NEWEST FIRST
# (recent matches carry the most weight in the model), and never spends more than `budget` requests.
XG_MISSING_CSV = os.path.join(DATA_DIR, "xg_missing.csv")      # matches the API has no xG for (yet)
XG_GIVE_UP_TRIES = 3                                           # after this many empty answers ...
XG_GIVE_UP_AGE_DAYS = 7                                        # ... on a match older than this, stop asking


def fetch_match_xg_status(event_id):
    """
    Like fetch_match_xg, but says WHY there is no result, so a quota failure is never mistaken for
    "this match has no xG":  ("ok", {...}) | ("none", None) | ("quota", None) | ("error", None)
    """
    url = f"{BZZOIRO_BASE_URL}/events/{event_id}/stats/"
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
    except Exception:
        return "error", None
    if r.status_code == 429:
        return "quota", None
    if r.status_code in (404, 204):
        return "none", None
    if r.status_code != 200:
        return "error", None
    try:
        data = r.json()
    except Exception:
        return "error", None
    stats = (data or {}).get("stats") or {}
    hxg = (stats.get("home") or {}).get("expected_goals")
    axg = (stats.get("away") or {}).get("expected_goals")
    try:
        hxg, axg = float(hxg), float(axg)
    except (TypeError, ValueError):
        return "none", None
    if pd.isna(hxg) or pd.isna(axg):
        return "none", None
    out = {"home_xg": hxg, "away_xg": axg}
    try:                                   # the same response carries shots: keep them for the Shots tab (no extra request)
        from shots_data import parse_event_stats
        sh = parse_event_stats(data)
        if sh:
            out["home_shots"], out["away_shots"] = sh
    except Exception:
        pass
    return "ok", out


def _load_xg_missing():
    if not os.path.exists(XG_MISSING_CSV):
        return {}
    try:
        m = pd.read_csv(XG_MISSING_CSV)
        return {int(r.event_id): (int(r.tries), str(r.last_try)) for r in m.itertuples()}
    except Exception:
        return {}


def _save_xg_missing(missing):
    ensure_data_dir()
    pd.DataFrame([{"event_id": e, "tries": t, "last_try": d} for e, (t, d) in sorted(missing.items())],
                 columns=["event_id", "tries", "last_try"]).to_csv(XG_MISSING_CSV, index=False)


def write_proxy_xg(df):
    """First-run helper: write matches_with_xg.csv using goals as xG (no API requests), then apply any cached real xG."""
    ensure_data_dir()
    out = df.copy()
    out["home_xg"] = out["home_goals"]
    out["away_xg"] = out["away_goals"]
    cache = _load_xg_cache()
    if cache:
        for col, i in (("home_xg", 0), ("away_xg", 1)):
            out[col] = [cache[int(e)][i] if int(e) in cache else v for e, v in zip(out["event_id"], out[col])]
    out.to_csv(MATCHES_XG_CSV, index=False)
    return out


def _backlog(df, cache, missing, today, now):
    """Event ids that still lack real xG and are due another try, newest match first."""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce", utc=True, format="ISO8601")
    d = d[~d["event_id"].astype(int).isin(cache.keys())].sort_values("date", ascending=False)
    due = []
    for e, dt in zip(d["event_id"].astype(int), d["date"]):
        tries, last = missing.get(e, (0, ""))
        age_days = (now - dt).days if pd.notna(dt) else 999
        if tries >= XG_GIVE_UP_TRIES and age_days > XG_GIVE_UP_AGE_DAYS:
            continue                                  # the API has no xG for it: stop spending requests
        if last == today:
            continue                                  # already asked today
        due.append(e)
    return due


def xg_coverage():
    """Local-only (no API) coverage numbers for status displays."""
    out = {"matches": 0, "real": 0, "pct": 0, "backlog": 0, "gave_up": 0}
    if not os.path.exists(MATCHES_XG_CSV):
        return out
    try:
        df = pd.read_csv(MATCHES_XG_CSV, usecols=["event_id", "date"])
        cache, missing = _load_xg_cache(), _load_xg_missing()
        ids = df["event_id"].astype(int)
        out["matches"] = len(df)
        out["real"] = int(ids.isin(cache.keys()).sum())
        out["pct"] = round(100 * out["real"] / max(len(df), 1))
        now = datetime.now(timezone.utc)
        out["backlog"] = len(_backlog(df, cache, missing, "never", now))
        out["gave_up"] = int(sum(1 for e in ids if e not in cache and missing.get(e, (0, ""))[0] >= XG_GIVE_UP_TRIES))
    except Exception:
        pass
    return out


def upgrade_xg(budget=400, workers=None, verbose=True):
    """
    Replace goals-as-xG with real xG for up to `budget` matches, newest first.
    Quota-safe: stops submitting as soon as the API answers 429, and a 429 or a network error is never
    recorded as "no xG". Matches the API answers empty for are remembered in data/xg_missing.csv and
    retried once per day until XG_GIVE_UP_TRIES (older than XG_GIVE_UP_AGE_DAYS days).
    Returns {"backlog", "tried", "upgraded", "no_xg", "errors", "quota_hit", "left"}.
    """
    import threading
    stats = {"backlog": 0, "tried": 0, "upgraded": 0, "no_xg": 0, "errors": 0, "quota_hit": False, "left": 0}
    if not os.path.exists(MATCHES_XG_CSV):
        return {**stats, "error": "matches_with_xg.csv not found"}

    df = pd.read_csv(MATCHES_XG_CSV)
    cache, missing = _load_xg_cache(), _load_xg_missing()
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")

    due = _backlog(df, cache, missing, today, now)
    stats["backlog"] = len(due)
    todo = due[:max(int(budget), 0)]
    if verbose:
        print(f"  [XG UPGRADE] backlog {len(due)} | budget {budget} | asking for {len(todo)} (newest first)")

    stop = threading.Event()

    def work(eid):
        if stop.is_set():
            return eid, "skipped", None
        status, xg = fetch_match_xg_status(eid)
        if status == "quota":
            stop.set()
        return eid, status, xg

    shots_found = []
    if todo:
        with ThreadPoolExecutor(max_workers=workers or XG_WORKERS) as pool:
            for eid, status, xg in pool.map(work, todo):
                if status == "skipped":
                    continue
                stats["tried"] += 1
                if status == "ok":
                    cache[eid] = (xg["home_xg"], xg["away_xg"])
                    if "home_shots" in xg:
                        shots_found.append((eid, xg["home_shots"], xg["away_shots"]))
                    missing.pop(eid, None)
                    stats["upgraded"] += 1
                elif status == "none":
                    tries, _ = missing.get(eid, (0, ""))
                    missing[eid] = (tries + 1, today)
                    stats["no_xg"] += 1
                elif status == "quota":
                    stats["quota_hit"] = True
                else:
                    stats["errors"] += 1
        if stats["quota_hit"]:
            if verbose:
                print("  [XG UPGRADE] API quota used up (HTTP 429): stopped early, will continue tomorrow.")
        _save_xg_cache(cache)
        _save_xg_missing(missing)
        if shots_found:
            try:
                from shots_data import record_many
                record_many(shots_found)
            except Exception as e:
                print(f"  [XG UPGRADE] could not save shots: {e}")

    # keep matches_with_xg.csv consistent with the cache (also fixes rows upgraded in earlier runs)
    ids = df["event_id"].astype(int)
    for col, i in (("home_xg", 0), ("away_xg", 1)):
        df[col] = [cache[int(e)][i] if int(e) in cache else v for e, v in zip(ids, df[col])]
    df.to_csv(MATCHES_XG_CSV, index=False)

    stats["left"] = max(len(due) - stats["upgraded"], 0)       # still lacking real xG (empty ones are retried until given up)
    if verbose:
        print(f"  [XG UPGRADE] upgraded {stats['upgraded']} | no xG yet {stats['no_xg']} | errors {stats['errors']} "
              f"| still to do {stats['left']}")
    return stats


# ----------------------------------------------------------------------
# Team lookup
# ----------------------------------------------------------------------
def find_team(team_name, league_id=None):
    params = {"name": team_name}
    if league_id:
        params["league_id"] = league_id
    data = _get(f"{BZZOIRO_BASE_URL}/teams/", params)
    if not data:
        return None, None
    results = data.get("results", []) if isinstance(data, dict) else data
    if not results:
        return None, None
    return results[0].get("id"), results[0].get("name")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == "leagues":
        country = sys.argv[2] if len(sys.argv) > 2 else None
        for lg in get_leagues(country):
            cs = lg.get("current_season") or {}
            print(f"{lg['id']:>5}  {lg['country']:<15}  {lg['name']:<40}  "
                  f"season={cs.get('id')} ({cs.get('year')})")
    else:
        print("Usage: python fetch_data.py leagues [country]")
