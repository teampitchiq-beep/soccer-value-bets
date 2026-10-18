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
    if not os.path.exists(MATCHES_CSV):
        return

    existing = pd.read_csv(MATCHES_CSV)
    existing_ids = set(existing["event_id"].astype(int).tolist())

    now = datetime.now(timezone.utc)
    date_from = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")
    date_to = now.strftime("%Y-%m-%d")

    new_rows = []
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
                break

            if r.status_code != 200:
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

    if new_rows:
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
