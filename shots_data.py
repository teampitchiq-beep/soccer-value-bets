# shots_data.py
"""
Team shots per fixture, split by venue, for the Shots tab.

For a fixture  HOME vs AWAY  on a given day:
    home avg = average shots taken by HOME in its last 10 HOME matches before kickoff
    away avg = average shots taken by AWAY in its last 10 AWAY matches before kickoff
    avg      = (home avg + away avg) / 2          (the tab sorts by this, highest first)
    combined = home avg + away avg                (expected shots in the fixture, shown in the detail)

Only matches BEFORE the fixture's kickoff are used, so a past date shows what the numbers were
on the day, and an upcoming date uses everything played so far.

Data: shots come from the same Bzzoiro endpoint as xG (/events/{id}/stats/). They are stored in
data/shots_cache.csv (event_id, home_shots, away_shots). Nothing here touches the model's
xG/goals files. Matches the API has no shots for are remembered in data/shots_missing.csv and
are retried a few times, then given up (same idea as the xG upgrade).

CLI:
    python shots_data.py --probe EVENT_ID     show the raw shot-related fields the API returns for one match
    python shots_data.py --status             cache coverage
    python shots_data.py --all [N]            fetch shots for EVERY finished match that lacks them (optionally at most N requests)
    python shots_data.py --backfill [N]       same, default 300 requests
"""
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests

import config
from config import BZZOIRO_API_KEY, BZZOIRO_BASE_URL, DATA_DIR, MATCHES_CSV, league_name

XG_WORKERS = getattr(config, "XG_WORKERS", 8)
SHOTS_WINDOW = int(getattr(config, "SHOTS_WINDOW", 10))                # matches per team and venue
SHOTS_CACHE_CSV = os.path.join(DATA_DIR, "shots_cache.csv")
SHOTS_MISSING_CSV = os.path.join(DATA_DIR, "shots_missing.csv")
SHOTS_GIVE_UP_TRIES = 3                                                # empty answers before a match is given up ...
SHOTS_GIVE_UP_AGE_DAYS = 7                                             # ... once it is older than this
# Optional override if the API names the stat differently: SHOTS_STAT_KEYS = ["total_shots", ...]
EXTRA_KEYS = list(getattr(config, "SHOTS_STAT_KEYS", []) or [])
# Expected-shots model (same for/against engine as xG, fitted on shots): decay half-life in days and ridge penalty.
# Shots are ~9x more numerous than goals, so the ridge is stronger than the goals model's 10.
SHOTS_HALF_LIFE = float(getattr(config, "SHOTS_HALF_LIFE", 180.0))     # 180 beat 365 on real data (shots_tests.py)
SHOTS_RIDGE = float(getattr(config, "SHOTS_RIDGE", 40.0))
SHOTS_MIN_FIT_MATCHES = 200
# Blend shots-based expected goals into the goals model (lambda = goals^(1-w) * shots^w). ON by default at 0.2; set 0 in config.py to turn off.
# Tested on real data: w 0.2-0.3 gave a small gain (goals log loss -0.0017 to -0.0022), best w ~0.4, interval close to zero.
SHOTS_BLEND_WEIGHT = float(getattr(config, "SHOTS_BLEND_WEIGHT", 0.2))
# Recency weighting inside the last-N window of the Avg columns: the newest match has weight 1, the next DECAY, then DECAY^2 ...
# 1.0 = plain average (default). 0.9 = gentle, 0.8 = strong (the 10th-newest match then counts ~13% of the newest).
SHOTS_AVG_DECAY = float(getattr(config, "SHOTS_AVG_DECAY", 1.0))

HEADERS = {"Authorization": f"Token {BZZOIRO_API_KEY}"}
EAT = timezone(timedelta(hours=3))
_lock = threading.RLock()


# ----------------------------------------------------------------------
# Reading shots out of an API stats block
# ----------------------------------------------------------------------
_TOTAL_KEYS = EXTRA_KEYS + ["total_shots", "shots_total", "shots", "total_shot", "shots_all", "total_shots_all",
                            "team_total_shots", "shot_total", "total_scoring_att"]
_ON_KEYS = ["shots_on_target", "shots_on_goal", "on_target_shots", "shots_ontarget", "on_target"]
_OFF_KEYS = ["shots_off_target", "shots_off_goal", "off_target_shots", "shots_offtarget", "off_target"]
_BLOCK_KEYS = ["blocked_shots", "shots_blocked", "blocked"]


def _num(v):
    """A stat value as a float. Accepts numbers, numeric strings and {'all'/'total'/'value': n} wrappers."""
    if isinstance(v, dict):
        for k in ("all", "total", "value", "full", "match"):
            if k in v:
                return _num(v[k])
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(x) else x


def _first(block, keys):
    for k in keys:
        if k in block:
            x = _num(block[k])
            if x is not None:
                return x
    return None


def shots_from_block(block):
    """Total shots from one team's stats block, or None. Prefers a total-shots field; if the API only
    gives on-target + off-target (+ blocked), those are added up."""
    if not isinstance(block, dict):
        return None
    lower = {str(k).lower(): v for k, v in block.items()}
    total = _first(lower, [k.lower() for k in _TOTAL_KEYS])
    if total is not None:
        return total
    on, off = _first(lower, _ON_KEYS), _first(lower, _OFF_KEYS)
    if on is not None and off is not None:
        return on + off + (_first(lower, _BLOCK_KEYS) or 0.0)
    return None


def parse_event_stats(data):
    """(home_shots, away_shots) from the /stats/ response, or None if it carries no shots."""
    stats = (data or {}).get("stats") or {}
    h, a = shots_from_block(stats.get("home")), shots_from_block(stats.get("away"))
    if h is None or a is None or h <= 0 or a <= 0:        # a team never takes 0 shots: that is missing data
        return None
    return h, a


# ----------------------------------------------------------------------
# Cache files
# ----------------------------------------------------------------------
def _ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def load_cache():
    """{event_id: (home_shots, away_shots)}"""
    if not os.path.exists(SHOTS_CACHE_CSV):
        return {}
    try:
        df = pd.read_csv(SHOTS_CACHE_CSV)
    except Exception:
        return {}
    out = {}
    for r in df.itertuples(index=False):
        try:
            if float(r.home_shots) <= 0 or float(r.away_shots) <= 0:
                continue                                  # 0 shots = missing data, ignore
            out[int(r.event_id)] = (float(r.home_shots), float(r.away_shots))
        except (TypeError, ValueError):
            continue
    return out


def save_cache(cache):
    _ensure_dir()
    rows = [{"event_id": e, "home_shots": h, "away_shots": a} for e, (h, a) in sorted(cache.items())]
    tmp = SHOTS_CACHE_CSV + ".tmp"
    pd.DataFrame(rows, columns=["event_id", "home_shots", "away_shots"]).to_csv(tmp, index=False)
    os.replace(tmp, SHOTS_CACHE_CSV)


def record(event_id, home_shots, away_shots):
    """Add one match's shots to the cache (used by the xG upgrade, which already made the request)."""
    with _lock:
        cache = load_cache()
        cache[int(event_id)] = (float(home_shots), float(away_shots))
        save_cache(cache)


def record_many(items):
    """items: iterable of (event_id, home_shots, away_shots). One write for all of them."""
    items = list(items)
    if not items:
        return
    with _lock:
        cache = load_cache()
        for e, h, a in items:
            cache[int(e)] = (float(h), float(a))
        save_cache(cache)


def _load_missing():
    if not os.path.exists(SHOTS_MISSING_CSV):
        return {}
    try:
        m = pd.read_csv(SHOTS_MISSING_CSV)
        return {int(r.event_id): (int(r.tries), str(r.last_try)) for r in m.itertuples()}
    except Exception:
        return {}


def _save_missing(missing):
    _ensure_dir()
    pd.DataFrame([{"event_id": e, "tries": t, "last_try": d} for e, (t, d) in sorted(missing.items())],
                 columns=["event_id", "tries", "last_try"]).to_csv(SHOTS_MISSING_CSV, index=False)


# ----------------------------------------------------------------------
# Daily request counter (so the automation never spends more than SHOTS_DAILY_BUDGET a UTC day,
# however often it runs or the app restarts; the manual button and CLI count towards it too)
# ----------------------------------------------------------------------
SHOTS_USAGE_JSON = os.path.join(DATA_DIR, "shots_usage.json")


def _today():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def used_today():
    try:
        import json
        with open(SHOTS_USAGE_JSON, encoding="utf-8") as f:
            d = json.load(f)
        return int(d.get("used", 0)) if d.get("day") == _today() else 0
    except Exception:
        return 0


def _add_usage(n):
    if n <= 0:
        return
    try:
        import json
        _ensure_dir()
        with _lock:
            total = used_today() + int(n)               # read BEFORE the file is reopened for writing (which empties it)
            with open(SHOTS_USAGE_JSON, "w", encoding="utf-8") as f:
                json.dump({"day": _today(), "used": total}, f)
    except Exception:
        pass


# ----------------------------------------------------------------------
# Fetching
# ----------------------------------------------------------------------
def fetch_match_shots_status(event_id):
    """('ok', (h, a)) | ('none', None) | ('quota', None) | ('error', None). A 429 is never taken for 'no shots'."""
    try:
        r = requests.get(f"{BZZOIRO_BASE_URL}/events/{event_id}/stats/", headers=HEADERS, timeout=20)
    except Exception:
        return "error", None
    if r.status_code == 429:
        return "quota", None
    if r.status_code in (404, 204):
        return "none", None
    if r.status_code != 200:
        return "error", None
    try:
        got = parse_event_stats(r.json())
    except Exception:
        return "error", None
    return ("ok", got) if got else ("none", None)


def fetch_shots(event_ids, budget=300, verbose=False):
    """Ask the API for shots of up to `budget` matches (in the order given). Quota-safe: stops at the first
    429. Returns {"asked", "ok", "none", "errors", "quota_hit"}."""
    out = {"asked": 0, "ok": 0, "none": 0, "errors": 0, "quota_hit": False}
    ids = [int(e) for e in event_ids][:max(int(budget), 0)]
    if not ids:
        return out
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    stop = threading.Event()

    def work(eid):
        if stop.is_set():
            return eid, "skipped", None
        st, got = fetch_match_shots_status(eid)
        if st == "quota":
            stop.set()
        return eid, st, got

    found = []
    with _lock:
        missing = _load_missing()
    with ThreadPoolExecutor(max_workers=XG_WORKERS) as pool:
        for eid, st, got in pool.map(work, ids):
            if st == "skipped":
                continue
            out["asked"] += 1
            if st == "ok":
                found.append((eid, got[0], got[1]))
                missing.pop(eid, None)
                out["ok"] += 1
            elif st == "none":
                tries, _ = missing.get(eid, (0, ""))
                missing[eid] = (tries + 1, today)
                out["none"] += 1
            elif st == "quota":
                out["quota_hit"] = True
            else:
                out["errors"] += 1
    record_many(found)
    with _lock:
        _save_missing(missing)
    _add_usage(out["asked"])
    if verbose:
        print(f"  [SHOTS] asked {out['asked']} | got {out['ok']} | none {out['none']} | errors {out['errors']}"
              + (" | quota used up" if out["quota_hit"] else ""))
    return out


# ----------------------------------------------------------------------
# Finished-match table and last-N averages
# ----------------------------------------------------------------------
_TABLE = {"mtime": None, "df": None}


def _matches_table():
    """Finished matches (matches.csv), cached by file modification time."""
    try:
        mt = os.path.getmtime(MATCHES_CSV)
    except OSError:
        return pd.DataFrame()
    if _TABLE["df"] is not None and _TABLE["mtime"] == mt:
        return _TABLE["df"]
    df = pd.read_csv(MATCHES_CSV)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True, format="ISO8601")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["event_id"] = df["event_id"].astype(int)
    _TABLE.update(mtime=mt, df=df)
    return df


def _window_ids(df, team_id, venue, before, n):
    """event_ids of `team_id`'s last n matches at `venue` ('home'|'away') played before `before` (newest first)."""
    col = "home_team_id" if venue == "home" else "away_team_id"
    sub = df[(df[col] == int(team_id)) & (df["date"] < before)]
    return sub.sort_values("date", ascending=False).head(n)["event_id"].tolist()


def team_average(df, cache, team_id, venue, before, n=None):
    """{'avg', 'n', 'window'}: the team's average shots over its last n venue matches before `before`.
    `window` is how many matches that is (fewer than n early in a season / for new teams), `n` how many
    of them have shots data."""
    n = n or SHOTS_WINDOW
    ids = _window_ids(df, team_id, venue, before, n)                     # newest first
    idx = 0 if venue == "home" else 1
    vals = [cache[e][idx] for e in ids if e in cache]
    if not vals:
        return {"avg": None, "n": 0, "window": len(ids)}
    w = [SHOTS_AVG_DECAY ** k for k in range(len(vals))]                 # recency weights (all 1.0 when decay = 1)
    return {"avg": sum(a * b for a, b in zip(vals, w)) / sum(w), "n": len(vals), "window": len(ids)}


# ----------------------------------------------------------------------
# Expected shots: team shots-FOR vs opponent shots-CONCEDED (same engine as the xG model)
# ----------------------------------------------------------------------
_XS_FITS = {}


def _shots_view(df, cache):
    """Finished matches that have shots, with the shots in the columns the ratings model reads."""
    v = df[df["event_id"].isin(cache.keys())].copy()
    v["home_xg"] = v["event_id"].map(lambda e: cache[e][0])
    v["away_xg"] = v["event_id"].map(lambda e: cache[e][1])
    v["home_goals"], v["away_goals"] = v["home_xg"], v["away_xg"]          # unused: xg_weight is 1.0 (target = shots)
    return v


def _xs_fit(cutoff, df, cache):
    """Ratings fitted on shots from matches before `cutoff`. Cached per (cutoff hour, data size)."""
    from model import fit_model
    key = (cutoff.strftime("%Y-%m-%d %H"), len(df), len(cache))
    if key in _XS_FITS:
        return _XS_FITS[key]
    view = _shots_view(df, cache)
    past = view[view["date"] < cutoff]
    fit = None
    if len(past) >= SHOTS_MIN_FIT_MATCHES:
        now = min(cutoff, pd.Timestamp(datetime.now(timezone.utc)))
        fit = fit_model(past, now=now, half_life=SHOTS_HALF_LIFE, ridge=SHOTS_RIDGE, xg_weight=1.0)
    if len(_XS_FITS) > 12:
        _XS_FITS.clear()
    _XS_FITS[key] = fit
    return fit


def expected_shots(fit, home_id, away_id, league_id):
    """Expected shots for one fixture, or None if either team has no shots history.
    home xS = league baseline x home advantage x HOME shots-for strength x AWAY shots-conceded weakness
    away xS = league baseline x AWAY shots-for strength x HOME shots-conceded weakness.
    Strength indices are relative to the league average (1.00 = average)."""
    from model import lambdas_from_fit
    import numpy as np
    i, j = fit["tidx"].get(home_id), fit["tidx"].get(away_id)
    if i is None or j is None:
        return None
    lh, la = lambdas_from_fit(fit, home_id, away_id, league_id)
    return {"xs_home": lh, "xs_away": la,
            "home_for": float(np.exp(fit["att"][i])), "away_conceded": float(np.exp(-fit["dfn"][j])),
            "away_for": float(np.exp(fit["att"][j])), "home_conceded": float(np.exp(-fit["dfn"][i]))}


_CONV = {}


def _conversion(df, cache, league_id, cutoff):
    """Goals per shot (home side, away side) in the league over the year before `cutoff`; global fallback if the league is thin."""
    key = (cutoff.strftime("%Y-%m-%d %H"), len(df), len(cache), league_id)
    if key in _CONV:
        return _CONV[key]
    sub = df[df["event_id"].isin(cache.keys()) & (df["date"] < cutoff) & (df["date"] >= cutoff - pd.Timedelta(days=365))]
    hs = sub["event_id"].map(lambda e: cache[e][0])
    as_ = sub["event_id"].map(lambda e: cache[e][1])
    lg = sub["league_id"] == league_id
    if lg.sum() and hs[lg].sum() > 300:
        out = (sub.loc[lg, "home_goals"].sum() / hs[lg].sum(), sub.loc[lg, "away_goals"].sum() / as_[lg].sum())
    elif len(sub) and hs.sum() > 0:
        out = (sub["home_goals"].sum() / hs.sum(), sub["away_goals"].sum() / as_.sum())
    else:
        out = None
    if len(_CONV) > 200:
        _CONV.clear()
    _CONV[key] = out
    return out


def blend_with_goals(df, home_id, away_id, league_id, lh, la):
    """The goals model's expected goals (lh, la) blended with shots-based expected goals when SHOTS_BLEND_WEIGHT > 0.
    shots-based goals = expected shots (for/against model) x league goals per shot. Uses only matches in `df`, so it is
    point-in-time for history. Falls back to (lh, la) unchanged when shots data is missing for either team."""
    w = SHOTS_BLEND_WEIGHT
    if w <= 0:
        return lh, la
    try:
        import numpy as np
        cache = load_cache()
        if not cache:
            return lh, la
        cutoff = df["date"].max() + pd.Timedelta(seconds=1)
        fit = _xs_fit(cutoff, df, cache)
        if fit is None:
            return lh, la
        xs = expected_shots(fit, home_id, away_id, league_id)
        conv = _conversion(df, cache, league_id, cutoff)
        if xs is None or conv is None or conv[0] <= 0 or conv[1] <= 0:
            return lh, la
        sg_h, sg_a = xs["xs_home"] * conv[0], xs["xs_away"] * conv[1]
        return (float(np.exp(w * np.log(sg_h) + (1 - w) * np.log(lh))),
                float(np.exp(w * np.log(sg_a) + (1 - w) * np.log(la))))
    except Exception:
        return lh, la


# ----------------------------------------------------------------------
# Fixtures for one day
# ----------------------------------------------------------------------
def eat_date_of(value):
    try:
        ts = pd.Timestamp(value)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        return ts.astimezone(EAT).strftime("%Y-%m-%d")
    except Exception:
        return None


def eat_time_of(value):
    try:
        ts = pd.Timestamp(value)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        return ts.astimezone(EAT).strftime("%H:%M")
    except Exception:
        return ""


_FIX_CACHE = {}          # iso day -> (fetched_at, [fixtures]); light fixture lookups for days outside the loaded window
_FIX_TTL = 6 * 3600


def fetch_fixtures_for_day(iso):
    """Not-yet-played fixtures for one Nairobi day, straight from the API (one request per league, no odds).
    Returns (fixtures, error_or_None). Cached for a few hours so flicking between days costs nothing twice."""
    hit = _FIX_CACHE.get(iso)
    if hit and (datetime.now().timestamp() - hit[0]) < _FIX_TTL:
        return hit[1], None
    day = datetime.strptime(iso, "%Y-%m-%d")
    date_from = (day - timedelta(days=1)).strftime("%Y-%m-%d")      # an EAT day straddles two UTC days
    date_to = (day + timedelta(days=1)).strftime("%Y-%m-%d")        # a day of slack in case date_to is exclusive
    out, seen, err = [], set(), None
    for lid in getattr(config, "LEAGUES_TO_FETCH", []):
        offset, limit = 0, 200
        while True:
            try:
                r = requests.get(f"{BZZOIRO_BASE_URL}/events/", headers=HEADERS, timeout=30,
                                 params={"league_id": lid, "date_from": date_from, "date_to": date_to,
                                         "status": "notstarted", "limit": limit, "offset": offset})
            except Exception as e:
                err = f"network error: {e}"
                break
            if r.status_code != 200:
                err = "API quota used up (HTTP 429)" if r.status_code == 429 else f"API error HTTP {r.status_code}"
                break
            data = r.json()
            res = data.get("results", [])
            if not res:
                break
            for m in res:
                if not m.get("home_team_id") or not m.get("away_team_id") or m.get("id") in seen:
                    continue
                if eat_date_of(m.get("event_date")) != iso:
                    continue
                seen.add(m.get("id"))
                out.append({"event_id": m.get("id"), "date": m.get("event_date"), "league_id": m.get("league_id"),
                            "home": m.get("home_team"), "home_id": m.get("home_team_id"),
                            "away": m.get("away_team"), "away_id": m.get("away_team_id"), "status": "upcoming"})
            if not data.get("next"):
                break
            offset += limit
        if err and "429" in err:
            break
    if not err:
        _FIX_CACHE[iso] = (datetime.now().timestamp(), out)
    return out, err


def day_fixtures(iso, upcoming=None):
    """Every fixture on Nairobi day `iso` that is already known locally: finished matches (matches.csv) plus the
    upcoming fixtures passed in (today's saved picks). No API calls. Each is a dict with event_id, date, league_id,
    home/away names and ids, status ('finished'|'upcoming'), and for finished ones the score."""
    out, seen = [], set()
    df = _matches_table()
    if not df.empty:
        # a Nairobi day is UTC [iso-1 21:00, iso 21:00)
        start = pd.Timestamp(iso, tz=EAT).tz_convert("UTC")
        end = start + pd.Timedelta(days=1)
        day = df[(df["date"] >= start) & (df["date"] < end)]
        for r in day.itertuples(index=False):
            seen.add(int(r.event_id))
            out.append({"event_id": int(r.event_id), "date": r.date.isoformat(), "league_id": int(r.league_id),
                        "home": r.home_team, "home_id": int(r.home_team_id),
                        "away": r.away_team, "away_id": int(r.away_team_id),
                        "status": "finished", "score": f"{int(r.home_goals)}-{int(r.away_goals)}"})
    for m in upcoming or []:
        try:
            if m.get("event_id") in seen or eat_date_of(m.get("date")) != iso:
                continue
            out.append({"event_id": m["event_id"], "date": m["date"], "league_id": m["league_id"],
                        "home": m["home"], "home_id": int(m["home_id"]),
                        "away": m["away"], "away_id": int(m["away_id"]), "status": "upcoming"})
        except (KeyError, TypeError, ValueError):
            continue
    return out


def build_rows(fixtures):
    """Shots rows for the given fixtures, sorted by average shots (highest first), plus coverage numbers.
    Fixtures without both teams' averages go to the bottom (marked incomplete)."""
    df = _matches_table()
    cache = load_cache()
    missing = _load_missing()
    now = datetime.now(timezone.utc)
    rows, need = [], set()
    xs_fit = None
    if not df.empty and fixtures:
        kos = []
        for f in fixtures:
            try:
                k = pd.Timestamp(f["date"])
                kos.append(k.tz_localize("UTC") if k.tzinfo is None else k.tz_convert("UTC"))
            except Exception:
                pass
        if kos:
            xs_fit = _xs_fit(min(kos), df, cache)
    for f in fixtures:
        try:
            ko = pd.Timestamp(f["date"])
            ko = ko.tz_localize("UTC") if ko.tzinfo is None else ko.tz_convert("UTC")
        except Exception:
            continue
        hw = _window_ids(df, f["home_id"], "home", ko, SHOTS_WINDOW) if not df.empty else []
        aw = _window_ids(df, f["away_id"], "away", ko, SHOTS_WINDOW) if not df.empty else []
        need.update(hw)
        need.update(aw)
        h = team_average(df, cache, f["home_id"], "home", ko) if not df.empty else {"avg": None, "n": 0, "window": 0}
        a = team_average(df, cache, f["away_id"], "away", ko) if not df.empty else {"avg": None, "n": 0, "window": 0}
        complete = h["avg"] is not None and a["avg"] is not None
        row = {
            "event_id": f["event_id"], "kickoff": eat_time_of(f["date"]), "date": str(f["date"]),
            "league": league_name(f["league_id"]), "home": f["home"], "away": f["away"],
            "status": f.get("status", "upcoming"), "score": f.get("score"),
            "home_avg": round(h["avg"], 1) if h["avg"] is not None else None,
            "away_avg": round(a["avg"], 1) if a["avg"] is not None else None,
            "avg": round((h["avg"] + a["avg"]) / 2, 1) if complete else None,
            "combined": round(h["avg"] + a["avg"], 1) if complete else None,
            "home_n": h["n"], "home_window": h["window"], "away_n": a["n"], "away_window": a["window"],
            "complete": complete,
        }
        xs = expected_shots(xs_fit, f["home_id"], f["away_id"], f["league_id"]) if xs_fit else None
        if xs:
            row.update(xs_home=round(xs["xs_home"], 1), xs_away=round(xs["xs_away"], 1),
                       xs_total=round(xs["xs_home"] + xs["xs_away"], 1),
                       home_for=round(xs["home_for"], 2), away_conceded=round(xs["away_conceded"], 2),
                       away_for=round(xs["away_for"], 2), home_conceded=round(xs["home_conceded"], 2))
        else:
            row.update(xs_home=None, xs_away=None, xs_total=None)
        if f.get("status") == "finished" and f["event_id"] in cache:           # past day: what actually happened
            row["actual_home"], row["actual_away"] = cache[f["event_id"]]
        rows.append(row)
    rows.sort(key=lambda r: (r["avg"] is None, -(r["avg"] or 0), r["kickoff"]))
    have = sum(1 for e in need if e in cache)
    ages = dict(zip(df["event_id"], df["date"])) if not df.empty else {}
    def _retryable(e):
        tries = missing.get(e, (0, ""))[0]
        age = (now - ages[e]).days if e in ages else 999
        return tries < SHOTS_GIVE_UP_TRIES or age <= SHOTS_GIVE_UP_AGE_DAYS
    fetchable = [e for e in need if e not in cache and _retryable(e)]
    return rows, {"needed": len(need), "have": have, "missing": len(need) - have, "fetchable": len(fetchable),
                  "fetchable_ids": sorted(fetchable)}


def missing_ids_for(fixtures):
    """Event ids still worth fetching for these fixtures (their last-N venue windows), newest first."""
    df = _matches_table()
    if df.empty:
        return []
    cache, missing = load_cache(), _load_missing()
    dates = dict(zip(df["event_id"], df["date"]))
    need = set()
    for f in fixtures:
        try:
            ko = pd.Timestamp(f["date"])
            ko = ko.tz_localize("UTC") if ko.tzinfo is None else ko.tz_convert("UTC")
        except Exception:
            continue
        need.update(_window_ids(df, f["home_id"], "home", ko, SHOTS_WINDOW))
        need.update(_window_ids(df, f["away_id"], "away", ko, SHOTS_WINDOW))
    now = datetime.now(timezone.utc)
    def _live(e):
        tries = missing.get(e, (0, ""))[0]
        age = (now - dates[e]).days if e in dates else 999
        return tries < SHOTS_GIVE_UP_TRIES or age <= SHOTS_GIVE_UP_AGE_DAYS
    todo = [e for e in need if e not in cache and _live(e)]
    return sorted(todo, key=lambda e: dates.get(e), reverse=True)


# ----------------------------------------------------------------------
# Daily budgeted backfill (runs from app.py's automation)
# ----------------------------------------------------------------------
def backlog_ids(upcoming=None):
    """Event ids to fetch, most useful first: (1) the last-10 venue windows of the teams in the upcoming fixtures,
    then (2) every other finished match, newest first. Skips cached ones, and ones given up on."""
    df = _matches_table()
    if df.empty:
        return []
    cache, missing = load_cache(), _load_missing()
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    first = missing_ids_for([{"date": now.isoformat(), "home_id": m["home_id"], "away_id": m["away_id"]}
                             for m in (upcoming or []) if m.get("home_id") and m.get("away_id")])
    rest = df.sort_values("date", ascending=False)
    ids, seen = [], set()
    ages = dict(zip(df["event_id"], df["date"]))
    for e in list(first) + rest["event_id"].tolist():
        if e in seen or e in cache:
            continue
        seen.add(e)
        tries, last = missing.get(e, (0, ""))
        age = (now - ages[e]).days if e in ages else 999
        if tries >= SHOTS_GIVE_UP_TRIES and age > SHOTS_GIVE_UP_AGE_DAYS:
            continue                                  # still nothing a week after the match: the API has no shots for it
        if last == today:
            continue                                  # already asked today
        ids.append(e)
    return ids


def coverage():
    """Local-only numbers for status displays (no API calls)."""
    out = {"matches": 0, "have": 0, "pct": 0}
    df = _matches_table()
    if df.empty:
        return out
    cache = load_cache()
    out["matches"] = len(df)
    out["have"] = int(df["event_id"].isin(cache.keys()).sum())
    out["pct"] = round(100 * out["have"] / max(len(df), 1))
    return out


def fetch_all_shots(budget=None, upcoming=None, chunk=200, verbose=True):
    """ONE function that fetches shots for every finished match in the database (matches.csv) that has none yet.

    - Order: the last-10 home/away windows of the teams in `upcoming` fixtures first, then everything else newest first.
    - One API request per match (the API has no per-team call), so ~6,000 matches = ~6,000 requests.
    - Resumable: results are saved after every chunk, matches already cached are skipped, so stopping or a crash loses nothing.
    - Quota-safe: stops at the first HTTP 429; matches with no shots are retried up to 3 times, then given up.
    - budget: max requests this call (None = no cap, i.e. until everything is fetched or the quota stops it).
    Returns {"backlog", "asked", "ok", "none", "errors", "quota_hit", "left"}.
    """
    ids = backlog_ids(upcoming)
    tot = {"backlog": len(ids), "asked": 0, "ok": 0, "none": 0, "errors": 0, "quota_hit": False}
    cap = len(ids) if budget is None else max(int(budget), 0)
    if verbose:
        print(f"  [SHOTS] {len(ids)} matches without shots | this run: up to {cap} requests")
    pos = 0
    while pos < len(ids) and tot["asked"] < cap and not tot["quota_hit"]:
        n = min(chunk, cap - tot["asked"])
        part = fetch_shots(ids[pos:pos + n], budget=n, verbose=False)
        pos += n
        for k in ("asked", "ok", "none", "errors"):
            tot[k] += part[k]
        tot["quota_hit"] = part["quota_hit"]
        if verbose:
            print(f"  [SHOTS] {tot['asked']}/{min(cap, len(ids))} asked | {tot['ok']} with shots | "
                  f"{tot['none']} none | {tot['errors']} errors" + (" | QUOTA USED UP" if tot["quota_hit"] else ""))
        if part["asked"] and part["errors"] == part["asked"]:
            break                                     # every request failed (network down?): stop instead of looping
    tot["left"] = max(len(ids) - tot["ok"] - tot["none"], 0)
    return tot


def upgrade_shots(budget=300, upcoming=None, verbose=True):
    """Daily automation entry: fetch_all_shots limited to `budget`."""
    return fetch_all_shots(budget=budget, upcoming=upcoming, verbose=verbose)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _probe(event_id):
    r = requests.get(f"{BZZOIRO_BASE_URL}/events/{event_id}/stats/", headers=HEADERS, timeout=20)
    print(f"HTTP {r.status_code}")
    if r.status_code != 200:
        print(r.text[:300])
        return
    data = r.json()
    stats = (data or {}).get("stats") or {}
    for side in ("home", "away"):
        blk = stats.get(side) or {}
        print(f"\n[{side}] {len(blk)} fields. Shot-related:")
        for k, v in blk.items():
            if "shot" in str(k).lower() or "att" in str(k).lower() or "target" in str(k).lower():
                print(f"   {k!r}: {v!r}")
    print("\nParsed shots (home, away):", parse_event_stats(data))
    print("If that says None, find the right field name above and add it to config.py, e.g.\n"
          "    SHOTS_STAT_KEYS = ['the_field_name']")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--probe" in args:
        _probe(args[args.index("--probe") + 1])
    elif "--all" in args:
        i = args.index("--all")
        n = int(args[i + 1]) if len(args) > i + 1 and args[i + 1].isdigit() else None
        print(fetch_all_shots(budget=n))
    elif "--backfill" in args:
        i = args.index("--backfill")
        n = int(args[i + 1]) if len(args) > i + 1 and args[i + 1].isdigit() else 300
        r = upgrade_shots(budget=n)
        print(r)
    else:
        c = coverage()
        print(f"shots on {c['have']} of {c['matches']} finished matches ({c['pct']}%)")
        print(f"cache: {SHOTS_CACHE_CSV}")
