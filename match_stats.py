# match_stats.py
"""
Team stats per match (shots, shots on target, corners, fouls, cards, offsides, throw-ins, goalkeeper saves) and the
"Trends" built on them.

  Store     data/match_stats.csv  (event_id, <market>_h, <market>_a). One /events/{id}/stats/ request fills every market at
            once, and also saves the match's shots into the Shots cache. Matches the API has no stats for are remembered in
            data/stats_missing.csv and retried for a week, like the shots cache.

  Trend     For a team, a stat and a line, how many of the team's LAST 10 matches (home and away together) went Over / Under.
            Three views per stat:  FOR (the team's own number), AGAINST (what it conceded = the opponent's number),
            TOTAL (both teams together in its matches).
            For each team, stat, view and side the card shows the most demanding line that still hit >= MIN_HITS of 10.
            Lines are limited to a sensible range (between the 10th and 90th percentile of that stat), so "Over 0.5 fouls"
            is never offered.

  Model     The same ridge for/against engine as the xG and shots models, fitted on each stat, gives the expected value for
            the fixture; a negative binomial (spread measured on the last 120 days, out of sample) turns it into the
            probability of the line. It is shown NEXT to the hit rate: a 10/10 streak is a fact about the past, the model
            says how likely the line is for this opponent.

CLI:  python match_stats.py --status | --fill [N]
"""
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

import config
from config import BZZOIRO_API_KEY, BZZOIRO_BASE_URL, DATA_DIR
import shots_data as SD

MARKETS = {                       # key: (label, API field)
    "shots": ("Shots", "total_shots"),
    "sot": ("Shots on target", "shots_on_target"),
    "corners": ("Corners", "corner_kicks"),
    "fouls": ("Fouls", "fouls"),
    "cards": ("Yellow cards", "yellow_cards"),
    "offsides": ("Offsides", "offsides"),
    "throwins": ("Throw-ins", "throw_ins"),
    "saves": ("GK saves", "goalkeeper_saves"),
}
VARIANTS = {"for": "For", "against": "Against", "total": "Match total"}
GOAL_MARKETS = {"goals": "Goals O/U 2.5", "btts": "BTTS"}      # built from the scores already in matches.csv: no extra requests
GOAL_LINE = 2.5
WINDOW = 10
STORE_CSV = os.path.join(DATA_DIR, "match_stats.csv")
MISSING_CSV = os.path.join(DATA_DIR, "stats_missing.csv")
USAGE_JSON = os.path.join(DATA_DIR, "stats_usage.json")
GIVE_UP_TRIES, GIVE_UP_AGE_DAYS = 3, 7
MODEL_HALF_LIFE = float(getattr(config, "STATS_HALF_LIFE", 180.0))
MIN_FIT_MATCHES = 200
HEADERS = {"Authorization": f"Token {BZZOIRO_API_KEY}"}
XG_WORKERS = getattr(config, "XG_WORKERS", 8)
COLS = [f"{k}_{s}" for k in MARKETS for s in ("h", "a")]
_lock = threading.RLock()


# ---------------------------------------------------------------- store
_STORE = {"mtime": None, "df": None}


def load_store():
    """DataFrame indexed by event_id with the *_h / *_a columns (cached by file modification time)."""
    try:
        mt = os.path.getmtime(STORE_CSV)
    except OSError:
        return pd.DataFrame(columns=COLS)
    if _STORE["df"] is not None and _STORE["mtime"] == mt:
        return _STORE["df"]
    df = pd.read_csv(STORE_CSV)
    df["event_id"] = df["event_id"].astype(int)
    df = df.drop_duplicates("event_id", keep="last").set_index("event_id")
    for c in COLS:
        if c not in df.columns:
            df[c] = np.nan
    _STORE.update(mtime=mt, df=df[COLS])
    return _STORE["df"]


def save_rows(rows):
    """rows: {event_id: {col: value}} merged into the store file."""
    if not rows:
        return
    with _lock:
        os.makedirs(DATA_DIR, exist_ok=True)
        cur = load_store().copy()
        new = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=COLS)
        new.index = new.index.astype(int)
        cur = pd.concat([cur[~cur.index.isin(new.index)], new]).sort_index()
        tmp = STORE_CSV + ".tmp"
        cur.reset_index().rename(columns={"index": "event_id"}).to_csv(tmp, index=False)
        os.replace(tmp, STORE_CSV)
        _STORE["mtime"] = None


def _load_missing():
    if not os.path.exists(MISSING_CSV):
        return {}
    try:
        m = pd.read_csv(MISSING_CSV)
        return {int(r.event_id): (int(r.tries), str(r.last_try)) for r in m.itertuples()}
    except Exception:
        return {}


def _save_missing(missing):
    pd.DataFrame([{"event_id": e, "tries": t, "last_try": d} for e, (t, d) in sorted(missing.items())],
                 columns=["event_id", "tries", "last_try"]).to_csv(MISSING_CSV, index=False)


def _today():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def used_today():
    try:
        with open(USAGE_JSON, encoding="utf-8") as f:
            d = json.load(f)
        return int(d.get("used", 0)) if d.get("day") == _today() else 0
    except Exception:
        return 0


def _add_usage(n):
    if n <= 0:
        return
    with _lock:
        total = used_today() + int(n)
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(USAGE_JSON, "w", encoding="utf-8") as f:
            json.dump({"day": _today(), "used": total}, f)


# ---------------------------------------------------------------- fetching
def parse_stats(data):
    """{col: value} for every market the response carries, or None if it has too little."""
    stats = (data or {}).get("stats") or {}
    out, got = {}, 0
    for key, (_label, field) in MARKETS.items():
        for side, suffix in (("home", "h"), ("away", "a")):
            blk = stats.get(side) or {}
            lower = {str(k).lower(): v for k, v in blk.items()} if isinstance(blk, dict) else {}
            v = SD._num(lower.get(field))
            if key == "shots" and v is not None and v <= 0:
                v = None                                   # 0 shots = missing data
            out[f"{key}_{suffix}"] = v
            got += v is not None
    return out if got >= 6 else None


def fetch_one(event_id):
    try:
        r = requests.get(f"{BZZOIRO_BASE_URL}/events/{event_id}/stats/", headers=HEADERS, timeout=25)
    except Exception:
        return "error", None
    if r.status_code == 429:
        return "quota", None
    if r.status_code in (204, 404):
        return "none", None
    if r.status_code != 200:
        return "error", None
    try:
        data = r.json()
        got = parse_stats(data)
        xg = parse_xg(data)
    except Exception:
        return "error", None
    return ("ok", (got, xg)) if (got or xg) else ("none", None)


def parse_xg(data):
    """(home_xg, away_xg) if the response carries both, else None. Same field the xG upgrade reads."""
    stats = (data or {}).get("stats") or {}
    try:
        h, a = float((stats.get("home") or {}).get("expected_goals")), float((stats.get("away") or {}).get("expected_goals"))
    except (TypeError, ValueError):
        return None
    return None if (pd.isna(h) or pd.isna(a)) else (h, a)


def fetch_stats(event_ids, budget=300, verbose=False):
    """One request per match, quota-safe (stops at the first 429). Returns {"asked","ok","none","errors","quota_hit"}."""
    out = {"asked": 0, "ok": 0, "none": 0, "errors": 0, "quota_hit": False, "xg_added": 0}
    ids = [int(e) for e in event_ids][:max(int(budget), 0)]
    if not ids:
        return out
    stop = threading.Event()

    def work(eid):
        if stop.is_set():
            return eid, "skipped", None
        st, got = fetch_one(eid)
        if st == "quota":
            stop.set()
        return eid, st, got

    rows, shots, xgs, missing = {}, [], {}, _load_missing()
    with ThreadPoolExecutor(max_workers=XG_WORKERS) as pool:
        for eid, st, got in pool.map(work, ids):
            if st == "skipped":
                continue
            out["asked"] += 1
            if st == "ok":
                row, xg = got
                if row:
                    rows[eid] = row
                    if row.get("shots_h") and row.get("shots_a"):
                        shots.append((eid, row["shots_h"], row["shots_a"]))
                if xg:
                    xgs[eid] = xg
                missing.pop(eid, None)
                out["ok"] += 1
            elif st == "none":
                tries, _ = missing.get(eid, (0, ""))
                missing[eid] = (tries + 1, _today())
                out["none"] += 1
            elif st == "quota":
                out["quota_hit"] = True
            else:
                out["errors"] += 1
    save_rows(rows)
    SD.record_many(shots)                                    # keep the Shots cache in step, at no extra cost
    if xgs:                                                  # ... and the real-xG cache the goals model upgrades from
        try:
            import fetch_data as FD
            cache = FD._load_xg_cache()
            new = {e: v for e, v in xgs.items() if e not in cache}
            cache.update(new)
            FD._save_xg_cache(cache)
            out["xg_added"] = len(new)
        except Exception as e:
            print(f"  [STATS] could not save xG: {e}")
    with _lock:
        _save_missing(missing)
    _add_usage(out["asked"])
    if verbose:
        print(f"  [STATS] asked {out['asked']} | got {out['ok']} | none {out['none']} | errors {out['errors']}"
              + (" | quota used up" if out["quota_hit"] else ""))
    return out


# ---------------------------------------------------------------- windows and backlog
def _team_window_ids(df, team_id, before, n=WINDOW):
    sub = df[((df["home_team_id"] == int(team_id)) | (df["away_team_id"] == int(team_id))) & (df["date"] < before)]
    return sub.sort_values("date", ascending=False).head(n)["event_id"].tolist()


def _ko(v):
    ts = pd.Timestamp(v)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def needed_ids(fixtures):
    """Event ids still worth fetching for these fixtures: each team's last-10 matches. Newest first."""
    df = SD._matches_table()
    if df.empty:
        return []
    store, missing = load_store(), _load_missing()
    now = datetime.now(timezone.utc)
    dates = dict(zip(df["event_id"], df["date"]))
    need = set()
    for f in fixtures:
        try:
            ko = _ko(f["date"])
        except Exception:
            continue
        need.update(_team_window_ids(df, f["home_id"], ko))
        need.update(_team_window_ids(df, f["away_id"], ko))
    def live(e):
        tries = missing.get(e, (0, ""))[0]
        age = (now - dates[e]).days if e in dates else 999
        return tries < GIVE_UP_TRIES or age <= GIVE_UP_AGE_DAYS
    return sorted([e for e in need if e not in store.index and live(e)], key=lambda e: dates.get(e), reverse=True)


def backlog_ids(upcoming=None):
    """(priority, rest): priority = the last-10 windows of the teams in the upcoming fixtures (what the Trends tab shows
    first); rest = every other finished match without stats, newest first."""
    df = SD._matches_table()
    if df.empty:
        return [], []
    store, missing = load_store(), _load_missing()
    now = datetime.now(timezone.utc)
    ages = dict(zip(df["event_id"], df["date"]))
    first = needed_ids([{"date": now.isoformat(), "home_id": m["home_id"], "away_id": m["away_id"]}
                        for m in (upcoming or []) if m.get("home_id") and m.get("away_id")])

    def due(e):
        tries, last = missing.get(e, (0, ""))
        age = (now - ages[e]).days if e in ages else 999
        return not (tries >= GIVE_UP_TRIES and age > GIVE_UP_AGE_DAYS) and last != _today()

    prio = [e for e in first if due(e)]
    pset = set(prio)
    rest = [e for e in df.sort_values("date", ascending=False)["event_id"].tolist()
            if e not in pset and e not in store.index and due(e)]
    return prio, rest


PRIORITY_CAP = 800            # the teams playing soon are always fetched, even past the daily budget, up to this many


def upgrade_stats(budget=1000, upcoming=None, chunk=200, verbose=True):
    """Daily automation entry. Order: (1) last-10 windows of the teams in the upcoming fixtures (up to PRIORITY_CAP, even
    beyond `budget`, so the Trends tab works at launch), then (2) every other match, newest first, until `budget` is spent.
    Saves after every chunk and stops at the first 429. Every request also fills the xG and shots caches."""
    prio, rest = backlog_ids(upcoming)
    ids = prio + rest
    cap = min(len(ids), max(int(budget), min(len(prio), PRIORITY_CAP)))
    tot = {"backlog": len(ids), "priority": len(prio), "asked": 0, "ok": 0, "none": 0, "errors": 0,
           "quota_hit": False, "xg_added": 0}
    if verbose:
        print(f"  [STATS] {len(prio)} priority + {len(rest)} other matches without stats | this run: up to {cap} requests")
    pos = 0
    while pos < len(ids) and tot["asked"] < cap and not tot["quota_hit"]:
        n = min(chunk, cap - tot["asked"])
        part = fetch_stats(ids[pos:pos + n], budget=n)
        pos += n
        for k in ("asked", "ok", "none", "errors", "xg_added"):
            tot[k] += part[k]
        tot["quota_hit"] = part["quota_hit"]
        if verbose:
            print(f"  [STATS] {tot['asked']}/{cap} asked | {tot['ok']} stored | {tot['none']} none | {tot['errors']} errors")
        if part["asked"] and part["errors"] == part["asked"]:
            break
    tot["left"] = max(len(ids) - tot["ok"] - tot["none"], 0)
    return tot


def coverage():
    df = SD._matches_table()
    store = load_store()
    n = len(df)
    have = int(df["event_id"].isin(store.index).sum()) if n else 0
    return {"matches": n, "have": have, "pct": round(100 * have / max(n, 1))}


# ---------------------------------------------------------------- series, lines, trends
def _value(row, key, who, variant):
    """Value of one stat for team side `who` ('h'/'a' = the team's own side in that match) under a view."""
    other = "a" if who == "h" else "h"
    a, b = row.get(f"{key}_{who}"), row.get(f"{key}_{other}")
    if variant == "for":
        return a
    if variant == "against":
        return b
    return None if (a is None or b is None or pd.isna(a) or pd.isna(b)) else a + b


def team_matches(df, store, team_id, before, n=WINDOW):
    """The team's last n matches before `before` (newest first) as dicts with the stat row attached."""
    out = []
    ids = _team_window_ids(df, team_id, before, n)
    byid = df.set_index("event_id")
    for e in ids:
        m = byid.loc[e]
        home = int(m["home_team_id"]) == int(team_id)
        row = store.loc[e].to_dict() if e in store.index else {}
        out.append({"event_id": e, "date": m["date"], "who": "h" if home else "a", "venue": "H" if home else "A",
                    "opp": m["away_team"] if home else m["home_team"], "row": row,
                    "hg": int(m["home_goals"]), "ag": int(m["away_goals"])})
    return out


def line_grid(store, key, variant):
    """Offered lines for a stat and view: .5 lines between the 10th and 90th percentile of its values."""
    if store.empty:
        return []
    h, a = store[f"{key}_h"].astype(float), store[f"{key}_a"].astype(float)
    v = (h + a).dropna() if variant == "total" else pd.concat([h, a]).dropna()
    if len(v) < 100:
        return []
    lo, hi = np.percentile(v, [10, 90])
    return [x + 0.5 for x in range(int(np.floor(lo)) - 1, int(np.ceil(hi)) + 1) if x + 0.5 > 0 and lo - 0.5 <= x + 0.5 <= hi + 0.5]


LEVELS = (10, 9, 8)           # hit thresholds a card can be built for (the Top 10 uses all of them)


def best_lines(vals, grid, level):
    """{'Over': (line, hits), 'Under': (line, hits)}: the most demanding line still hit >= `level` times."""
    res = {}
    for L in grid:
        over = sum(v > L for v in vals)
        under = sum(v < L for v in vals)
        if over >= level and ("Over" not in res or L > res["Over"][0]):
            res["Over"] = (L, over)
        if under >= level and ("Under" not in res or L < res["Under"][0]):
            res["Under"] = (L, under)
    return res


# ---------------------------------------------------------------- model
_MODELS = {}


def _stat_view(df, store, key):
    v = df[df["event_id"].isin(store.index)].copy()
    v["home_xg"] = v["event_id"].map(store[f"{key}_h"])
    v["away_xg"] = v["event_id"].map(store[f"{key}_a"])
    v = v.dropna(subset=["home_xg", "away_xg"])
    v["home_goals"], v["away_goals"] = v["home_xg"], v["away_xg"]
    return v


def _nb_k(mu, y):
    """Negative-binomial size k from out-of-sample residuals: var = mu + mu^2/k. Larger k = closer to Poisson."""
    mu, y = np.asarray(mu, float), np.asarray(y, float)
    den = np.sum((y - mu) ** 2 - mu)
    return float(np.clip(np.sum(mu ** 2) / den, 3.0, 300.0)) if den > 0 else 300.0


def stat_model(df, store, key, cutoff):
    """{'fit','k_for','k_total'} for one stat as of `cutoff`, or None when there is too little data."""
    from model import fit_model, lambdas_from_fit
    ck = (key, cutoff.strftime("%Y-%m-%d %H"), len(df), len(store))
    if ck in _MODELS:
        return _MODELS[ck]
    view = _stat_view(df, store, key)
    past = view[view["date"] < cutoff]
    out = None
    if len(past) >= MIN_FIT_MATCHES:
        mean = float(pd.concat([past["home_xg"], past["away_xg"]]).mean())
        ridge = max(5.0, 3.0 * mean)
        now = min(cutoff, pd.Timestamp(datetime.now(timezone.utc)))
        fit = fit_model(past, now=now, half_life=MODEL_HALF_LIFE, ridge=ridge, xg_weight=1.0)
        # spread: fit on data older than 120 days, score the last 120 days
        split = cutoff - pd.Timedelta(days=120)
        old, recent = past[past["date"] < split], past[past["date"] >= split]
        k_for = k_tot = 60.0
        if len(old) >= MIN_FIT_MATCHES and len(recent) >= 100:
            f2 = fit_model(old, now=split, half_life=MODEL_HALF_LIFE, ridge=ridge, xg_weight=1.0)
            mus = np.array([lambdas_from_fit(f2, r.home_team_id, r.away_team_id, r.league_id) for r in recent.itertuples()])
            ys = recent[["home_xg", "away_xg"]].values
            k_for = _nb_k(mus.ravel(), ys.ravel())
            k_tot = _nb_k(mus.sum(axis=1), ys.sum(axis=1))
        out = {"fit": fit, "k_for": k_for, "k_total": k_tot, "ridge": ridge}
    if len(_MODELS) > 40:
        _MODELS.clear()
    _MODELS[ck] = out
    return out


def p_line(mu, k, line, side):
    """Probability of Over/Under `line` (a .5 line) for a negative binomial with mean mu and size k."""
    from scipy.stats import nbinom
    cdf = nbinom.cdf(int(np.floor(line)), k, k / (k + mu))
    return float(1 - cdf) if side == "Over" else float(cdf)


# ---------------------------------------------------------------- goals / BTTS trends
_GDF = {"mtime": None, "df": None}


def _goal_df():
    """The goals model's data (matches_with_xg.csv via model.load_data), cached by file modification time."""
    import model
    path = model.MATCHES_XG_CSV if os.path.exists(model.MATCHES_XG_CSV) else model.MATCHES_CSV
    mt = os.path.getmtime(path)
    if _GDF["df"] is None or _GDF["mtime"] != mt:
        _GDF.update(mtime=mt, df=model.load_data())
    return _GDF["df"]


def goal_cards(f, side_name, name, opp_name, where, ms, hist, probs):
    """Over/Under 2.5 goals and BTTS Yes/No cards for one team from its last 10 matches (all venues)."""
    cards = []
    totals = [m["hg"] + m["ag"] for m in ms]
    both = [m["hg"] > 0 and m["ag"] > 0 for m in ms]
    base = {"event_id": f["event_id"], "team": name, "opp": opp_name, "where": where,
            "kickoff": SD.eat_time_of(f["date"]), "league": SD.league_name(f["league_id"]), "n": len(ms),
            "variant": "total", "variant_label": "Match", "fixed": True, "avg": round(float(np.mean(totals)), 1),
            "median": round(float(np.median(totals)), 1), "expected": probs["exp_goals"]}

    def bars(hit_fn):
        return [{"d": m["date"].strftime("%d/%m"), "v": float(t), "ven": m["venue"], "opp": m["opp"],
                 "sc": f"{m['hg']}-{m['ag']}", "hit": bool(hit_fn(m, t))} for m, t in zip(ms, totals)]

    over, under = sum(t > GOAL_LINE for t in totals), sum(t < GOAL_LINE for t in totals)
    for sd, hits, pr in (("Over", over, probs["over"]), ("Under", under, probs["under"])):
        if hits >= min(LEVELS):
            cards.append({**base, "market": "goals", "market_label": "Goals", "side": sd, "line": GOAL_LINE, "hits": hits, "level": hits,
                          "model_prob": round(pr * 100, 1),
                          "bars": bars(lambda m, t, sd=sd: t > GOAL_LINE if sd == "Over" else t < GOAL_LINE)})
    yes = sum(both)
    for sd, hits, pr in (("Yes", yes, probs["btts_yes"]), ("No", len(ms) - yes, probs["btts_no"])):
        if hits >= min(LEVELS):
            cards.append({**base, "market": "btts", "market_label": "Both teams to score", "side": sd, "line": None,
                          "hits": hits, "level": hits, "model_prob": round(pr * 100, 1),
                          "bars": bars(lambda m, t, sd=sd: (m["hg"] > 0 and m["ag"] > 0) == (sd == "Yes"))})
    return cards


def fixture_goal_probs(hist, f):
    """The app's own probabilities for this fixture: Over/Under 2.5 (goals model incl. the shots blend) and BTTS."""
    import model
    import btts_model
    lh_raw, la_raw = model.predict_lambdas(hist, f["home_id"], f["away_id"], f["league_id"])
    lh, la = lh_raw * model.LAMBDA_BOOST, la_raw * model.LAMBDA_BOOST
    p_over, p_under, _ = model.over_under_2_5(lh, la)
    py = btts_model.p_yes(lh, la)
    return {"over": float(p_over), "under": float(p_under), "btts_yes": float(py), "btts_no": 1 - float(py),
            "exp_goals": round(float(lh + la), 1)}


# ---------------------------------------------------------------- build the cards for one day
CONF_PRIOR = 20          # the model counts like this many matches next to the 10-match hit rate (tests: last-10 alone is noisy)
# Lines the model rates above ~80% are priced around 1.20 or less by bookmakers (no value, and often not offered at all), so the
# Top 10 only considers lines in this band; the trend cards themselves still show everything the filters allow.
TOP_MODEL_RANGE = (float(getattr(config, "TOP_MODEL_MIN", 55.0)), float(getattr(config, "TOP_MODEL_MAX", 80.0)))
TOP_PER_FIXTURE = 2


def confidence(card):
    """Model probability and last-10 hit rate combined (model weighs ~2x): (20 * model + hits) / (20 + n)."""
    p = card["model_prob"] / 100.0
    return round(100.0 * (CONF_PRIOR * p + card["hits"]) / (CONF_PRIOR + card["n"]), 1)


def evaluate_card(card, df, store):
    """'won' / 'lost' / None for a card whose fixture has been played (None while unplayed or if the stat is missing)."""
    try:
        m = df[df["event_id"] == card["event_id"]]
        if m.empty:
            return None
        m = m.iloc[0]
        hg, ag = int(m["home_goals"]), int(m["away_goals"])
        if card["market"] == "goals":
            won = (hg + ag > GOAL_LINE) == (card["side"] == "Over")
        elif card["market"] == "btts":
            won = ((hg > 0 and ag > 0) == (card["side"] == "Yes"))
        else:
            if card["event_id"] not in store.index:
                return None
            v = _value(store.loc[card["event_id"]].to_dict(), card["market"], "h" if card["where"] == "H" else "a", card["variant"])
            if v is None or pd.isna(v):
                return None
            won = (v > card["line"]) if card["side"] == "Over" else (v < card["line"])
        return "won" if won else "lost"
    except Exception:
        return None


def top_bets(cards, df, store, finished_ids, n=10):
    """The n most confident trend bets, highest first. Candidates: cards (any hit level) with a model probability in
    TOP_MODEL_RANGE and >= 8 of 10 hits. Duplicates (both teams of a fixture giving the same match-total bet) are merged,
    and at most TOP_PER_FIXTURE bets per fixture are kept so the list is not one match repeated."""
    lo, hi = TOP_MODEL_RANGE
    best = {}
    for c in cards:
        if c.get("model_prob") is None or not (lo <= c["model_prob"] <= hi) or c["hits"] < min(LEVELS):
            continue
        c = dict(c, confidence=confidence(c))
        owner = None if c["variant"] == "total" else c["team"]
        key = (c["event_id"], owner, c["market"], c["variant"], c["side"])
        if key not in best or c["confidence"] > best[key]["confidence"]:
            best[key] = c
    picked, per_fixture = [], {}
    for c in sorted(best.values(), key=lambda c: (-c["confidence"], -c["hits"], -c["model_prob"])):
        if per_fixture.get(c["event_id"], 0) >= TOP_PER_FIXTURE:
            continue
        per_fixture[c["event_id"]] = per_fixture.get(c["event_id"], 0) + 1
        c["fair_odds"] = round(100.0 / c["confidence"], 2)
        c["result"] = evaluate_card(c, df, store) if c["event_id"] in finished_ids else None
        picked.append(c)
        if len(picked) >= n:
            break
    return picked


def build_cards(fixtures):
    """Trend cards for every team in `fixtures` at every hit level (10, 9, 8), plus the Top 10. Returns (cards, info, top)."""
    from model import lambdas_from_fit
    df, store = SD._matches_table(), load_store()
    info = {"needed": 0, "have": 0, "missing": 0, "fetchable": 0}
    if df.empty or not fixtures:
        return [], info, []
    kos = []
    for f in fixtures:
        try:
            kos.append(_ko(f["date"]))
        except Exception:
            pass
    cutoff = min(kos) if kos else pd.Timestamp(datetime.now(timezone.utc))
    models = {k: stat_model(df, store, k, cutoff) for k in MARKETS}
    gdf = _goal_df()
    hist = gdf[gdf["date"] < cutoff]
    grids = {(k, v): line_grid(store, k, v) for k in MARKETS for v in VARIANTS}
    need = set()
    cards = []
    for f in fixtures:
        try:
            ko = _ko(f["date"])
        except Exception:
            continue
        try:
            goal_probs = fixture_goal_probs(hist, f)
        except Exception:
            goal_probs = None
        exp = {}
        for key, mdl in models.items():
            exp[key] = (lambdas_from_fit(mdl["fit"], f["home_id"], f["away_id"], f["league_id"])
                        if mdl and f["home_id"] in mdl["fit"]["tidx"] and f["away_id"] in mdl["fit"]["tidx"] else None)
        for side_name, tid, name, opp_name, where in (("h", f["home_id"], f["home"], f["away"], "H"),
                                                       ("a", f["away_id"], f["away"], f["home"], "A")):
            ms = team_matches(df, store, tid, ko)
            need.update(m["event_id"] for m in ms)
            if len(ms) < WINDOW:
                continue
            if goal_probs is not None:
                cards.extend(goal_cards(f, side_name, name, opp_name, where, ms, hist, goal_probs))
            for key, (label, _field) in MARKETS.items():
                for variant in VARIANTS:
                    series = [(m, _value(m["row"], key, m["who"], variant)) for m in ms]
                    vals = [v for _, v in series if v is not None and not pd.isna(v)]
                    if len(vals) < min(LEVELS):
                        continue
                    for level in LEVELS:
                        if len(vals) < level:
                            continue
                        for sd, (line, hits) in best_lines(vals, grids[(key, variant)], level).items():
                            mu = k = None
                            if exp[key] is not None:
                                mh, ma = exp[key]
                                mu = {"for": mh if side_name == "h" else ma, "against": ma if side_name == "h" else mh,
                                      "total": mh + ma}[variant]
                                k = models[key]["k_total" if variant == "total" else "k_for"]
                            cards.append({
                                "event_id": f["event_id"], "team": name, "opp": opp_name, "where": where,
                                "kickoff": SD.eat_time_of(f["date"]), "league": SD.league_name(f["league_id"]),
                                "market": key, "market_label": label, "variant": variant, "fixed": False, "level": level,
                                "variant_label": VARIANTS[variant], "side": sd, "line": line, "hits": hits, "n": len(vals),
                                "avg": round(float(np.mean(vals)), 1), "median": round(float(np.median(vals)), 1),
                                "expected": round(float(mu), 1) if mu is not None else None,
                                "model_prob": round(p_line(mu, k, line, sd) * 100, 1) if mu is not None else None,
                                "bars": [{"d": m["date"].strftime("%d/%m"), "v": (None if v is None or pd.isna(v) else float(v)),
                                          "ven": m["venue"], "opp": m["opp"],
                                          "hit": (None if v is None or pd.isna(v) else
                                                  bool(v > line if sd == "Over" else v < line))} for m, v in series],
                            })
    have = sum(1 for e in need if e in store.index)
    missing = needed_ids(fixtures)
    info.update(needed=len(need), have=have, missing=len(need) - have, fetchable=len(missing))
    finished = {f["event_id"] for f in fixtures if f.get("status") == "finished"}
    top = top_bets(cards, df, store, finished)
    return cards, info, top


def cards_for_level(cards, level):
    """The cards shown for a selected hit level: stat cards built for exactly that level, goals/BTTS cards with >= level hits."""
    out = [c for c in cards if (c["hits"] >= level) if c.get("fixed")] + [c for c in cards if not c.get("fixed") and c["level"] == level]
    out.sort(key=lambda c: (-c["hits"], -(c["model_prob"] or 0)))
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--fill" in a:
        i = a.index("--fill")
        n = int(a[i + 1]) if len(a) > i + 1 and a[i + 1].isdigit() else 300
        print(upgrade_stats(budget=n, chunk=200))
    else:
        c = coverage()
        print(f"full stats on {c['have']} of {c['matches']} finished matches ({c['pct']}%)")
