# scan.py
"""
Full-fixture scanner: predictions for every upcoming match.
CLI:
    python scan.py                 # every upcoming match with predictions
    python scan.py --days 3        # look further ahead
"""
import sys
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    BZZOIRO_API_KEY, BZZOIRO_BASE_URL, LEAGUES_TO_FETCH,
    HIGH_CONF_OVER_PROB, HIGH_CONF_UNDER_PROB, LAMBDA_BOOST,
    league_name,
)
from model import load_data, predict_lambdas, over_under_2_5, classify_confidence

HEADERS = {"Authorization": f"Token {BZZOIRO_API_KEY}"}
DEFAULT_MIN_APPS = 10   # total appearances across ALL leagues (was 15 within one league)
ODDS_WORKERS = 6


# ----------------------------------------------------------------------
# Fetch upcoming fixtures
# ----------------------------------------------------------------------
FETCH_ERRORS = []          # filled by the fetchers; read by get_todays_matches (e.g. "429 quota")


def _note_error(what, status):
    FETCH_ERRORS.append(f"{what}: HTTP {status}")


def fetch_upcoming(days=2):
    now = datetime.now(timezone.utc)
    end = now + timedelta(days=days)
    date_from = now.strftime("%Y-%m-%d")
    date_to = end.strftime("%Y-%m-%d")

    upcoming = []
    for lid in LEAGUES_TO_FETCH:
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
                        "status": "notstarted",
                        "limit": limit,
                        "offset": offset,
                    },
                    timeout=30,
                )
            except Exception:
                break
            if r.status_code != 200:
                _note_error(f"fixtures league {lid}", r.status_code)
                break

            data = r.json()
            results = data.get("results", [])
            if not results:
                break

            for m in results:
                if not m.get("home_team_id") or not m.get("away_team_id"):
                    continue                      # teams not decided yet (e.g. qualifier TBD)
                upcoming.append({
                    "event_id":     m.get("id"),
                    "league_id":    m.get("league_id"),
                    "date":         m.get("event_date"),
                    "home_team":    m.get("home_team"),
                    "home_team_id": m.get("home_team_id"),
                    "away_team":    m.get("away_team"),
                    "away_team_id": m.get("away_team_id"),
                })

            if not data.get("next"):
                break
            offset += limit

    seen, unique = set(), []
    for m in upcoming:                            # same event can come back twice
        if m["event_id"] in seen:
            continue
        seen.add(m["event_id"])
        unique.append(m)
    return unique


# ----------------------------------------------------------------------
# Fetch odds
# ----------------------------------------------------------------------
def fetch_event_odds(event_id):
    try:
        r = requests.get(
            f"{BZZOIRO_BASE_URL}/events/{event_id}/odds/",
            headers=HEADERS, timeout=15,
        )
    except Exception:
        return None
    if r.status_code != 200:
        if r.status_code != 404:                  # 404 = this fixture simply has no odds
            _note_error(f"odds event {event_id}", r.status_code)
        return None
    try:
        odds = (r.json() or {}).get("odds", {}) or {}
    except Exception:
        return None
    return {
        "over_25":  odds.get("over_25_goals"),
        "under_25": odds.get("under_25_goals"),
        "btts_yes": odds.get("btts_yes"),
        "btts_no":  odds.get("btts_no"),
    }


def fetch_odds_batch(event_ids):
    """Fetch odds for many events in parallel. Returns dict {eid: odds_or_None}."""
    out = {}
    if not event_ids:
        return out
    with ThreadPoolExecutor(max_workers=ODDS_WORKERS) as pool:
        futures = {pool.submit(fetch_event_odds, eid): eid for eid in event_ids}
        for fut in as_completed(futures):
            eid = futures[fut]
            try:
                out[eid] = fut.result()
            except Exception:
                out[eid] = None
    return out


# ----------------------------------------------------------------------
# Appearance index (cup filter)
# ----------------------------------------------------------------------
def build_appearance_index(df):
    counts = {}
    for _, r in df.iterrows():
        hk = (int(r["league_id"]), int(r["home_team_id"]))
        ak = (int(r["league_id"]), int(r["away_team_id"]))
        counts[hk] = counts.get(hk, 0) + 1
        counts[ak] = counts.get(ak, 0) + 1
    return counts


# ----------------------------------------------------------------------
# Main data function used by Flask
# ----------------------------------------------------------------------
def get_todays_matches(days=2, min_apps=DEFAULT_MIN_APPS):
    """
    Returns dict with:
      - matches: ALL upcoming matches with predictions (list of dicts)
      - stats: counts
    """
    df = load_data()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    # Appearances are counted across ALL leagues: the model keeps one rating per team
    # (domestic games inform Champions League predictions), so a CL team with only a few
    # CL games but a full domestic season has plenty of data. The old per-league count of
    # 15 silently dropped ~90% of Champions League fixtures.
    totals = (df["home_team_id"].value_counts()
              .add(df["away_team_id"].value_counts(), fill_value=0))
    totals = {int(k): int(v) for k, v in totals.items()}

    del FETCH_ERRORS[:]
    upcoming = fetch_upcoming(days=days)

    # Pre-filter out no-history matches
    candidates = []
    skipped_no_history = 0
    for m in upcoming:
        try:
            ha = totals.get(int(m["home_team_id"]), 0)
            aa = totals.get(int(m["away_team_id"]), 0)
        except (TypeError, ValueError):
            skipped_no_history += 1
            continue
        if ha < min_apps or aa < min_apps:
            skipped_no_history += 1
            continue
        m["home_apps"] = ha
        m["away_apps"] = aa
        candidates.append(m)

    # Fetch odds in parallel for all candidates
    odds_map = fetch_odds_batch([m["event_id"] for m in candidates])

    matches = []
    for m in candidates:
        try:
            lh, la = predict_lambdas(
                df, m["home_team_id"], m["away_team_id"], m["league_id"]
            )
        except Exception:
            continue

        lh *= LAMBDA_BOOST
        la *= LAMBDA_BOOST
        po, pu, top = over_under_2_5(lh, la)

        # Confidence tier (same thresholds as the Single Match tab: model.classify_confidence)
        tier = classify_confidence(po)[0]

        # Top 2 scores
        top_2 = []
        for (h, a), p in (top[:2] if top else []):
            top_2.append({"score": f"{h}-{a}", "prob": round(p * 100, 1)})

        odds = odds_map.get(m["event_id"]) or {}
        over_o = odds.get("over_25")
        under_o = odds.get("under_25")

        # The model's preferred side and the bookmaker price for it
        book_odds = None
        model_pick = None
        model_prob = None
        if po >= pu and over_o:
            model_pick, model_prob, book_odds = "Over 2.5", po, over_o
        elif pu > po and under_o:
            model_pick, model_prob, book_odds = "Under 2.5", pu, under_o

        matches.append({
            "event_id":     m["event_id"],
            "date":         m["date"],
            "league_id":    m["league_id"],
            "league_name":  league_name(m["league_id"]),
            "home":         m["home_team"],
            "away":         m["away_team"],
            "home_id":      int(m["home_team_id"]),
            "away_id":      int(m["away_team_id"]),
            "home_apps":    m["home_apps"],
            "away_apps":    m["away_apps"],
            "lh":           round(lh, 2),
            "la":           round(la, 2),
            "po":           round(po * 100, 1),
            "pu":           round(pu * 100, 1),
            "top_2_scores": top_2,
            "tier":         tier,
            "over_odds":    over_o,
            "under_odds":   under_o,
            "btts_yes_odds": odds.get("btts_yes"),
            "btts_no_odds":  odds.get("btts_no"),
            "model_pick":   model_pick,
            "model_prob":   model_prob,
            "book_odds":    book_odds,
        })

    # Save pre-match odds + model probability for later backtesting
    try:
        from odds_logger import log_snapshot
        log_snapshot(matches)
    except Exception as e:
        print(f"  (odds logging skipped: {e})")

    # ------------------------------------------------------------------
    # Sort All Matches by Over 2.5 probability (highest first)
    # ------------------------------------------------------------------
    matches.sort(key=lambda x: x["po"], reverse=True)

    return {
        "matches": matches,
        "stats": {
            "total_upcoming": len(upcoming),
            "with_history": len(candidates),
            "skipped_no_history": skipped_no_history,
            "fetch_errors": list(FETCH_ERRORS),
            "quota_hit": any("429" in e for e in FETCH_ERRORS),
            # odds calls failed for some fixtures: today's list is incomplete, do not freeze it
            "partial": any(e.startswith("odds") for e in FETCH_ERRORS),
        },
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    days = 2
    if "--days" in sys.argv:
        try:
            days = int(sys.argv[sys.argv.index("--days") + 1])
        except (IndexError, ValueError):
            pass

    result = get_todays_matches(days=days)
    s = result["stats"]

    print("=" * 78)
    print(f"  SCANNER  —  next {days} days")
    print("=" * 78)
    print(f"  Upcoming fixtures: {s['total_upcoming']}")
    print(f"  With history:      {s['with_history']}")
    print(f"  Skipped (no data): {s['skipped_no_history']}\n")

    for m in result["matches"]:
        icon = "🟢" if m["po"] >= 50 else "🔵"
        print(f"{icon}  {m['date'][:16]}  [{m['league_name']}]")
        print(f"     {m['home']} vs {m['away']}")
        print(f"     Over 2.5: {m['po']}%   Under: {m['pu']}%   "
              f"Confidence: {m['tier']}")
        print(f"     xG: {m['lh']} – {m['la']}")
        print(f"     Top scores: "
              f"{', '.join([x['score'] + ' (' + str(x['prob']) + '%)' for x in m['top_2_scores']])}")
        if m["over_odds"] and m["under_odds"]:
            print(f"     Odds (O/U): {m['over_odds']:.2f} / {m['under_odds']:.2f}")
        print()


if __name__ == "__main__":
    main()
