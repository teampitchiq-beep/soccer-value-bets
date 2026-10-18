# scan.py
"""
Value bet scanner + full-fixture scanner.
CLI:
    python scan.py                 # value bets only
    python scan.py --all-matches   # every upcoming match with predictions
"""
import sys
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    BZZOIRO_API_KEY, BZZOIRO_BASE_URL, LEAGUES_TO_FETCH,
    HIGH_CONF_OVER_PROB, HIGH_CONF_UNDER_PROB, LAMBDA_BOOST,
    MIN_EV, MAX_EV, MIN_ODDS, league_name,
)
from model import load_data, predict_lambdas, over_under_2_5

HEADERS = {"Authorization": f"Token {BZZOIRO_API_KEY}"}
DEFAULT_MIN_APPS = 15
ODDS_WORKERS = 6


# ----------------------------------------------------------------------
# Fetch upcoming fixtures
# ----------------------------------------------------------------------
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
                break

            data = r.json()
            results = data.get("results", [])
            if not results:
                break

            for m in results:
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

    return upcoming


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
        return None
    try:
        odds = (r.json() or {}).get("odds", {}) or {}
    except Exception:
        return None
    return {
        "over_25":  odds.get("over_25_goals"),
        "under_25": odds.get("under_25_goals"),
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


def expected_value(prob, decimal_odds):
    if decimal_odds is None or decimal_odds <= 1.0:
        return None
    return prob * decimal_odds - 1.0


# ----------------------------------------------------------------------
# Main data function used by Flask
# ----------------------------------------------------------------------
def get_todays_matches(days=2, min_apps=DEFAULT_MIN_APPS):
    """
    Returns dict with:
      - matches: ALL upcoming matches with predictions (list of dicts)
      - value_bets: subset of matches with positive EV within [MIN_EV, MAX_EV]
      - stats: counts
    """
    df = load_data()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    appearance_counts = build_appearance_index(df)

    upcoming = fetch_upcoming(days=days)

    # Pre-filter out no-history matches
    candidates = []
    skipped_no_history = 0
    for m in upcoming:
        hk = (int(m["league_id"]), int(m["home_team_id"]))
        ak = (int(m["league_id"]), int(m["away_team_id"]))
        ha = appearance_counts.get(hk, 0)
        aa = appearance_counts.get(ak, 0)
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

        # Confidence tier
        if po >= HIGH_CONF_OVER_PROB or po <= HIGH_CONF_UNDER_PROB:
            tier = "HIGH"
        elif 0.60 <= po < HIGH_CONF_OVER_PROB or HIGH_CONF_UNDER_PROB < po <= 0.40:
            tier = "MEDIUM"
        else:
            tier = "LOW"

        # Top 2 scores
        top_2 = []
        for (h, a), p in (top[:2] if top else []):
            top_2.append({"score": f"{h}-{a}", "prob": round(p * 100, 1)})

        odds = odds_map.get(m["event_id"]) or {}
        over_o = odds.get("over_25")
        under_o = odds.get("under_25")

        # EV calc
        ev = None
        book_odds = None
        model_pick = None
        model_prob = None
        if over_o is not None or under_o is not None:
            if po >= pu and over_o:
                model_pick, model_prob, book_odds = "Over 2.5", po, over_o
            elif pu > po and under_o:
                model_pick, model_prob, book_odds = "Under 2.5", pu, under_o
            if book_odds and book_odds >= MIN_ODDS:
                ev = expected_value(model_prob, book_odds)

        is_value = (tier == "HIGH"
                    and ev is not None
                    and MIN_EV <= ev <= MAX_EV)

        matches.append({
            "event_id":     m["event_id"],
            "date":         m["date"],
            "league_id":    m["league_id"],
            "league_name":  league_name(m["league_id"]),
            "home":         m["home_team"],
            "away":         m["away_team"],
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
            "model_pick":   model_pick,
            "model_prob":   model_prob,
            "book_odds":    book_odds,
            "ev":           ev,
            "is_value":     is_value,
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

    # Value bets remain sorted by EV (highest edge first)
    value_bets = [m for m in matches if m["is_value"]]
    value_bets.sort(key=lambda x: x["ev"], reverse=True)

    return {
        "matches": matches,
        "value_bets": value_bets,
        "stats": {
            "total_upcoming": len(upcoming),
            "with_history": len(candidates),
            "skipped_no_history": skipped_no_history,
            "value_bets_count": len(value_bets),
        },
    }


def get_value_picks(days=2, min_apps=DEFAULT_MIN_APPS,
                    min_ev=MIN_EV, max_ev=MAX_EV):
    """Backwards-compatible wrapper — returns value bets only."""
    result = get_todays_matches(days=days, min_apps=min_apps)
    return [m for m in result["value_bets"]
            if min_ev <= (m.get("ev") or 0) <= max_ev]


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    days = 2
    show_all = "--all-matches" in sys.argv
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
    print(f"  Skipped (no data): {s['skipped_no_history']}")
    print(f"  Value bets:        {s['value_bets_count']}\n")

    if show_all:
        for m in result["matches"]:
            icon = "🟢" if m["po"] >= 50 else "🔵"
            print(f"{icon}  {m['date'][:16]}  [{m['league_name']}]")
            print(f"     {m['home']} vs {m['away']}")
            print(f"     Over 2.5: {m['po']}%   Under: {m['pu']}%   "
                  f"Confidence: {m['tier']}")
            print(f"     xG: {m['lh']} – {m['la']}")
            print(f"     Top scores: "
                  f"{', '.join([s['score'] + ' (' + str(s['prob']) + '%)' for s in m['top_2_scores']])}")
            if m["ev"] is not None:
                print(f"     EV: {m['ev']*100:+.1f}%  @ {m['book_odds']:.2f}")
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
            print(f"     Edge: {p['ev']*100:+.1f}%   "
                  f"Model: {p['model_prob']*100:.1f}%")
            print(f"     xG: {p['lh']} – {p['la']}")
            print(f"     Top: "
                  f"{', '.join([s['score'] + ' (' + str(s['prob']) + '%)' for s in p['top_2_scores']])}")
            print()


if __name__ == "__main__":
    main()
