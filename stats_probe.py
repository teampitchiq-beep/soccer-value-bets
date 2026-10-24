# stats_probe.py
"""
Checks which match statistics Bzzoiro returns for FINISHED matches - in particular
CORNERS for the home and away team. Read-only: nothing in your data is changed.

    python stats_probe.py              # ~5 matches per league, spread over time
    python stats_probe.py --per-league 10

The app already calls /events/{id}/stats/ to get xG; this shows what ELSE that
response contains (corners, shots, possession ...) and for which leagues/months.
Needs internet.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from config import BZZOIRO_API_KEY, BZZOIRO_BASE_URL, MATCHES_XG_CSV, league_name

HEADERS = {"Authorization": f"Token {BZZOIRO_API_KEY}"}


def fetch_stats(event_id):
    try:
        r = requests.get(f"{BZZOIRO_BASE_URL}/events/{event_id}/stats/", headers=HEADERS, timeout=20)
    except Exception as e:
        return event_id, None, f"request failed: {type(e).__name__}"
    if r.status_code != 200:
        return event_id, None, f"HTTP {r.status_code}"
    try:
        return event_id, r.json(), None
    except Exception:
        return event_id, None, "bad JSON"


def num(v):
    try:
        f = float(v)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-league", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(MATCHES_XG_CSV)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True, format="ISO8601")
    df = df.dropna(subset=["date"]).sort_values("date")

    picks = []
    for lid, g in df.groupby("league_id"):
        idx = [int(i * (len(g) - 1) / max(args.per_league - 1, 1)) for i in range(min(args.per_league, len(g)))]
        picks.append(g.iloc[sorted(set(idx))])
    sample = pd.concat(picks)
    print(f"Probing {len(sample)} finished matches across {sample['league_id'].nunique()} leagues ...\n")

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda e: fetch_stats(int(e)), sample["event_id"]))
    time.sleep(0.1)

    errors, rows, key_cover, example = {}, [], {}, None
    meta = sample.set_index("event_id")
    for eid, data, err in results:
        if err:
            errors[err] = errors.get(err, 0) + 1
            continue
        stats = (data or {}).get("stats") or {}
        home, away = stats.get("home") or {}, stats.get("away") or {}
        if example is None and home:
            example = data
        for side, block in (("home", home), ("away", away)):
            for k, v in block.items():
                key_cover.setdefault(k, {"home": 0, "away": 0})
                if v is not None:
                    key_cover[k][side] += 1
        m = meta.loc[eid]
        rows.append({"event_id": eid, "league": league_name(int(m["league_id"])),
                     "month": m["date"].strftime("%Y-%m"), "home": home, "away": away})

    ok = len(rows)
    print(f"Responses OK: {ok}/{len(sample)}" + (f"   Errors: {errors}" if errors else ""))
    with_stats = [r for r in rows if r["home"] or r["away"]]
    print(f"With a stats block: {len(with_stats)}/{ok}\n")
    if not with_stats:
        print("The stats endpoint returned no statistics for the sampled matches.")
        return

    print("All statistics returned (matches with a value, home / away):")
    for k, c in sorted(key_cover.items(), key=lambda kv: -(kv[1]["home"] + kv[1]["away"])):
        flag = "   <-- corners" if "corner" in k.lower() else ""
        print(f"  {k:<32}{c['home']:>4} / {c['away']:<4}{flag}")

    ckeys = [k for k in key_cover if "corner" in k.lower()]
    print()
    if not ckeys:
        print("RESULT: no corner statistics in this response.")
        print("(Other stats above are available; if any look useful tell me and I can store them.)")
        return

    key = ckeys[0]
    print(f"RESULT: corners ARE returned, under the key '{key}'"
          + (f" (other corner keys: {ckeys[1:]})" if len(ckeys) > 1 else ""))
    vals = []
    for r in with_stats:
        h, a = num(r["home"].get(key)), num(r["away"].get(key))
        vals.append({"league": r["league"], "month": r["month"], "h": h, "a": a})
    v = pd.DataFrame(vals)
    both = v.dropna(subset=["h", "a"])
    print(f"Matches with home AND away corners: {len(both)}/{len(sample)}")
    if len(both):
        print(f"Average per match: home {both['h'].mean():.1f}, away {both['a'].mean():.1f}, "
              f"total {(both['h'] + both['a']).mean():.1f}")

    print("\nCoverage by league (with corners / sampled):")
    tot = pd.DataFrame(rows).groupby("league").size()
    got = both.groupby("league").size()
    for lg in tot.index:
        print(f"  {lg:<22}{int(got.get(lg, 0))}/{int(tot[lg])}")
    print("\nCoverage by month (with corners / sampled):")
    totm = pd.DataFrame(rows).groupby("month").size()
    gotm = both.groupby("month").size()
    for mo in totm.index:
        print(f"  {mo}: {int(gotm.get(mo, 0))}/{int(totm[mo])}")

    print("\nExample values (home / away corners):")
    for r in both.head(5).itertuples():
        print(f"  {r.league:<20}{r.month}   {r.h:.0f} / {r.a:.0f}")
    if example:
        print("\nRaw example of one stats block (truncated):")
        print(json.dumps(example, indent=1, default=str)[:900])


if __name__ == "__main__":
    main()
