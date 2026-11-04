# stats_probe.py
"""
What does the stats endpoint (/events/{id}/stats/) actually return?  Read-only: never changes your data or cache.

    python stats_probe.py EVENT_ID          every field for ONE match (home and away), plus any other sections
                                            of the response (e.g. players / incidents) with their shape
    python stats_probe.py --scan [N]        samples N recent finished matches spread over your leagues (default 40,
                                            one API request each) and reports, per stat field, how often it is present,
                                            how often it is numeric, its typical value and how many leagues have it
                                            -> saved to data/stats_probe_report.json

Use it to decide which markets a Trends tab can cover (shots, shots against, corners, fouls, cards, saves ...).
"""
import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

import config
from config import BZZOIRO_API_KEY, BZZOIRO_BASE_URL, DATA_DIR, MATCHES_CSV, league_name

HEADERS = {"Authorization": f"Token {BZZOIRO_API_KEY}"}
REPORT = os.path.join(DATA_DIR, "stats_probe_report.json")
WORKERS = getattr(config, "XG_WORKERS", 8)
KEYWORDS = ["shot", "corner", "foul", "throw", "card", "yellow", "red", "save", "offside", "possess", "pass", "tackle",
            "cross", "clearance", "intercept", "xg", "expected", "dribble", "duel", "block", "goal_kick", "penalt"]


def fetch(event_id):
    """(status, json_or_None). status: 'ok' | 'none' | 'quota' | 'error'."""
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
        return "ok", r.json()
    except Exception:
        return "error", None


def as_number(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return None if (isinstance(v, float) and math.isnan(v)) else float(v)
    if isinstance(v, str):
        try:
            return float(v.replace("%", "").strip())
        except ValueError:
            return None
    return None


def flatten(obj, prefix=""):
    """{dotted.path: value} for every scalar inside nested dicts (lists are summarised by their length)."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        out[prefix + "[]"] = f"list({len(obj)})"
    else:
        out[prefix] = obj
    return out


def describe_other_sections(data):
    """Top-level parts of the response besides stats: what they are and their shape (players, incidents, lineups ...)."""
    lines = []
    for k, v in (data or {}).items():
        if k == "stats":
            continue
        if isinstance(v, list):
            first = v[0] if v else None
            shape = (f"keys of first item: {', '.join(list(first.keys())[:14])}" if isinstance(first, dict)
                     else (f"first item: {first!r}" if first is not None else "empty"))
            lines.append(f"  {k}: list of {len(v)} | {shape}")
        elif isinstance(v, dict):
            lines.append(f"  {k}: object with keys {', '.join(list(v.keys())[:14])}{' ...' if len(v) > 14 else ''}")
        else:
            lines.append(f"  {k}: {v!r}")
    return lines


def one_match(event_id):
    st, data = fetch(event_id)
    print(f"event {event_id}: {st}")
    if st != "ok":
        if st == "quota":
            print("API quota used up: try again after 03:00 Nairobi time.")
        return
    stats = (data or {}).get("stats") or {}
    print(f"\nTop-level sections: {', '.join(data.keys())}")
    side_keys = {}
    for side in ("home", "away"):
        side_keys[side] = flatten(stats.get(side) or {})
    names = sorted(set(side_keys["home"]) | set(side_keys["away"]))
    print(f"\nstats: {len(names)} fields per side (home | away)")
    for n in names:
        mark = "  <--" if any(w in n.lower() for w in KEYWORDS) else ""
        print(f"  {n:<42} {str(side_keys['home'].get(n, '-')):>14} | {str(side_keys['away'].get(n, '-')):<14}{mark}")
    others = describe_other_sections(data)
    if others:
        print("\nOther sections in the response:")
        print("\n".join(others))
    print("\n(<-- marks fields that look like shots / corners / fouls / cards / saves / xG ...)")


def pick_matches(n):
    df = pd.read_csv(MATCHES_CSV)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True, format="ISO8601")
    df = df.dropna(subset=["date"]).sort_values("date", ascending=False)
    df = df[df["date"] >= df["date"].max() - pd.Timedelta(days=75)]
    leagues = df["league_id"].unique()
    per = max(1, math.ceil(n / max(len(leagues), 1)))
    picks = df.groupby("league_id").head(per).sort_values("date", ascending=False).head(n)
    return [(int(r.event_id), int(r.league_id)) for r in picks.itertuples()]


def scan(n):
    picks = pick_matches(n)
    print(f"Sampling {len(picks)} recent matches over {len({l for _, l in picks})} leagues ...")
    results, quota = [], False

    def work(item):
        if quota:
            return item, "skipped", None
        return (item,) + fetch(item[0])

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for item, st, data in pool.map(work, picks):
            if st == "quota":
                quota = True
            results.append((item, st, data))
    ok = [(it, d) for it, st, d in results if st == "ok"]
    print(f"  got stats for {len(ok)} of {len(picks)}"
          + ("  (API quota ran out part-way: results are partial)" if quota else ""))
    if not ok:
        print("Nothing to report. Check the API key / quota.")
        return

    agg = defaultdict(lambda: {"present": 0, "numeric": 0, "vals": [], "leagues": set()})
    for (eid, lid), data in ok:
        stats = (data or {}).get("stats") or {}
        seen = set()
        for side in ("home", "away"):
            for path, v in flatten(stats.get(side) or {}).items():
                a = agg[path]
                num = as_number(v)
                a["vals"].append(num) if num is not None else None
                a["numeric"] += num is not None
                a["leagues"].add(lid)
                seen.add(path)
        for path in seen:
            agg[path]["present"] += 1
    rows = []
    for path, a in agg.items():
        vals = a["vals"]
        rows.append({"field": path, "matches_with_field": a["present"], "pct_of_sample": round(100 * a["present"] / len(ok)),
                     "numeric_pct": round(100 * a["numeric"] / max(2 * a["present"], 1)),
                     "mean": round(sum(vals) / len(vals), 2) if vals else None,
                     "min": min(vals) if vals else None, "max": max(vals) if vals else None,
                     "leagues": len(a["leagues"])})
    rows.sort(key=lambda r: (-r["pct_of_sample"], r["field"]))
    print(f"\n{'stat field':<42}{'in %':>6}{'num %':>7}{'mean':>9}{'min':>7}{'max':>7}{'leagues':>9}")
    for r in rows:
        mark = "  <--" if any(w in r["field"].lower() for w in KEYWORDS) else ""
        print(f"{r['field']:<42}{r['pct_of_sample']:>6}{r['numeric_pct']:>7}{str(r['mean']):>9}{str(r['min']):>7}"
              f"{str(r['max']):>7}{r['leagues']:>9}{mark}")
    other = describe_other_sections(ok[0][1])
    if other:
        print("\nOther sections in a response (first sampled match):")
        print("\n".join(other))
    by_lg = defaultdict(lambda: [0, 0])
    for (eid, lid), st, _ in results:
        by_lg[lid][1] += 1
        by_lg[lid][0] += st == "ok"
    print("\nStats returned per league (ok / sampled): " +
          ", ".join(f"{league_name(l)} {o}/{t}" for l, (o, t) in sorted(by_lg.items())))
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump({"sampled": len(picks), "with_stats": len(ok), "fields": rows}, f, indent=1)
    print(f"\nSaved {REPORT}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        print(__doc__)
    elif a[0] == "--scan":
        scan(int(a[1]) if len(a) > 1 and a[1].isdigit() else 40)
    else:
        one_match(int(a[0]))
