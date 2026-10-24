# add_leagues.py
"""
Adds leagues to your project (config.py + downloaded history + xG).

    python add_leagues.py --add "54:Eliteserien:Norway" --add "15:Swiss Super League:Switzerland"
    python add_leagues.py                       # no arguments: looks up Europa League + Conference League
    python add_leagues.py --europa 8 --conference 12   # Europa/Conference with IDs you already know
    python add_leagues.py --add "54:Eliteserien:Norway" --no-fetch   # only edit config.py

--add format is  ID:Name[:Country].  Find IDs with:  python fetch_data.py leagues <Country>

What it does:
  1. (no-argument mode) looks the Europa/Conference leagues up in the API list
  2. Adds each league to TOP_10_LEAGUES and LEAGUE_NAMES in config.py (backup: config.py.bak)
  3. Downloads current + previous season, merges into data/matches.csv
  4. Fetches xG for the new matches only (existing xG is cached)

Safe to re-run: leagues already in config.py are not duplicated.
Needs internet. Afterwards restart the app:  pkill -f app.py ; python app.py
"""
import argparse
import re
import shutil

import pandas as pd

CONFIG = "config.py"
WANTED = {
    "europa":     {"label": "Europa League",     "must": "europa league",     "not": ("conference", "women")},
    "conference": {"label": "Conference League", "must": "conference league", "not": ("women",)},
}
EUROPE_WORDS = ("europe", "world", "international", "uefa", "")


def find_candidates(leagues, key):
    spec = WANTED[key]
    out = []
    for lg in leagues:
        name = str(lg.get("name", "")).lower()
        if spec["must"] in name and not any(bad in name for bad in spec["not"]):
            out.append(lg)
    return out


def choose(cands, key):
    """Pick one candidate; ask the user only if it is genuinely ambiguous."""
    label = WANTED[key]["label"]
    if not cands:
        return None
    pan_eu = [c for c in cands if str(c.get("country", "")).lower() in EUROPE_WORDS]
    pool = pan_eu or cands
    if len(pool) == 1:
        return pool[0]
    print(f"\nSeveral matches for {label}:")
    for i, c in enumerate(pool, 1):
        print(f"  {i}. id={c.get('id')}  country={c.get('country')!r}  name={c.get('name')!r}")
    while True:
        pick = input(f"Which one is the main {label}? [1-{len(pool)}] ").strip()
        if pick.isdigit() and 1 <= int(pick) <= len(pool):
            return pool[int(pick) - 1]


def update_config(entries, path=CONFIG):
    """entries: list of (league_id, display_name, region). Idempotent. Returns ids added."""
    s = open(path, encoding="utf-8").read()
    shutil.copy(path, path + ".bak")
    added = []

    for lid, name, region in entries:
        m = re.search(r"(TOP_10_LEAGUES\s*=\s*\[)(.*?)(\n\s*\])", s, re.S)
        if not m:
            raise RuntimeError("Could not find TOP_10_LEAGUES in config.py")
        existing = [int(x) for x in re.findall(r"^\s*(\d+)\s*,", m.group(2), re.M)]
        if lid not in existing:
            s = s[:m.end(2)] + f"\n    {lid},  # {name} ({region})" + s[m.end(2):]
            added.append(lid)

        m = re.search(r"(LEAGUE_NAMES\s*=\s*\{)(.*?)(\n\s*\})", s, re.S)
        if not m:
            raise RuntimeError("Could not find LEAGUE_NAMES in config.py")
        if not re.search(rf"^\s*{lid}\s*:", m.group(2), re.M):
            s = s[:m.end(2)] + f'\n    {lid}: "{name}",' + s[m.end(2):]

    open(path, "w", encoding="utf-8").write(s)
    return added


def download(league_ids):
    from fetch_data import fetch_two_seasons, enrich_with_xg
    from config import MATCHES_CSV

    frames = []
    for lid in league_ids:
        print(f"\nFetching league {lid} ...")
        df = fetch_two_seasons(lid)
        if df is not None and not df.empty:
            frames.append(df)
        else:
            print(f"  No data returned for league {lid}")
    if not frames:
        print("Nothing downloaded.")
        return

    old = pd.read_csv(MATCHES_CSV)
    old["date"] = pd.to_datetime(old["date"], errors="coerce", utc=True, format="ISO8601")
    combined = pd.concat([old] + frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce", utc=True, format="ISO8601")
    combined = (combined.drop_duplicates(subset=["event_id"], keep="last")
                        .sort_values("date").reset_index(drop=True))
    combined.to_csv(MATCHES_CSV, index=False)
    print(f"\nmatches.csv now has {len(combined)} matches "
          f"(+{len(combined) - len(old)} new)")

    print("\nFetching xG for new matches (cached ones are skipped) ...")
    enrich_with_xg(combined, progress=True)


def parse_add(text):
    parts = text.split(":")
    if len(parts) < 2 or not parts[0].strip().isdigit():
        raise SystemExit(f'Bad --add value {text!r}. Use ID:Name[:Country], e.g. "54:Eliteserien:Norway"')
    lid = int(parts[0])
    name = parts[1].strip()
    region = parts[2].strip() if len(parts) > 2 and parts[2].strip() else "added"
    return lid, name, region


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", action="append", default=[], metavar="ID:Name[:Country]",
                    help='add any league, e.g. --add "54:Eliteserien:Norway" (repeatable)')
    ap.add_argument("--europa", type=int, help="Europa League id (skips the lookup)")
    ap.add_argument("--conference", type=int, help="Conference League id (skips the lookup)")
    ap.add_argument("--no-fetch", action="store_true", help="only edit config.py")
    args = ap.parse_args()

    entries = [parse_add(t) for t in args.add]

    explicit_europe = bool(args.europa or args.conference)
    if explicit_europe or not args.add:       # no --add at all => original Europa/Conference mode
        chosen = {}
        if args.europa:
            chosen["europa"] = args.europa
        if args.conference:
            chosen["conference"] = args.conference
        # look up only when no Europa/Conference id was passed explicitly
        missing = [k for k in WANTED if k not in chosen] if not explicit_europe else []
        if missing:
            from fetch_data import get_leagues
            print("Looking up leagues via the API ...")
            leagues = get_leagues(None)
            if not leagues:
                print("Could not fetch the league list (offline?). Connect and retry, "
                      "or pass --europa ID --conference ID.")
                return
            for key in missing:
                cands = find_candidates(leagues, key)
                if not cands:
                    print(f"No league matching '{WANTED[key]['must']}' found. "
                          f"Check `python fetch_data.py leagues` and pass --{key} ID.")
                    continue
                c = choose(cands, key)
                if c:
                    chosen[key] = int(c["id"])
                    print(f"  {WANTED[key]['label']}: id={c['id']}  ({c.get('name')} / {c.get('country')})")
        entries += [(lid, WANTED[k]["label"], "Europe") for k, lid in chosen.items()]

    if not entries:
        print("Nothing to add.")
        return

    added = update_config(entries)
    print(f"\nconfig.py updated (backup: config.py.bak). Newly added ids: {added or 'none (already present)'}")

    if args.no_fetch:
        print("Skipping download (--no-fetch). Run again without it to download data.")
        return
    download([lid for lid, _, _ in entries])
    print("\nDone. Restart the app:  pkill -f app.py ; python app.py")


if __name__ == "__main__":
    main()
