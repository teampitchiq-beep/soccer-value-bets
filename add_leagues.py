# add_leagues.py
"""
Adds leagues to your project (config.py + downloaded history + xG).

    python add_leagues.py --add "54:Eliteserien:Norway" --add "15:Swiss Super League:Switzerland"
    python add_leagues.py                       # no arguments: looks up Europa League + Conference League
    python add_leagues.py --europa 8 --conference 12   # Europa/Conference with IDs you already know
    python add_leagues.py --add "54:Eliteserien:Norway" --no-fetch   # only edit config.py

--add format is  ID:Name[:Country].  Find IDs with:  python fetch_data.py leagues <Country>

Add by NAME (the script looks the IDs up for you, one API request):
    python add_leagues.py --preset a              # batch A: Championship, Serie B, MLS, Liga MX, 2. Bundesliga
    python add_leagues.py --preset b              # batch B: Belgian Pro League, Brasileirão Série B, Argentina,
                                                  #          Saudi Pro League, Danish Superliga
    python add_leagues.py --name "Serie B|Italy"  # any single league: "name|country"
    python add_leagues.py --preset a --dry-run    # only show which IDs it found, change nothing

Each batch downloads ~2,500-3,000 matches and fetches xG for each (1 API request per match), so run
batch A and batch B on DIFFERENT days to stay inside the free daily quota (resets 00:00 UTC). The xG
download is resumable: if the quota runs out, run the same command again after the reset.

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

# Chosen from Bzzoiro's coverage table: "Full" tier = most played matches carry xG AND shot maps
# (Argentina is "Advanced": no xG, the model falls back to actual goals for those matches).
# (search text, country, name shown in the app)
PRESETS = {
    "a": [("Championship", "England", "Championship"),
          ("Serie B", "Italy", "Serie B"),
          ("MLS", "USA", "MLS"),
          ("Liga MX", "Mexico", "Liga MX"),
          ("2. Bundesliga", "Germany", "2. Bundesliga")],
    "b": [("Pro League", "Belgium", "Belgian Pro League"),
          ("Serie B", "Brazil", "Brasileirão Série B"),
          ("Liga Profesional", "Argentina", "Liga Profesional"),
          ("Saudi Pro League", "Saudi Arabia", "Saudi Pro League"),
          ("Superliga", "Denmark", "Danish Superliga")],
}
BAD_WORDS = ("women", "femenin", "feminin", " w ", "u21", "u20", "u19", "u18", "youth", "reserve", "cup", "copa", "pokal")


def _norm(text):
    import unicodedata
    t = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]+", " ", t).strip()


def resolve_names(specs, leagues):
    """specs: [(search, country, display)]. Returns [(id, display, country)]; asks only if ambiguous."""
    out = []
    for search, country, display in specs:
        want, ctry = _norm(search), _norm(country)
        pool = [lg for lg in leagues
                if ctry in (_norm(lg.get("country", "")), _norm(lg.get("country", "")).replace("united states", "usa"))
                or (ctry == "usa" and _norm(lg.get("country", "")) in ("usa", "united states", "us"))]
        if display == "Brasileirão Série B":
            pool = [lg for lg in pool if "b" in _norm(lg.get("name", "")).split()]
        elif display == "Serie B":
            pool = [lg for lg in pool if "b" in _norm(lg.get("name", "")).split()]
        cands = [lg for lg in pool
                 if want in _norm(lg.get("name", ""))
                 and not any(b in f" {_norm(lg.get('name', ''))} " for b in BAD_WORDS)]
        exact = [lg for lg in cands if _norm(lg.get("name", "")) == want]
        cands = exact or cands
        if not cands:
            print(f"  ! Could not find '{search}' ({country}) in the API list. "
                  f"Try:  python fetch_data.py leagues \"{country}\"  then use --add ID:Name:Country")
            continue
        if len(cands) > 1:
            print(f"\nSeveral matches for {display} ({country}):")
            for i, c in enumerate(cands, 1):
                print(f"  {i}. id={c.get('id')}  name={c.get('name')!r}")
            pick = input(f"Which one? [1-{len(cands)}, Enter = skip] ").strip()
            if not (pick.isdigit() and 1 <= int(pick) <= len(cands)):
                print("  skipped")
                continue
            c = cands[int(pick) - 1]
        else:
            c = cands[0]
        print(f"  {display}: id={c['id']}  ({c.get('name')} / {c.get('country')})")
        out.append((int(c["id"]), display, country))
    return out


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
    ap.add_argument("--preset", choices=sorted(PRESETS), action="append", default=[],
                    help="add a ready-made batch of 5 leagues by name (a or b)")
    ap.add_argument("--name", action="append", default=[], metavar="NAME|COUNTRY",
                    help='add a league by name, e.g. --name "Serie B|Italy" (repeatable)')
    ap.add_argument("--dry-run", action="store_true", help="only show the IDs found, change nothing")
    args = ap.parse_args()

    entries = [parse_add(t) for t in args.add]

    by_name = [spec for key in args.preset for spec in PRESETS[key]]
    for text in args.name:
        parts = [p.strip() for p in text.split("|")]
        if len(parts) != 2 or not all(parts):
            raise SystemExit(f'Bad --name value {text!r}. Use "NAME|COUNTRY", e.g. "Serie B|Italy"')
        by_name.append((parts[0], parts[1], parts[0]))
    if by_name:
        from fetch_data import get_leagues
        print("Looking up league IDs via the API (1-2 requests) ...")
        leagues = get_leagues(None)
        if not leagues:
            print("Could not fetch the league list (offline or quota used up?). Retry later.")
            return
        entries += resolve_names(by_name, leagues)
        if args.dry_run:
            print("\nDry run: nothing changed.")
            return
        if not entries:
            print("Nothing resolved.")
            return

    explicit_europe = bool(args.europa or args.conference)
    if explicit_europe or not (args.add or by_name):   # nothing given at all => original Europa/Conference mode
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
