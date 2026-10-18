# repair_dates.py
"""
One-off repair: restores the dates that were blanked out of matches.csv and
matches_with_xg.csv by the mixed-date-format bug, so those matches show up
again in Results and count in the model.

    python repair_dates.py            # needs internet (re-reads the affected seasons from the API)
    python repair_dates.py --dry-run  # show what would be fixed, change nothing
"""
import argparse
import shutil

import pandas as pd

from config import MATCHES_CSV, MATCHES_XG_CSV
from fetch_data import fetch_matches


def parse(col):
    return pd.to_datetime(col, errors="coerce", utc=True, format="ISO8601")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    m = pd.read_csv(MATCHES_CSV)
    m["date"] = parse(m["date"])
    bad = m[m["date"].isna()]
    print(f"{len(bad)} matches have no date in {MATCHES_CSV}")
    if bad.empty:
        return

    # Re-read each affected (league, season) from the API to recover the original dates
    id_to_date = {}
    pairs = sorted({(int(r.league_id), int(r.season_id)) for r in bad.itertuples()})
    for lid, sid in pairs:
        rows = fetch_matches(lid, sid)
        print(f"  league {lid} season {sid}: {len(rows)} matches fetched")
        for r in rows:
            d = pd.to_datetime(r["date"], errors="coerce", utc=True, format="ISO8601")
            if pd.notna(d):
                id_to_date[int(r["event_id"])] = d

    fixable = [int(e) for e in bad["event_id"] if int(e) in id_to_date]
    missing = [int(e) for e in bad["event_id"] if int(e) not in id_to_date]
    print(f"\nRecoverable: {len(fixable)}   still unknown: {len(missing)}")
    if missing:
        print("  Could not recover (API has no date for them):", missing)
    if args.dry_run or not fixable:
        return

    for path in (MATCHES_CSV, MATCHES_XG_CSV):
        shutil.copy(path, path + ".bak")
        df = pd.read_csv(path)
        df["date"] = parse(df["date"])
        fill = df["date"].isna() & df["event_id"].astype(int).isin(id_to_date)
        df.loc[fill, "date"] = df.loc[fill, "event_id"].astype(int).map(id_to_date)
        df = df.sort_values("date", na_position="last").reset_index(drop=True)
        df.to_csv(path, index=False)
        print(f"  {path}: restored {int(fill.sum())} dates (backup: {path}.bak)")

    print("\nDone. Restart the app: pkill -f app.py && python app.py")


if __name__ == "__main__":
    main()
