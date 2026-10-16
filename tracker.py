# tracker.py
"""
Bet tracker with auto-settle.

Usage:
    python tracker.py              # full report
    python tracker.py --add        # interactive bet logger (manual)
    python tracker.py --settle     # manual settle
    python tracker.py --auto-settle  # auto-settle from Bzzoiro results
"""
import os
import sys
import requests
import pandas as pd
from datetime import datetime

from config import BZZOIRO_API_KEY, BZZOIRO_BASE_URL

BETS_CSV = "bets_log.csv"
HEADERS = {"Authorization": f"Token {BZZOIRO_API_KEY}"}

COLUMNS = ["date", "event_id", "fixture", "league", "pick",
           "model_prob", "odds", "stake", "result", "pnl", "notes"]


# ----------------------------------------------------------------------
# Load / save
# ----------------------------------------------------------------------
def load_bets():
    if not os.path.exists(BETS_CSV):
        df = pd.DataFrame(columns=COLUMNS)
        df.to_csv(BETS_CSV, index=False)
        return df

    df = pd.read_csv(BETS_CSV)

    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None

    df["event_id"]   = pd.to_numeric(df["event_id"], errors="coerce")
    df["stake"]      = pd.to_numeric(df["stake"], errors="coerce")
    df["odds"]       = pd.to_numeric(df["odds"], errors="coerce")
    df["model_prob"] = pd.to_numeric(df["model_prob"], errors="coerce")
    df["pnl"]        = pd.to_numeric(df["pnl"], errors="coerce")
    df["date"]       = pd.to_datetime(df["date"], errors="coerce")
    df["result"]     = df["result"].astype(str).str.lower().replace("nan", "")
    return df


def save_bets(df):
    df.to_csv(BETS_CSV, index=False)


def auto_pnl(row):
    if pd.notna(row["pnl"]):
        return row["pnl"]
    stake = row["stake"]
    odds  = row["odds"]
    res   = str(row["result"]).lower()
    if pd.isna(stake) or pd.isna(odds):
        return None
    if res == "win":
        return round(stake * (odds - 1), 4)
    if res == "loss":
        return round(-stake, 4)
    if res in ("void", "push"):
        return 0.0
    return None


# ----------------------------------------------------------------------
# Add a single bet (used by Flask /log_bet endpoint)
# ----------------------------------------------------------------------
def add_bet_from_pick(pick_dict, stake=1.0, notes=""):
    """
    Add a bet from a value pick dictionary (from scan.get_todays_matches).
    Returns (success: bool, message: str).
    """
    df = load_bets()

    eid = int(pick_dict["event_id"])

    # Prevent duplicates
    if eid in df["event_id"].values:
        return False, f"Already logged (event_id={eid})"

    new_row = {
        "date":       datetime.now().strftime("%Y-%m-%d"),
        "event_id":   eid,
        "fixture":    f"{pick_dict['home']} vs {pick_dict['away']}",
        "league":     pick_dict["league_name"],
        "pick":       pick_dict["model_pick"],
        "model_prob": round(pick_dict["model_prob"], 4),
        "odds":       round(pick_dict["book_odds"], 3),
        "stake":      float(stake),
        "result":     "",
        "pnl":        None,
        "notes":      notes,
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_bets(df)
    return True, f"Logged: {new_row['fixture']} @ {new_row['odds']}"


# ----------------------------------------------------------------------
# AUTO-SETTLE from Bzzoiro
# ----------------------------------------------------------------------
def fetch_event_result(event_id):
    """Return (home_goals, away_goals, status) or None."""
    try:
        r = requests.get(
            f"{BZZOIRO_BASE_URL}/events/{int(event_id)}/",
            headers=HEADERS, timeout=15,
        )
    except Exception:
        return None
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except Exception:
        return None

    status = data.get("status")
    hg = data.get("home_score")
    ag = data.get("away_score")
    return (hg, ag, status)


def auto_settle(verbose=True):
    """
    Check all pending bets against Bzzoiro results.
    Mark settled bets with win/loss + PnL.
    Returns (n_settled, details_list).
    """
    df = load_bets()
    if len(df) == 0:
        return 0, []

    pending_mask = df["result"].isin(["", "nan"]) | df["result"].isna()
    pending = df[pending_mask]

    if len(pending) == 0:
        return 0, []

    settled_count = 0
    details = []

    for idx, row in pending.iterrows():
        eid = row["event_id"]
        if pd.isna(eid):
            continue

        result = fetch_event_result(int(eid))
        if not result:
            continue

        hg, ag, status = result
        if status != "finished" or hg is None or ag is None:
            continue

        total = hg + ag
        pick = str(row["pick"]).strip()

        if pick == "Over 2.5":
            won = total > 2.5
        elif pick == "Under 2.5":
            won = total < 2.5
        else:
            continue

        df.at[idx, "result"] = "win" if won else "loss"
        df.at[idx, "pnl"] = auto_pnl(df.loc[idx])
        df.at[idx, "notes"] = (
            f"{str(row['notes'] or '').strip()} "
            f"[auto-settled {hg}-{ag}]"
        ).strip()

        settled_count += 1
        details.append({
            "fixture": row["fixture"],
            "pick": pick,
            "score": f"{hg}-{ag}",
            "result": "win" if won else "loss",
            "pnl": float(df.at[idx, "pnl"]),
        })

    if settled_count > 0:
        save_bets(df)

    if verbose and settled_count > 0:
        print(f"\n  Auto-settled {settled_count} bet(s):")
        for d in details:
            icon = "🟢" if d["result"] == "win" else "🔴"
            print(f"  {icon} {d['fixture'][:40]:<42} "
                  f"{d['score']:>5}  {d['result']:<5} {d['pnl']:+.2f}u")

    return settled_count, details


# ----------------------------------------------------------------------
# Manual add (interactive)
# ----------------------------------------------------------------------
def add_bet_interactive():
    print("\n" + "=" * 60)
    print("  ADD NEW BET (manual)")
    print("=" * 60)

    df = load_bets()

    def ask(prompt, default=None):
        val = input(f"  {prompt}" + (f" [{default}]: " if default else ": ")).strip()
        return val if val else (default or "")

    date       = ask("Date (YYYY-MM-DD)", datetime.now().strftime("%Y-%m-%d"))
    event_id   = ask("Event ID (from picks page, or blank)")
    fixture    = ask("Fixture (Home vs Away)")
    league     = ask("League")
    pick       = ask("Pick (Over 2.5 / Under 2.5)")
    model_prob = ask("Model probability (0-1)")
    odds       = ask("Odds (decimal)")
    stake      = ask("Stake (units)", "1.0")
    notes      = ask("Notes", "")

    new_row = {
        "date":       date,
        "event_id":   int(event_id) if event_id else None,
        "fixture":    fixture,
        "league":     league,
        "pick":       pick,
        "model_prob": float(model_prob) if model_prob else None,
        "odds":       float(odds) if odds else None,
        "stake":      float(stake) if stake else 1.0,
        "result":     "",
        "pnl":        None,
        "notes":      notes,
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_bets(df)
    print(f"\n  ✓ Bet added ({len(df)} total)")


# ----------------------------------------------------------------------
# Settle (manual)
# ----------------------------------------------------------------------
def settle_interactive():
    df = load_bets()
    pending_mask = df["result"].isin(["", "nan"]) | df["result"].isna()
    unsettled = df[pending_mask]

    if len(unsettled) == 0:
        print("No unsettled bets.")
        return

    print(f"\n{len(unsettled)} unsettled bets:")
    for idx, row in unsettled.iterrows():
        d = row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else "?"
        print(f"\n  [{idx}] {d}  {row['fixture']}")
        print(f"      Pick: {row['pick']} @ {row['odds']}   Stake: {row['stake']}")

    idx = input("\nEnter row index to settle (or blank to cancel): ").strip()
    if not idx:
        return
    try:
        idx = int(idx)
    except ValueError:
        print("Invalid index.")
        return
    if idx not in df.index:
        print("Row not found.")
        return

    result = input("  Result (win/loss/void): ").strip().lower()
    if result not in ("win", "loss", "void"):
        print("  Invalid result.")
        return

    df.at[idx, "result"] = result
    df.at[idx, "pnl"] = auto_pnl(df.loc[idx])
    save_bets(df)
    print(f"  ✓ Settled row {idx} as {result}")


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
def report():
    df = load_bets()
    df["pnl"] = df.apply(
        lambda r: auto_pnl(r) if pd.isna(r["pnl"]) else r["pnl"], axis=1
    )

    settled = df[df["result"].isin(["win", "loss", "void"])].copy()
    pending = df[~df["result"].isin(["win", "loss", "void"])].copy()

    print("\n" + "=" * 68)
    print("  BETTING TRACKER")
    print("=" * 68)

    if len(df) == 0:
        print("\n  No bets logged yet.")
        return

    print(f"\n  Total bets logged : {len(df)}")
    print(f"  Settled           : {len(settled)}")
    print(f"  Pending           : {len(pending)}")

    if len(settled) == 0:
        print("\n  No settled bets yet.")
        if len(pending) > 0:
            print("\n  PENDING BETS:")
            for _, row in pending.iterrows():
                print(f"    {row['fixture'][:40]:<42} "
                      f"{row['pick']} @ {row['odds']}")
        return

    settled = settled.dropna(subset=["pnl", "stake"])
    wins = (settled["result"] == "win").sum()
    losses = (settled["result"] == "loss").sum()
    voids = (settled["result"] == "void").sum()
    total_stake = settled["stake"].sum()
    total_pnl = settled["pnl"].sum()
    roi = total_pnl / total_stake * 100 if total_stake else 0
    win_rate = wins / (wins + losses) * 100 if (wins + losses) else 0
    avg_odds = settled["odds"].mean()
    avg_stake = settled["stake"].mean()

    print(f"\n  {'RESULT':<15} {'COUNT':>6}")
    print(f"  {'-' * 25}")
    print(f"  {'Wins':<15} {wins:>6}")
    print(f"  {'Losses':<15} {losses:>6}")
    print(f"  {'Voids':<15} {voids:>6}")

    print(f"\n  {'PERFORMANCE':<22} {'VALUE':>15}")
    print(f"  {'-' * 40}")
    print(f"  {'Win rate':<22} {win_rate:>14.1f}%")
    print(f"  {'Avg odds':<22} {avg_odds:>15.2f}")
    print(f"  {'Avg stake':<22} {avg_stake:>15.2f}u")
    print(f"  {'Total staked':<22} {total_stake:>14.2f}u")
    print(f"  {'Total P&L':<22} {total_pnl:>+14.2f}u")

    roi_emoji = "🟢" if roi > 5 else "🟡" if roi > 0 else "🔴"
    print(f"  {'ROI':<22} {roi_emoji} {roi:>+12.1f}%")

    if avg_odds and avg_odds > 1:
        breakeven_wr = 1 / avg_odds * 100
        edge = win_rate - breakeven_wr
        print(f"\n  Break-even win rate : {breakeven_wr:.1f}%")
        print(f"  Your edge           : {edge:+.1f} pts")

    print(f"\n  BY PICK TYPE")
    print(f"  {'-' * 55}")
    print(f"  {'Type':<14} {'Bets':>5} {'Wins':>6} {'WR%':>7} {'P&L':>9} {'ROI%':>8}")
    for pick in sorted(settled["pick"].dropna().unique()):
        sub = settled[settled["pick"] == pick]
        w = (sub["result"] == "win").sum()
        l = (sub["result"] == "loss").sum()
        stk = sub["stake"].sum()
        pnl = sub["pnl"].sum()
        wr = w / (w + l) * 100 if (w + l) else 0
        sroi = pnl / stk * 100 if stk else 0
        print(f"  {pick:<14} {len(sub):>5} {w:>6} {wr:>6.1f}% "
              f"{pnl:>+8.2f}u {sroi:>+7.1f}%")

    if len(pending) > 0:
        print(f"\n  PENDING ({len(pending)}):")
        for _, row in pending.iterrows():
            d = row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else "?"
            print(f"    {d}  {str(row['fixture'])[:40]:<42} "
                  f"{row['pick']} @ {row['odds']}")

    print("\n" + "=" * 68 + "\n")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if "--add" in sys.argv:
        add_bet_interactive()
    elif "--settle" in sys.argv:
        settle_interactive()
    elif "--auto-settle" in sys.argv:
        n, details = auto_settle(verbose=True)
        if n == 0:
            print("Nothing to settle.")
    else:
        # Default: auto-settle first, then report
        auto_settle(verbose=True)
        report()
