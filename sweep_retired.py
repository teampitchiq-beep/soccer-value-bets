#!/usr/bin/env python3
"""
sweep_retired.py - one-time cleanup after the model was cut down to Over/Under 2.5 and BTTS only.

    python sweep_retired.py            # DRY RUN: shows what would happen, changes nothing
    python sweep_retired.py --apply    # does it

What --apply does (nothing is deleted):
  1. MOVES the retired code and data files into  ~/retired_markets/<timestamp>/  (same folder layout),
     so you can move them back if you ever want them.
  2. Removes the retired settings blocks from config.py and config.example.py. Each file is backed up first
     as <name>.before_sweep. The new file is checked to be valid Python and to keep every other setting;
     if anything looks wrong it is NOT written.

Makes no API requests. When you are happy with the result you can delete this script.
"""
import ast
import os
import re
import shutil
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

RETIRED_CODE = [
    "markets_model.py", "markets_backtest.py",
    "corners_model.py", "corners_data.py", "corners_backtest.py",
    "ht_model.py", "ht_data.py", "ht_backtest.py",
    "stats_probe.py",
]
RETIRED_DATA = [
    "data/corners.csv", "data/ht_goals.csv", "data/corner_pred_history.csv", "data/ht_pred_history.csv",
    "data/markets_pred_history.csv", "data/markets_lambdas.csv", "data/odds_log_markets.csv",
]
CONFIG_FILES = ["config.py", "config.example.py"]
# a settings block (lines separated from the next by a blank line) is retired when any line matches this
RETIRED_SETTING = re.compile(r"corner|first-half|\bHT_|\bHT\b|CACCA", re.IGNORECASE)


def assigned_names(src):
    names = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
    return names


def split_blocks(text):
    """Group lines into blocks separated by blank lines, keeping the blank lines with the block before them."""
    blocks, cur = [], []
    for line in text.splitlines(keepends=True):
        cur.append(line)
        if not line.strip():
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    return blocks


def clean_config(path, apply):
    if not os.path.exists(path):
        return "not found"
    src = open(path, encoding="utf-8").read()
    kept, removed = [], []
    for block in split_blocks(src):
        (removed if any(RETIRED_SETTING.search(l) for l in block) else kept).append(block)
    if not removed:
        return "nothing to remove"
    new = "".join("".join(b) for b in kept)
    try:
        before, after = assigned_names(src), assigned_names(new)
    except SyntaxError as e:
        return f"SKIPPED: {e}"
    lost = sorted(n for n in before - after if not RETIRED_SETTING.search(n))
    if lost:
        return f"SKIPPED: would also lose non-retired settings {lost}"
    gone = sorted(before - after)
    if apply:
        shutil.copy2(path, path + ".before_sweep")
        with open(path, "w", encoding="utf-8") as f:
            f.write(new)
    return f"{'removed' if apply else 'would remove'} {len(removed)} block(s): {', '.join(gone) if gone else 'comments only'}"


def main():
    apply = "--apply" in sys.argv
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_root = os.path.join(os.path.expanduser("~"), "retired_markets", stamp)
    print(("APPLYING" if apply else "DRY RUN (nothing changes; add --apply to do it)") + f"  ·  folder: {HERE}\n")

    moved = 0
    for rel in RETIRED_CODE + RETIRED_DATA:
        if os.path.exists(rel):
            moved += 1
            print(f"  move  {rel}  ->  {dest_root}/{rel}")
            if apply:
                target = os.path.join(dest_root, rel)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.move(rel, target)
    if not moved:
        print("  (no retired files found: already clean)")

    print()
    for cfg in CONFIG_FILES:
        print(f"  {cfg}: {clean_config(cfg, apply)}")

    print()
    if apply:
        if moved:
            print(f"Done. Retired files are in {dest_root}")
        print("Config backups are saved as config.py.before_sweep / config.example.py.before_sweep.\n"
              "Restart app.py. You can now delete sweep_retired.py.")
    else:
        print("This was a dry run. Run again with --apply to do it.")


if __name__ == "__main__":
    main()
