# acca_shared.py
"""
Keeps the four daily accas (goals, corners, half-time, BTTS) from picking the same fixture.

A new acca skips any fixture that is already a leg in another market's acca for the same day.
An acca that was skipped / voided frees its fixtures again. Accas that already exist are never changed.

Turn off with  ACCA_UNIQUE_FIXTURES = False  in config.py.
"""
import json
import os

import config

UNIQUE = getattr(config, "ACCA_UNIQUE_FIXTURES", True)

STATE_FILES = ["data/acca_current.json", "data/cacca_current.json",
               "data/hacca_current.json", "data/bacca_current.json"]


def _used_by_others(own_state_file):
    """{local_date: {event_id, ...}} for the other markets' live / settled (not void) accas."""
    used = {}
    own = os.path.normpath(own_state_file)
    for path in STATE_FILES:
        if os.path.normpath(path) == own or not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                s = json.load(f)
        except Exception:
            continue
        if s.get("graded") and s.get("result") == "void":
            continue
        a = s.get("acca") or {}
        day = a.get("date")
        for leg in a.get("legs", []):
            used.setdefault(day, set()).add(int(leg["event_id"]))
    return used


def exclude_used(cands, own_state_file):
    """cands: candidate legs (dicts with 'event_id' and 'date'). Returns them without clashes."""
    if not UNIQUE:
        return cands
    used = _used_by_others(own_state_file)
    return [c for c in cands if int(c["event_id"]) not in used.get(c.get("date"), set())]
