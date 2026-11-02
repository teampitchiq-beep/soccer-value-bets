#!/usr/bin/env python3
"""quota_check.py - show Bzzoiro API quota status (one request per run).

Usage:
    python quota_check.py          # print status
    python quota_check.py --log    # also append a row to data/quota_log.csv

Run it before and after an action (app launch, fetch, etc.). The drop in
'remaining' between two runs is what that action cost. The probe itself
may cost 1 request. The API key is never printed.
"""
import csv
import os
import re
import sys
from datetime import datetime, timezone, timedelta

import requests

import config

LOG_FILE = os.path.join("data", "quota_log.csv")
EAT = timezone(timedelta(hours=3))


def parse_ratelimit(headers):
    """Parse '"football";r=4964;t=20912' -> (remaining, seconds_to_reset)."""
    raw = headers.get("ratelimit", "")
    r = re.search(r"\br=(\d+)", raw)
    t = re.search(r"\bt=(\d+)", raw)
    return (int(r.group(1)) if r else None, int(t.group(1)) if t else None)


def parse_policy_quota(headers):
    """Parse '"football";q=7500;w=86400' -> 7500."""
    m = re.search(r"\bq=(\d+)", headers.get("ratelimit-policy", ""))
    return int(m.group(1)) if m else None


def fmt_hms(seconds):
    if seconds is None:
        return "unknown"
    h, rem = divmod(int(seconds), 3600)
    return f"{h}h {rem // 60:02d}m"


def main():
    try:
        resp = requests.get(
            config.BZZOIRO_BASE_URL + "/leagues/",
            headers={"Authorization": f"Token {config.BZZOIRO_API_KEY}"},
            timeout=20,
        )
    except requests.RequestException as exc:
        print(f"Network error: {exc}")
        return 1

    remaining, reset_s = parse_ratelimit(resp.headers)
    policy_q = parse_policy_quota(resp.headers)
    now = datetime.now(timezone.utc)

    print(f"HTTP status        : {resp.status_code}")
    print(f"Header remaining   : {remaining}")
    print(f"Header policy cap  : {policy_q}")
    if remaining is not None and policy_q:
        print(f"Used (by header)   : {policy_q - remaining}")
    print(f"Resets in          : {fmt_hms(reset_s)}")
    if reset_s is not None:
        reset_at = (now + timedelta(seconds=reset_s)).astimezone(EAT)
        print(f"Resets at (EAT)    : {reset_at:%Y-%m-%d %H:%M}")
    if resp.headers.get("retry-after"):
        print(f"Retry-After        : {resp.headers['retry-after']}s")
    if resp.status_code == 429:
        try:
            body = resp.json()
            print(f"API says           : {body.get('code')} - quota blocked")
        except ValueError:
            print("API says           : 429 (non-JSON body)")

    if "--log" in sys.argv:
        os.makedirs("data", exist_ok=True)
        new_file = not os.path.exists(LOG_FILE)
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new_file:
                w.writerow(["utc_time", "status", "remaining", "policy_cap"])
            w.writerow([now.strftime("%Y-%m-%d %H:%M:%S"), resp.status_code,
                        remaining, policy_q])
        print(f"Logged to {LOG_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
