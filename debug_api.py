# debug_api.py
import json
import requests
from config import BZZOIRO_API_KEY, BZZOIRO_BASE_URL

HEADERS = {"Authorization": f"Token {BZZOIRO_API_KEY}"}


def show(label, url, params=None):
    print("\n" + "=" * 60)
    print(f"  {label}")
    print(f"  GET {url}")
    if params:
        print(f"  Params: {params}")
    print("=" * 60)
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=30)
        print(f"  Status: {r.status_code}")
        try:
            data = r.json()
            # Print only first 1500 chars of pretty JSON
            txt = json.dumps(data, indent=2, default=str)
            print(txt[:1500])
            if len(txt) > 1500:
                print(f"\n  ... (truncated, total {len(txt)} chars)")
        except Exception:
            print(r.text[:1500])
    except Exception as e:
        print(f"  Request failed: {e}")


# 1. Auth check — hit base leagues endpoint
show("1. List leagues", f"{BZZOIRO_BASE_URL}/leagues/")

# 2. League detail
show("2. League 85 detail", f"{BZZOIRO_BASE_URL}/leagues/85/")

# 3. Season endpoint (current code uses this)
show("3. League 85 season", f"{BZZOIRO_BASE_URL}/leagues/85/season/")

# 4. Seasons endpoint (plural)
show("4. League 85 seasons", f"{BZZOIRO_BASE_URL}/leagues/85/seasons/")

# 5. Events sample for league 85
show("5. Events (league 85, recent)",
     f"{BZZOIRO_BASE_URL}/events/",
     {"league_id": 85, "limit": 3})
