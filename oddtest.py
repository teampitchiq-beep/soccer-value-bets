import requests
from config import BZZOIRO_API_KEY, BZZOIRO_BASE_URL

HEADERS = {"Authorization": f"Token {BZZOIRO_API_KEY}"}
# Using a known event ID from your previous tests
EVENT_ID = 209590 

url = f"{BZZOIRO_BASE_URL}/events/{EVENT_ID}/odds/"
response = requests.get(url, headers=HEADERS)

if response.status_code == 200:
    print(response.json())
else:
    print(f"Error: {response.status_code} - {response.text}")
