# config.example.py
#
# Copy this file to config.py and fill in your real values.
#   cp config.example.py config.py
#   nano config.py   # paste your Bzzoiro API key
#
# config.py is gitignored — your key stays local.

# ============ BZZOIRO API ============
BZZOIRO_API_KEY  = "YOUR_BZZOIRO_API_KEY_HERE"
BZZOIRO_BASE_URL = "https://sports.bzzoiro.com/api/v2"

# ============================================================
#  TOP 10 LEAGUES — FAST MODE
# ============================================================
TOP_10_LEAGUES = [
    1,   # Premier League (England)
    3,   # La Liga (Spain)
    4,   # Serie A (Italy)
    5,   # Bundesliga (Germany)
    6,   # Ligue 1 (France)
    2,   # Liga Portugal (Portugal)
    10,  # Eredivisie (Netherlands)
    11,  # Süper Lig (Turkey)
    9,   # Brasileirão Série A (Brazil)
    7,   # Champions League (Europe)
]

LEAGUE_NAMES = {
    1:  "Premier League",
    2:  "Liga Portugal",
    3:  "La Liga",
    4:  "Serie A",
    5:  "Bundesliga",
    6:  "Ligue 1",
    7:  "Champions League",
    9:  "Brasileirão Série A",
    10: "Eredivisie",
    11: "Süper Lig",
}

def league_name(lid):
    return LEAGUE_NAMES.get(int(lid), f"League {lid}")

LEAGUES_TO_FETCH  = TOP_10_LEAGUES
DEFAULT_LEAGUE_ID = 1

# ============ PIPELINE BEHAVIOR ============
AUTO_FETCH    = True
FORCE_REFRESH = False

# ============ MODEL SETTINGS ============
RECENT_WEIGHT = 0.6
SEASON_WEIGHT = 0.4
LAST_N_GAMES  = 10

# ============ MODEL CALIBRATION ============
LAMBDA_BOOST    = 1.10
OVER_THRESHOLD  = 0.50

# ============ CONFIDENCE BANDS ============
HIGH_CONF_OVER_PROB   = 0.75
HIGH_CONF_UNDER_PROB  = 0.25

# ============ VALUE BETTING ============
MIN_EV   = 0.03
MAX_EV   = 0.15
MIN_ODDS = 1.30

# ============ PERFORMANCE ============
XG_WORKERS = 8

# ============ DATA PATHS ============
DATA_DIR       = "data"
MATCHES_CSV    = "data/matches.csv"
MATCHES_XG_CSV = "data/matches_with_xg.csv"
XG_CACHE_CSV   = "data/xg_cache.csv"
