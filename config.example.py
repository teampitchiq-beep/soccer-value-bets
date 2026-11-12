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

# ============ SHOTS TAB (optional) ============
# SHOTS_DAILY_BUDGET = 1000   # shots requests per day (0 = off)
# SHOTS_WINDOW       = 10     # last N home / away matches averaged
# SHOTS_STAT_KEYS    = []     # extra API field names for total shots, if the default names are not found
# SHOTS_HALF_LIFE    = 180.0  # expected-shots model: recency half-life in days
# SHOTS_RIDGE        = 40.0   # expected-shots model: shrinkage toward the league average (shots are ~9x more numerous than goals)
# SHOTS_AVG_DECAY    = 1.0    # recency weight inside the last-10 Avg: 1.0 = plain average, 0.9 gentle, 0.8 strong (see shots_tests.py)
# SHOTS_BLEND_WEIGHT = 0.2    # ON by default: shots-based expected goals blended into Over/Under and BTTS. 0 = off (pure goals model)

# ============ TRENDS TAB (optional) ============
# STATS_DAILY_BUDGET = 1000   # match-stats requests per UTC day for the Trends tab (0 = off)
# STATS_HALF_LIFE    = 180.0  # recency half-life (days) of the per-stat expected-value model
# TOP_MODEL_MIN = 55.0; TOP_MODEL_MAX = 80.0   # Top 10 bets only considers lines the model rates within this % band

# ============ 1X2 (MATCH RESULT) (optional) ============
# DRAW_BOOST             = 0.05   # multiplies the probability of drawn scorelines (0 = plain Poisson)
# X12_HIGH_PROB          = 0.60   # confidence tiers on the most likely outcome
# X12_MEDIUM_PROB        = 0.50
# ACCA_MIN_LEG_PROB_1X2  = 0.45   # minimum model probability per leg for the 1X2 acca market

# ============ NATIONAL TEAMS (UEFA Nations League, World Cup qualifiers ...) ============
# National teams play only a handful of matches a year, so scan.py asks them for NATIONAL_MIN_APPS appearances instead of 10
# (otherwise their fixtures would be skipped). `python add_leagues.py --preset c` adds the UEFA Nations League and fills this list in.
NATIONAL_TEAM_LEAGUES = []
# NATIONAL_MIN_APPS = 4

# ============ BET OF THE DAY (optional) ============
# BOTD_MIN_ODDS = 1.40   # real odds range a selection must be in
# BOTD_MAX_ODDS = 1.90
# BOTD_MIN_PROB = 0.0    # optional floor on the model probability (e.g. 0.55)

# ============ ACCA TRACKING (optional) ============
# ACCA_DEFAULT_STAKE = 1.0   # units staked on every tracked acca (editable per acca)
# ACCA_STREAK_MIN    = 2     # a streak is this many identical results in a row (minimum 2)
