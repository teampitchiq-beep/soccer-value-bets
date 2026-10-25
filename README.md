# ⚽ PitchIQ Predictions — Goals & Corners Predictor

Ridge-regularised Poisson model with xG-weighted attack/defence ratings for
**Over/Under 2.5 goals**, **Over/Under corners**, **first-half Over/Under 1.5 goals** and **Both Teams To Score (BTTS)**, with daily accumulator picks that
log themselves. Built entirely in Termux on a phone.

## What it does

- Fetches match data across 14 leagues from the **Bzzoiro Sports API**
- Enriches every match with **expected goals (xG)**
- Predicts **Over/Under 2.5 goals** and the **top 2 most likely correct scores** for every fixture
- Predicts **corners** (expected home / away / total, Over/Under the line, default 8.5)
- Predicts **first-half (half-time) goals**: expected HT goals and Over/Under 1.5
- Predicts **BTTS Yes / No** from the same expected goals, with real bookmaker BTTS odds
- Builds **one goals acca, one corners acca, one half-time Under 1.5 acca and one BTTS Yes acca per day** and **logs them automatically**
- Tracks results, auto-settles accas and bets, and reports ROI / hit rate
- Shows all times in **Nairobi time (EAT, UTC+3)**

## Tabs

| Tab | What it shows |
|---|---|
| 📅 **All Matches** | Upcoming fixtures with O/U 2.5 %, odds, xG, top score, confidence tier. Date buttons along the top |
| 🎯 **2-Odds Acca** | Today's goals acca (frozen until graded). Skip / void buttons |
| 🚩 **Corners** | Corner predictions per fixture |
| 🚩 **Corners Acca** | Today's corners acca |
| ⏱ **HT O/U 1.5** | First-half goals predictions, highest probability first |
| ⏱ **HT Acca** | Today's half-time acca: **Under 1.5 HT legs only**, same rules as the other accas |
| ⚽ **BTTS** | Both Teams To Score Yes/No, highest probability first, with bookmaker odds and value (EV) |
| ⚽ **BTTS Acca** | Today's BTTS Yes acca at **real bookmaker odds**, same rules as the goals acca |
| 📋 **Results** | Finished matches (last 3 / 7 / 14 / 30 days): predicted vs actual for **Over/Under 2.5 + correct scores, BTTS, first-half Over/Under 1.5 + HT correct scores, and corners Over/Under**, with hit-rate summaries |
| 📆 **Date** | Pick any day (Prev / Next / Today): goals picks, corners picks and results for that day |
| 🔍 **Single Match** | Manual prediction lookup |

Separate pages: `/acca-tracker` (goals accas), `/corners-tracker` (corners accas), `/ht-tracker` (half-time accas), `/btts-tracker` (BTTS accas), `/tracker` (bet tracker).

The **Date tab** reads rows already loaded on the page, so it makes no API calls.
It can only show days the app has loaded: about 2 days of upcoming fixtures, plus
the results range chosen on the Results tab.

## Leagues Covered

Premier League · La Liga · Serie A · Bundesliga · Ligue 1 · Liga Portugal ·
Eredivisie · Süper Lig · Brasileirão Série A · Champions League · Europa League ·
Conference League · Eliteserien · Swiss Super League

Add more with `python add_leagues.py --add ID:Name[:Country]`.

## Requirements

- Termux (or any Linux/macOS environment with Python 3.10+)
- Python packages: `flask`, `requests`, `pandas`, `numpy`, `scipy`
- API key from [Bzzoiro Sports Data](https://sports.bzzoiro.com)

## Setup

### 1. Install dependencies

```bash
pkg update && pkg upgrade -y
pkg install python git nano -y
pip install flask requests pandas numpy scipy
```

### 2. Configure your API key

```bash
cp config.example.py config.py
nano config.py   # paste your Bzzoiro key into BZZOIRO_API_KEY
```

### 3. Run

```bash
python app.py
```

First run fetches the match history and enriches it with xG (~10–20 minutes).
Later runs start in seconds. Open `http://127.0.0.1:5000` in your browser.

## Daily Workflow

```bash
cd ~/over_model
python app.py
```

Then just open the page. Fixtures, picks and both accas are ready, and the accas are
already logged. Check **Results** and the trackers later to see how they did.

## API usage — loads once a day

The free Bzzoiro plan has a **daily request quota that resets at 00:00 UTC (03:00 Nairobi)**.
The app is built to spend very little of it:

| What | How often | Notes |
|---|---|---|
| Fixtures + odds + picks | **Once per UTC day** | Saved to `data/picks_cache.json`, reused on every page view and restart |
| Recent finished matches | **Once per UTC day** | Catches up from the newest match on disk (3–30 days). Also skipped at launch if already done today |
| Corner counts for new matches | **Once per UTC day** | Up to 400 requests |
| Half-time scores for new matches | **Once per UTC day** | Up to `HT_DAILY_MAX` (500) requests |
| ↻ Refresh button | Only when you tap it | Forces a fresh fetch of fixtures and odds |
| Auto-settle | Every 15 min | Only calls the API when you have pending bets/accas |

- The full 2-season download only runs when there is **no data on disk** (or `FORCE_REFRESH = True`).
  A partial download never overwrites your existing `matches.csv`.
- Which daily jobs already ran is stored in `data/daily_jobs.json`.
- An empty result (e.g. quota hit) is never saved, and the app retries at most every 30 minutes.
- Odds are a snapshot from the first load of the day. Tap **↻ Refresh** for fresh odds.
- If the page shows **0 matches**, check the quota: a `429 taster_exhausted` reply means
  the daily limit is used up. Wait for 03:00 Nairobi time, then launch or refresh.

## First-half goals (HT Over/Under 1.5)

- **Data:** half-time scores come from the API (`home_score_ht` / `away_score_ht`) into `data/ht_goals.csv`.
  Run `python ht_data.py` once to backfill. It first tries the cheap league-list endpoint (about 40 requests);
  if the list rows have no half-time score it falls back to one request per match and stops cleanly
  when the daily quota runs out. The app then adds new matches once a day (up to `HT_DAILY_MAX`, default 500).
- **Model:** the same ridge Poisson engine as goals and corners, fitted on half-time goals.
  Total HT goals is Poisson, so P(Over 1.5) = P(2 or more).
- **Odds:** the free API plan has **no half-time odds**. The HT acca uses **model-implied odds** (1 / probability),
  like the corners acca. Enter your bookmaker's combined odds when logging if you want profit tracked.
- **Validate before trusting it:** run `python ht_backtest.py`. It prints whether the model beats the plain
  base rate and the best `HT_SCALE` / `HT_CAL_A` / `HT_CAL_B` to put in `config.py`.
- **HT acca rules:** same as the other accas but **Under 1.5 HT legs only** (`HACCA_UNDER_ONLY = False` allows Over too).
  Override with `HACCA_MIN_ODDS`, `HACCA_MAX_ODDS`, `HACCA_MAX_LEGS`, `HACCA_MIN_LEG_PROB`.

## BTTS (Both Teams To Score)

- **Model:** `P(BTTS Yes) = (1 − e^−λhome) × (1 − e^−λaway)`, using the same expected goals as Over/Under 2.5.
  No extra data or API calls: BTTS odds (`btts_yes` / `btts_no`) arrive with the odds request already made for each fixture.
- **Backtest** (`python btts_backtest.py`, walk-forward on your data): well calibrated; BTTS Yes picks at 58%+ hit about 62%,
  65%+ about 72%. The edge over the plain base rate (56.5% BTTS) is small, so it is a lean, not a certainty.
- **BTTS acca:** BTTS Yes legs only, **real bookmaker odds**, same rules as the goals acca
  (one per day, 2–3 legs, combined odds 2.0–3.0, each leg ≥ 58%). A leg wins when both teams score in 90 minutes.
  BTTS Yes prices are often 1.7–1.9, so two legs can land above the 3.0 cap: raise `BACCA_MAX_ODDS` in `config.py`
  if the acca is often empty. Other overrides: `BACCA_MIN_ODDS`, `BACCA_MAX_LEGS`, `BACCA_MIN_LEG_PROB`.
- **Where it shows:** its own tab, the Acca tab and `/btts-tracker`, the Single Match card, the Date tab and the Results tab.
- Picks saved before this update have no BTTS odds: tap **↻ Refresh** once to load them.

## Auto-logging accas

All four accas (goals, corners, half-time, BTTS) are logged automatically the moment they are generated, with no button tap.
A background job checks every 15 minutes (it works even if the page is closed), and
opening the page also triggers it. An acca is never logged twice.

Settings are at the top of `app.py`:

```python
AUTO_LOG_ACCAS = True    # False = log by hand with the button
AUTO_LOG_STAKE = 1.0     # units staked on each auto-logged acca
```

- The **Skip** and **Void** buttons still work (e.g. a postponed match).
- The API has no corner odds, so corners accas log at **model-implied odds**. Profit for
  corners is therefore based on the model's odds, not your bookmaker's.
- Auto-logging records every acca the model generates, including ones you don't place.
  If you only want to track real bets, set `AUTO_LOG_ACCAS = False`.

### Acca rules (both markets)

- One acca per day, today's matches only, Over legs only
- 2–3 legs, combined odds 2.0–3.0, each leg at least 58 % model probability
- **No fixture appears in two accas** (goals, corners, half-time, BTTS) on the same day: a new acca skips fixtures already used by another market's acca. A skipped / voided acca frees its fixtures. Turn off with `ACCA_UNIQUE_FIXTURES = False` in `config.py`
- A new acca is generated only after the previous one is graded (won / lost / void), and **never twice in one day once an acca was won or lost**. Skipping or voiding one (e.g. a postponed match) still frees the slot. Turn this off with `ACCA_ONE_PER_DAY = False` in `config.py` (or per market: `CACCA_`, `HACCA_`, `BACCA_ONE_PER_DAY`)
- An ungraded acca is voided automatically 48 h after its last kickoff

Override in `config.py`: `ACCA_MIN_ODDS`, `ACCA_MAX_ODDS`, `ACCA_MAX_LEGS`, `ACCA_MIN_LEG_PROB`
(goals) and the `CACCA_*` equivalents (corners).
"Today" follows your phone's timezone. To force Nairobi, set `ACCA_UTC_OFFSET_HOURS = 3`.

## Time zone

Everything on screen is shown in **EAT (Nairobi, UTC+3)**: kickoff times, the Today / Tomorrow
buttons, day grouping and the Results tab. Matches are grouped by Nairobi date, so a
21:30 UTC kickoff appears as 00:30 on the next day. Stored data and acca logic stay in UTC.
`acca_tracker.html` and `corners_tracker.html` still label times as UTC.

## CLI Commands

| Command | Description |
|---|---|
| `python app.py` | Daily run (fetch if stale, serve UI) |
| `python app.py --no-refresh` | Skip the startup fetch, just serve |
| `python app.py --scan` | Print today's picks, exit |
| `python app.py --all-matches` | Print all upcoming matches, exit |
| `python app.py --status` | Show data freshness and file info |
| `python app.py --auto-settle` | Settle pending bets, exit |
| `python acca.py` | Preview today's goals acca (`--grade`, `--settle`, `--report`) |
| `python corners_data.py` | Fetch corner counts (`--limit N`, `--status`) |
| `python ht_data.py` | Fetch half-time scores (`--limit N`, `--status`) |
| `python ht_backtest.py` | Backtest the first-half model, get calibration values |
| `python btts_backtest.py` | Backtest BTTS, get calibration values |
| `python corners_backtest.py` | Backtest the corners model (`--tune` to tune) |
| `python tune.py` | Grid-search LAMBDA_BOOST and threshold |
| `python backtest.py` | Confidence-band accuracy report |
| `python tracker.py` | Full ROI report (`--add`, `--settle`, `--auto-settle`) |
| `python odds_logger.py` | Log odds snapshots (`--watch`, `--every N`) |
| `python add_leagues.py` | Add leagues to `config.py` and fetch them |

## Project Structure

```
over_model/
├── app.py                  # Flask app, daily pipeline, daily cache, auto-log
├── config.example.py       # Template config (safe to commit)
├── config.py               # Your config with API key (gitignored)
├── fetch_data.py           # Bzzoiro fetch + xG enrichment
├── model.py                # Goals model (ridge Poisson)
├── scan.py                 # Upcoming fixtures + picks + odds
├── acca.py                 # Goals acca: build, freeze, grade, log
├── acca_shared.py          # Keeps the four accas from sharing fixtures
├── corners_data.py         # Corner counts fetcher
├── corners_model.py        # Corners model (Negative Binomial)
├── corners_backtest.py     # Corners backtest / tuning
├── corner_acca.py          # Corners acca: build, freeze, grade, log
├── ht_data.py              # Half-time score fetcher
├── ht_model.py             # First-half goals model
├── ht_acca.py              # Half-time Under 1.5 acca
├── ht_backtest.py          # First-half backtest / calibration
├── btts_model.py           # BTTS model
├── btts_acca.py            # BTTS Yes acca (real odds)
├── btts_backtest.py        # BTTS backtest / calibration
├── results_extra.py        # HT, corners, BTTS predictions shown on the Results tab
├── predictions_history.py  # Stored predictions for the Results tab
├── tracker.py              # Bet tracker with auto-settle
├── tune.py / backtest.py   # Goals calibration and verification
├── odds_logger.py          # Odds snapshots
├── add_leagues.py          # League manager
├── requirements.txt
├── static/theme.css
├── templates/
│   ├── index.html          # Main dashboard (all tabs)
│   ├── acca_tracker.html   # Goals acca tracker
│   ├── corners_tracker.html# Corners acca tracker
│   ├── ht_tracker.html     # Half-time acca tracker
│   └── btts_tracker.html   # BTTS acca tracker
└── data/                   # Auto-created
    ├── matches.csv, matches_with_xg.csv, xg_cache.csv
    ├── corners.csv         # Corner counts
    ├── ht_goals.csv        # Half-time scores
    ├── ht_pred_history.csv, corner_pred_history.csv, btts_pred_history.csv  # Cached point-in-time predictions for Results
    ├── picks_cache.json    # Today's fixtures + picks (once-a-day cache)
    ├── daily_jobs.json     # Which daily jobs already ran
    ├── acca_log.csv, cacca_log.csv, hacca_log.csv, bacca_log.csv          # Logged accas
    └── acca_current.json, cacca_current.json, hacca_current.json, bacca_current.json # Today's frozen accas
```

## How the Model Works

For each match:

1. Compute league averages of home/away xG
2. Compute each team's attack & defence strength relative to league average
   (time-decayed, ridge-regularised, one rating per team across all leagues)
3. Apply multiplicative Poisson:
   ```
   λ_home = home_attack × away_defence × league_avg_home_xg × LAMBDA_BOOST
   λ_away = away_attack × home_defence × league_avg_away_xg × LAMBDA_BOOST
   ```
4. Compute the probability of every scoreline from 0-0 to 10-10
5. Sum to get Over/Under 2.5 probabilities
6. Rank scorelines to find the top 2 most likely correct scores

**Corners** use the same engine fitted on corner counts. Totals are more spread out
than a pure Poisson, so the Over/Under probability uses a Negative Binomial with a
dispersion factor estimated from the data.

**Value detection:** compare model probability vs bookmaker implied
probability (1 / decimal odds). If `model_prob × odds − 1 > 3%` and the
pick is high-confidence, flag as a value bet.

## Disclaimer

This is a personal research tool, not financial advice. Sports betting
carries risk of loss. Never bet more than you can afford to lose. Historical
backtest performance does not guarantee future results. Bet responsibly.

## License

MIT
