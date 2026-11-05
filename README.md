# ⚽ PitchIQ Predictions — Over/Under 2.5 and BTTS

Ridge-regularised Poisson model with xG-weighted attack/defence ratings. From one pair of
expected-goals numbers per fixture it predicts exactly **two markets**:

- **Over/Under 2.5 goals** (the most likely scorelines are shown alongside as supporting detail)
- **Both Teams To Score (BTTS)** Yes / No

An **Acca builder** then combines the most probable selections of one of these two markets, using real bookmaker odds only.

Runs from the terminal on Kali Linux (any Linux with Python 3.10+ works). Data comes from the
**Bzzoiro Sports API**.

> Research tool, not betting advice. See [Disclaimer](#disclaimer).

## What it does

- Fetches match data for 19 competitions (20 league IDs) and enriches every match with **expected goals (xG)**
- Predicts every upcoming fixture once per day and saves the picks, so page views and restarts cost no API requests
- Shows bookmaker odds next to the model's probability, plus the **expected value (EV)** of each selection
- Grades its own past predictions on the Results tab, with **separate hit rates for each side it picked** (Over vs Under, Yes vs No)
- Shows all times in **Nairobi time (EAT, UTC+3)**

## Tabs

| Tab | What it shows |
|---|---|
| 📅 **All Matches** | Upcoming fixtures: Over/Under 2.5 %, odds, xG, likely scorelines, confidence tier. Expand a row for BTTS as well |
| 🎯 **Acca** | Pick a day, a market (Over/Under 2.5 or BTTS) and a side; builds the most probable acca of 2 or 3 matches with combined odds of 2.00+ from real bookmaker odds, or tells you nothing qualified |
| 📈 **Trends** | For every team playing on a chosen day (Prev / Next / Today / calendar, past or future): the most demanding Over / Under line its last 10 matches cleared, for 8 stats (shots, shots on target, corners, fouls, yellow cards, offsides, throw-ins, GK saves) in three views (team For, Against, Match total). Each card shows hit rate, average, median, a last-10 bar chart with the line drawn across, the model's probability for that line in the actual fixture, and the next match |
| 📊 **Shots** | Team shots per fixture: home team's average shots in its last 10 home matches, away team's in its last 10 away matches, their mean (Avg), fixtures listed highest first. Date picker (Prev / Next / Today) works for any past or future day |
| ⚽ **BTTS** | Both Teams To Score Yes/No, highest probability first, with bookmaker odds and value (EV) |
| 📋 **Results** | Finished matches (last 3 / 7 / 14 / 30 days): a scoreboard of hit rates by side, then one tidy row per match |
| 📆 **Date** | Pick any day (Prev / Next / Today): goals picks, BTTS and results for that day |
| 🔍 **Single Match** | Manual prediction lookup (Over/Under 2.5 and BTTS ranked by probability, with real odds / EV where the fixture is in today's picks) |

The **Date tab** reads rows already loaded on the page, so it makes no API calls. It can only show days the app
has loaded: about 2 days of upcoming fixtures, plus the results range chosen on the Results tab.

### Results tab

- **Scoreboard** at the top: one bar per side the model picked (Over 2.5, Under 2.5, BTTS Yes, BTTS No), showing
  *wins ÷ picks made*. A single blended hit rate hides which side is doing the work, so each side is separate.
  Bars turn green at 55% or more and red below 50% once a side has at least 10 picks; with fewer picks they stay
  neutral, and a side with no picks shows `—`. A footnote gives how often the most likely scoreline (and the top two) came in.
- **One row per match**, grouped by day (Nairobi time). Each day header shows that day's record, for example
  `Over/Under 7/10 · BTTS 6/10`. A row shows the league and kickoff, both teams with their goals, and a small ✓/✗ pill
  for each pick with the model's probability.
- **Tap a row** for the detail: final score and total goals, whether both teams scored, and the most likely
  scorelines with ✓/✗. The Date tab shows the same rows for the day you pick.

These rates grade the model's **most likely side**. Finished matches have no pre-kickoff odds on the free plan, so
they say nothing about profit.

## Acca builder

Choose **a day**, **one market to focus on** (Over/Under 2.5 or BTTS), a
**side** (best side per match, or one specific side: Over, Under, Yes or No) and **how many matches** (best of 2–3, exactly 2, or exactly 3), then tap **Build acca**. The
selections come from today's saved fixtures, so the tab makes **no API requests**.

**Step 1: which selections qualify.** A fixture's selection is a candidate only if all of these hold:

1. It has a **real bookmaker price** (model-made prices are never used)
2. Model probability is at least your **minimum probability** (`ACCA_MIN_LEG_PROB`, default 58%)
3. The real odds are at least your **minimum odds per leg** (`ACCA_MIN_LEG_ODDS`, default 1.15)
4. The match has not kicked off yet (its odds would be stale)

Each match contributes at most one candidate (its best side).

**Step 2: the acca rule.** The acca has **2 or 3 matches** (never fewer than 2 or more than 3), as you choose on the page,
and **combined odds of at least `ACCA_MIN_ODDS`** (default 2.00, editable on the page). Of all the combinations that
satisfy both, the one with the **highest combined probability** is returned. So if the most probable pair only pays,
say, 1.33, the builder **scales down** to the most probable combination that does reach 2.00 (swapping in
lower-probability, higher-paying matches, or using 3 matches when the choice allows), and a note says so. The search is exact,
not a shortcut: it is checked against brute force in the self-test (`python acca_builder.py`).

There is **no value / expected-value test**. The result shows each leg's odds and model probability, the combined
odds, **the model's chance that every leg wins**, the bookmaker-implied chance, and a stake calculator.

If fewer than 2 selections qualify, or no combination reaches the combined-odds target, **no acca is built**: the tab says
that no matches fulfilled the acca criteria and why (for example the best combined odds that were possible), with a
one-line count showing where fixtures dropped out (fixtures, with odds, met probability, met odds).

Acca prices multiply and so do the risks. The "all legs win" chance assumes the legs are independent, and the
model's probabilities were **not better than the bookmaker's own** in backtests, so a bigger combined price is not a
better bet. Of the older accumulator settings in `config.py`, the new builder reads only `ACCA_MIN_LEG_PROB`,
`ACCA_MIN_LEG_ODDS`, `ACCA_MIN_ODDS` and `ACCA_MAX_LEGS`; `ACCA_MAX_ODDS` (an upper cap), `ACCA_MIN_LEGS`, `ACCA_OVER_ONLY`,
`ACCA_POOL` and the `CACCA_*` settings belonged to a previous generator that is no longer part of the project, and are ignored.
The same logic is available as JSON at
`/api/acca?market=ou25&side=Over&legs=auto&min_prob=58&min_odds=1.15&min_combined=2` (`market` is `ou25` or `btts`; `legs` is `auto` for the best of 2 or 3, `2`, or `3`).

## Look and feel

The dashboard uses a navy / green / gold theme inspired by modern football-stats sites: deep navy background, slightly lighter navy cards, a green accent (`#10b880`) for
hits and active controls, gold for lines and key figures. Colours were sampled from a reference screenshot, not copied from any site's code. Fonts are bundled in
`static/fonts` (all SIL Open Font License, so they work offline and are free to redistribute): Plus Jakarta Sans for text, Barlow Condensed for headings,
JetBrains Mono for numbers. No logos or images from other sites are used. The previous red theme is kept as `static/theme.classic.css`: to go back, change
`theme.css?v=6` to `theme.classic.css` in the `<link>` near the top of `templates/index.html`.

## Trends tab

- **Data:** one `/events/{id}/stats/` request returns every stat, stored in `data/match_stats.csv` (and the shots also go to the Shots cache).
  **At launch** the automation first fetches the last 10 matches of every team in the upcoming fixtures (always, even beyond the daily budget, up to 800), so the
  Trends tab works for today's slate straight away. It then backfills the rest of the history, newest first, up to `STATS_DAILY_BUDGET` (default 1000) requests per UTC day
  in total, and stops at the first quota error. One request per match feeds three things at once: the stats store, the Shots cache and the real-xG cache
  (the xG upgrade and the shots step also store the stats they already download), so nothing is fetched twice. Loading all ~6,000 matches therefore takes a few days at
  the default; `python match_stats.py --fill 6000` does it in one resumable go if your quota allows (see API usage: the free cap is unknown and below 7,500). **Fetch missing stats** on the tab fills one day's gaps now. Until a team's last 10 matches are stored, it has no cards.
- **🔥 Top 10 bets:** the ten most confident trend bets for the chosen day, highest first. Confidence = (20 × model probability + hits) ÷ (20 + 10), i.e. the model counts about
  twice as much as the last-10 hit rate (your tests showed last-10 alone is noisy). Only lines the model rates 55-80% are considered (higher ones are priced around 1.20 or less,
  `TOP_MODEL_MIN` / `TOP_MODEL_MAX`), duplicates are merged and at most 2 bets per match are kept. **Fair odds** = 100 ÷ confidence: a bet is only worth it if a bookmaker pays more.
  There are no bookmaker odds for these markets, so this ranks likelihood, not value. For past days each pick shows WON / LOST.
- **Lines:** only lines between the 10th and 90th percentile of the stat are offered, so trivial lines never appear. "Realistic lines only" (on by default)
  also hides cards where the model gives the line under 35% or over 92%.
- **Model probability:** a ridge for/against model per stat (`STATS_HALF_LIFE`, default 180 days) gives the expected value for the fixture;
  a negative binomial, with its spread measured out of sample over the last 120 days, turns it into the probability of the line. It is not tested against
  bookmaker odds (no odds for these markets), so treat it as a sanity check on the streak, not as a betting edge.
- **Goals and BTTS:** Over / Under 2.5 goals and BTTS Yes / No trends are built from the scores already in `matches.csv`, so they need no stats requests
  and appear straight away. Their model probability is the app's own (goals model incl. the shots blend, and the BTTS model). BTTS bars show the scoreline.
- **Player markets** (e.g. a goalkeeper's saves) are not possible: the API has no player stats.

## Shots tab

- **Numbers:** for each fixture, the home team's average shots over its last 10 **home** matches and the away team's over its
  last 10 **away** matches (all competitions in your data), before kickoff. **Avg** = mean of the two; the list is sorted by it,
  highest first. Expand a row for the combined total, how many of the 10 matches had shots data, and (for past days) the actual shots.
- **Expected shots (xS) and Total:** the same for/against idea as xG, fitted on shots: the home team's shots-for strength against the
  away team's shots-conceded weakness (and the reverse), with recency weighting and opponent adjustment (`SHOTS_HALF_LIFE`, `SHOTS_RIDGE`).
  A second list, **Total expected shots**, sits right after the Average shots list and ranks fixtures by home xS + away xS.
  Expand a row to see the strength indices (×1.00 = league average). Run `python shots_tests.py` to check these settings on your data.
- **Date:** Prev / Next / Today or the calendar. Past days come from `matches.csv`; today and tomorrow from the saved picks; a later
  day needs the button "Look up fixtures" (about one request per league, cached for 6 hours).
- **Data:** shots are read from the same `/events/{id}/stats/` response as xG and stored in `data/shots_cache.csv`.
  The automation starts fetching as soon as `python app.py` is running: up to `SHOTS_DAILY_BUDGET` (default 1000) requests per UTC day in total,
  starting with the last-10 windows of today's teams, then every other finished match newest first, until the whole database is covered.
  The xG upgrade also saves shots from requests it already makes. **Fetch missing shots** on the tab is now just a manual top-up.
  `python shots_data.py --all` fetches everything in one go (resumable, stops at the first quota error; `--all 2000` caps it).
- **First run:** nothing to do, the automation fills the cache on its own.
  If it finds nothing, run `python shots_data.py --probe EVENT_ID` to see the API's field names and set `SHOTS_STAT_KEYS = [...]` in `config.py`.

**Shots are blended into the goals model (on by default, weight 0.2):** the expected goals behind Over/Under 2.5, BTTS and the history mix
shots-based expected goals (expected shots x league goals per shot) at weight 0.2 with the goals model. On 2,605 real matches weights of 0.2-0.3
improved goals log loss by 0.0017-0.0022 (best weight ~0.4, interval just touching 0), so the gain is small. Set `SHOTS_BLEND_WEIGHT = 0` in
`config.py` to switch it off. Fixtures where either team has no shots history use the plain goals model.
Real-data tests also showed a 180-day half-life suits shots better than 365 (now the default), and that recency weights inside the last-10 Avg do not help.

Optional settings: `SHOTS_AVG_DECAY = 1.0` (recency weight inside the last-10 Avg; 1.0 = plain average, try 0.9 once `shots_tests.py` supports it on your data), `SHOTS_DAILY_BUDGET = 1000`, `SHOTS_WINDOW = 10`, `SHOTS_STAT_KEYS = []`.

## Leagues covered

Premier League · La Liga · Serie A · Bundesliga · Ligue 1 · Liga Portugal · Eredivisie · Süper Lig ·
Brasileirão Série A · Champions League · Europa League · Conference League · Eliteserien · Swiss Super League ·
Championship · Serie B · MLS · 2. Bundesliga · **Liga MX**

**Liga MX uses two league IDs** because the Mexican season is split in two: **ID 19 = Apertura** (July–December) and
**ID 20 = Clausura** (January–May). They are the same competition with the same clubs and no overlapping fixtures,
so nothing is double counted. They are labelled `Liga MX (Apertura)` and `Liga MX (Clausura)` so the two dropdown
entries are distinguishable. Team ratings are shared across both.

Add more with `python add_leagues.py --add ID:Name[:Country]`, or **by name** (the script finds the IDs):

```bash
python add_leagues.py --preset a --dry-run   # check the IDs first (changes nothing)
python add_leagues.py --preset a             # Championship, Serie B, MLS, Liga MX, 2. Bundesliga
python add_leagues.py --preset b             # Belgian Pro League, Brasileirão Série B, Argentina LPF,
                                             # Saudi Pro League, Danish Superliga
python add_leagues.py --name "Serie B|Italy" # any single league
```

Run batch A and batch B on **different days**: each batch downloads about 2,500–3,000 matches and fetches xG for
each (one API request per match). That can use up the day's quota, and the download is resumable. After adding, restart the app.
Argentina has no xG on the API, so the model uses actual goals there.

## Requirements

- Linux with Python 3.10+ (developed on Termux, now run from Kali Linux)
- Python packages: `flask`, `requests`, `pandas`, `numpy`, `scipy`
- API key from [Bzzoiro Sports Data](https://sports.bzzoiro.com)

## Setup (Kali Linux)

### 1. Install dependencies

```bash
sudo apt update
sudo apt install -y python3 python3-pip git python3-flask python3-requests python3-pandas python3-numpy python3-scipy
```

Prefer a virtual environment? `python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`

### 2. Configure your API key

```bash
cp config.example.py config.py
nano config.py   # paste your Bzzoiro key into BZZOIRO_API_KEY
```

Never share `config.py` or paste it into chats or zips: it contains your key.

### 3. Run

```bash
python app.py
```

The first run fetches the match history in a few minutes. Real xG is **not** fetched all at once (that is one request per match and can use up a whole day's quota); goals stand in for xG and the daily automation upgrades matches to real xG gradually, newest first. Later runs start in
seconds. Open `http://127.0.0.1:5000`. The server listens on all interfaces (`0.0.0.0`), so other devices on your
network can reach it at your machine's IP; run it only on a network you trust.

## Daily workflow

```bash
cd ~/soccer-value-bets          # your project folder
python app.py
```

Open the page. Fixtures and picks are ready. Check **Results** later to see how they did.

## Automation (everything runs from `app.py`)

Start `python app.py` and leave it running. A background job wakes every 15 minutes and runs these steps **in order,
once per UTC day** (the API quota day), starting after the 03:00 Nairobi reset:

| Step | What it does | API requests |
|---|---|---|
| 1. Finished matches | Fetches matches finished since the newest one on disk, reloads the data and refits the model | ~20 league lists + 1 per page |
| 2. Fixtures, odds & picks | Builds today's fixtures with real odds and picks, and logs the odds snapshot | ~20 + 1 per fixture |
| 3. Real xG upgrade | Replaces goals-as-xG with real xG for up to `XG_DAILY_BUDGET` matches (default 400), **newest first** | up to the budget |
| 4. Results grading | Grades recent predictions so the Results tab opens instantly | none (local) |

Fixtures and odds always get the quota first; xG uses what is left, within its budget.

- **Failures retry:** a step that fails (network, API error) retries an hour later. If the API answers `429` (quota used up)
  every API step waits until 00:02 UTC and then continues. A `429` is never mistaken for "no xG exists".
- **xG backlog:** matches added by the daily update, and matches whose xG request failed during a big download, only have
  goals as xG. The upgrade clears them a few hundred a day. A match the API answers "no xG" for is remembered in
  `data/xg_missing.csv` and retried once a day, then given up after 3 empty answers once it is older than 7 days
  (some leagues, such as Argentina, publish no xG at all). Real xG still keeps the model's xG weighting meaningful;
  with goals standing in the model is mostly goal-based for those matches.
- **Where to look:** the **Automation** line at the bottom of the page (tap it to expand), `/health`, and
  `python app.py --status` show each step's state, the time it last ran, and the real-xG coverage. The state lives in
  `data/automation_status.json` and `data/daily_jobs.json`.
- **One fetch at a time:** the automation and a page view can never fetch the same fixtures twice.

Optional settings in `config.py`:

```python
XG_DAILY_BUDGET = 400                    # real-xG requests per day (0 = turn the upgrade off)
XG_ENRICH_AT_STARTUP = False             # True = fetch ALL xG on a first run (thousands of requests)
AUTOMATION_INTERVAL_SECONDS = 900        # how often the background job wakes
```

**Deliberately manual** (they change or measure the model, so you should look at the result first): `tune.py`,
`backtest.py`, `btts_backtest.py`, `add_leagues.py`, `repair_dates.py`, `odds_probe.py`
and `quota_check.py`. Intraday odds snapshots (`odds_logger.py --watch`) are also left off: each snapshot costs
about a hundred requests.

## API usage

The free Bzzoiro plan has a **daily request quota that resets at 00:00 UTC (03:00 Nairobi)**. The response headers
advertise 7,500 requests a day, but this setup has been blocked (`429 taster_exhausted`) after fewer than 2,600
requests by the header's own count, so **treat the real free cap as unknown and lower than 7,500**.
Check it any time with `python quota_check.py`.

| What | How often | Notes |
|---|---|---|
| Acca tab | **Never** | It only reads today's saved picks |
| Fixtures + odds + picks | **Once per UTC day** | Saved to `data/picks_cache.json`, reused on every page view and restart |
| Recent finished matches | **Once per UTC day** | Catches up from the newest match on disk (3–30 days). If it fails (quota or network) it is **not** marked done and retries hourly |
| Real xG upgrade | **Once per UTC day**, up to `XG_DAILY_BUDGET` | Newest matches first; stops at the first `429` |
| ↻ Refresh button | Only when you tap it | Forces a fresh fetch of fixtures and odds |

- The full 2-season download only runs when there is **no data on disk** (or `FORCE_REFRESH = True`).
  A partial download never overwrites your existing `matches.csv`.
- Which daily jobs already ran is stored in `data/daily_jobs.json`, and each step's result in `data/automation_status.json`.
- An empty or half-failed result is never saved, and the app retries at most every 30 minutes.
- Odds are a snapshot from the first load of the day. Tap **↻ Refresh** for fresh odds.
- **Page shows 0 matches?** A `429 taster_exhausted` reply means the daily limit is used up. Wait for 03:00 Nairobi
  time, then launch or refresh.
- **Measuring what something costs:** run `python quota_check.py --log` before and after an action; the drop in
  "remaining" is its cost. `--log` appends to `data/quota_log.csv`.

### Two scripts that can spend a lot of quota

- `python odds_probe.py --backfill` downloads historical odds one request per match. It now goes **newest first**,
  stops after **500** requests (`--limit N` changes it) and **stops after 5 consecutive 429s**.
- `python add_leagues.py` fetches xG one request per match (see above).

## Odds data for backtesting

- `data/odds_log.csv` is written **automatically** each time the app scans fixtures: genuine pre-kickoff Over/Under 2.5
  prices, one row per change, no extra requests. `python odds_analysis.py` tests the model against it.
- `data/odds_history.csv` (from `odds_probe.py`) holds prices for finished matches. These are not pre-kickoff prices,
  so use them for sanity checks, and trust `odds_log.csv` for honest value tests.

## BTTS (Both Teams To Score)

- **Model:** `P(BTTS Yes) = (1 − e^−λhome) × (1 − e^−λaway)`, using the same expected goals as Over/Under 2.5.
  BTTS odds (`btts_yes` / `btts_no`) arrive with the odds request already made for each fixture.
- **Backtest** (`python btts_backtest.py`, walk-forward on your data): well calibrated; BTTS Yes picks at 58%+ hit about 62%,
  65%+ about 72%. The edge over the plain base rate is small, so it is a lean, not a certainty.
- **Where it shows:** its own tab, All Matches, the Single Match card, the Date tab and the Results tab.

## Time zone

Everything on screen is shown in **EAT (Nairobi, UTC+3)**: kickoff times, the Today / Tomorrow buttons, day
grouping and the Results tab. Matches are grouped by Nairobi date, so a 21:30 UTC kickoff appears as 00:30 on the
next day. Stored data stays in UTC.

## CLI commands

| Command | Description |
|---|---|
| `python app.py` | Daily run (fetch if stale, serve UI) |
| `python app.py --no-refresh` | Skip the startup fetch, just serve |
| `python app.py --scan` | Print today's picks, exit |
| `python app.py --all-matches` | Print all upcoming matches, exit |
| `python app.py --status` | Show data freshness and file info |
| `python quota_check.py` | Show API quota status (`--log` appends to `data/quota_log.csv`) |
| `python acca_builder.py` | Self-test of the acca builder |
| `python btts_backtest.py` | Backtest BTTS, get calibration values |
| `python tune.py` | Grid-search the goals model's settings |
| `python backtest.py` | Confidence-band accuracy report |
| `python odds_analysis.py` | Backtest Over/Under 2.5 against your own logged pre-kickoff odds |
| `python odds_logger.py` | Log odds snapshots by hand (`--watch`, `--every N`) |
| `python odds_probe.py` | Sample odds coverage; `--backfill` / `--refresh` download odds (quota-guarded, `--limit N`) |
| `python add_leagues.py` | Add leagues to `config.py` and fetch them |
| `python repair_dates.py` | One-off repair for blanked dates in the match files |
| `python debug_picks.py` | Trace every upcoming fixture through the pick pipeline to see where one drops out |

## Project structure

```
soccer-value-bets/
├── app.py                  # Flask app, daily pipeline, once-a-day cache, routes
├── config.example.py       # Template config (safe to share)
├── config.py               # Your config with API key (never share)
├── fetch_data.py           # Bzzoiro fetch + xG enrichment (reports failed fetches so they retry)
├── model.py                # Goals model (ridge Poisson)
├── acca_builder.py         # Acca builder: most probable combination for Over/Under 2.5 or BTTS, real odds only
├── btts_model.py           # BTTS model
├── match_stats.py          # Full match-stats store + Trends engine (python match_stats.py --status | --fill N; python stats_probe.py --scan)
├── shots_data.py           # Shots cache, last-10 home/away averages, Shots tab data (python shots_data.py --probe ID)
├── btts_backtest.py        # BTTS backtest / calibration
├── scan.py                 # Upcoming fixtures + picks + odds
├── results_extra.py        # BTTS predictions and per-side hit rates for the Results tab
├── predictions_history.py  # Stored O/U 2.5 predictions for the Results tab
├── odds_logger.py          # Pre-kickoff Over/Under 2.5 odds snapshots (odds_log.csv)
├── odds_probe.py           # Historical odds download (quota-guarded)
├── odds_analysis.py        # O/U 2.5 backtest against logged odds
├── quota_check.py          # API quota status
├── tune.py / backtest.py   # Goals calibration and verification
├── add_leagues.py          # League manager
├── debug_picks.py / debug_api.py / oddtest.py / repair_dates.py   # diagnostics and one-off repairs
├── requirements.txt
├── static/theme.css
├── templates/index.html    # Dashboard (all tabs)
└── data/                   # Auto-created
    ├── matches.csv, matches_with_xg.csv, xg_cache.csv
    ├── picks_cache.json            # Today's fixtures + picks (once-a-day cache)
    ├── daily_jobs.json             # Which daily jobs already ran
    ├── automation_status.json      # Result, detail and next try of each automation step
    ├── xg_missing.csv              # Matches the API has no xG for (retried daily, then given up)
    ├── predictions_history.csv     # Cached point-in-time O/U 2.5 predictions
    ├── btts_pred_history.csv       # Cached point-in-time BTTS predictions for Results
    ├── odds_log.csv, odds_history.csv
    └── quota_log.csv               # Written by quota_check.py --log
```

## How the model works

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
5. Sum the scorelines to get the two markets: Over/Under 2.5 (total goals) and BTTS (both teams ≥ 1)
6. Rank the scorelines to show the 2 most likely ones next to each pick

**Value:** value means the model's probability times the bookmaker's odds is above 1 (EV = `model_prob × odds − 1`).
The dashboard shows EV next to the odds on the BTTS tab and the Single Match card. The Acca builder ignores value and works on
probability alone. The `MIN_EV` (3%), `MAX_EV` (15%) and `MIN_ODDS` (1.30) settings in `config.py` are the rule used by
`odds_analysis.py` and `debug_picks.py` when they test or trace Over/Under 2.5 value picks.

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| 0 matches / empty tabs | Quota used up (`429 taster_exhausted`). Run `python quota_check.py`, wait for the reset (03:00 Nairobi) |
| "Real xG on only N% of matches" | Normal after a big download: the daily upgrade adds `XG_DAILY_BUDGET` a day, newest first. Raise it in `config.py` if your quota allows |
| An automation step shows "waiting for the quota reset" | The API answered `429`. Nothing is wrong: it continues after 03:00 Nairobi time |
| Results tab slow the first time | It computes and caches point-in-time predictions once; later loads are instant |
| Blank dates / matches missing from Results | Run `python repair_dates.py` once |
| `ModuleNotFoundError` | Install the packages in [Setup](#1-install-dependencies) |

## Disclaimer

This is a personal research tool, not financial or betting advice. Sports betting carries risk of loss. Never bet
more than you can afford to lose. Historical backtest performance does not guarantee future results, and on the
data this was developed with, the model's probabilities were not better than the bookmaker's own. Bet responsibly (18+).

## License

MIT
