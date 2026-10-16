# ⚽ Soccer Value Bets — Over/Under 2.5 Predictor

Dixon-Coles Poisson model with xG-weighted attack/defence, live odds comparison,
and automatic value-bet detection. Built entirely in Termux on a phone.

## What it does

- Fetches 2 seasons of match data across 10 top leagues from the **Bzzoiro Sports API**
- Enriches every match with **expected goals (xG)**
- Runs a **Dixon-Coles Poisson model** to predict Over/Under 2.5 probabilities
- Outputs the **top 2 most likely correct scores** for every fixture
- Compares model probabilities to **live bookmaker odds**
- Flags **value bets** where expected value is +3% to +15%
- Tracks bets, auto-settles results, and reports live ROI

## Features

- 📅 **All Matches** tab — every upcoming fixture with O/U 2.5 %, top 2 scores, and confidence tier
- 💎 **Value Bets** tab — only fixtures with positive edge vs the market
- 🔍 **Single Match** tab — manual prediction lookup
- 📊 **Bet Tracker** — one-tap logging, auto-settle from API, ROI dashboard
- 🔄 **Auto-refresh** — data refreshes if older than 24 hours
- ⚡ **Parallel xG enrichment** — 8 concurrent workers, no rate limits

## Leagues Covered

Premier League · La Liga · Serie A · Bundesliga · Ligue 1 · Liga Portugal ·
Eredivisie · Süper Lig · Brasileirão Série A · Champions League

## Requirements

- Termux (or any Linux/macOS environment with Python 3.10+)
- Python packages: `flask`, `requests`, `pandas`, `numpy`, `scipy`
- Free API key from [Bzzoiro Sports Data](https://sports.bzzoiro.com)

## Setup

### 1. Install dependencies

```bash
pkg update && pkg upgrade -y
pkg install python git nano -y
pip install flask requests pandas numpy scipy
```

### 2. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/soccer-value-bets.git
cd soccer-value-bets
```

### 3. Configure your API key

```bash
cp config.example.py config.py
nano config.py   # paste your Bzzoiro key
```

### 4. Run

```bash
python app.py
```

First run fetches 2 seasons of matches and enriches them with xG
(~10–20 minutes). Subsequent runs skip straight to serving the UI in ~5 seconds.

Open `http://127.0.0.1:5000` in your browser.

## Daily Workflow

```bash
cd ~/soccer-value-bets
python app.py
```

Then:

- Browse **All Matches** (sorted by Over 2.5 %)
- Check **Value Bets** for picks with positive edge
- Tap **✓ Log This Bet** when you place a wager
- View **Bet Tracker** for live ROI

## CLI Commands

| Command | Description |
|---|---|
| `python app.py` | Daily run (fetch if stale, serve UI) |
| `python app.py --no-refresh` | Skip fetch, just serve |
| `python app.py --force-refresh` | Wipe cache and refetch everything |
| `python app.py --scan` | Print value bets, exit |
| `python app.py --all-matches` | Print all upcoming matches, exit |
| `python app.py --status` | Show data freshness and file info |
| `python tune.py` | Grid-search LAMBDA_BOOST and threshold |
| `python backtest.py` | Confidence-band accuracy report |
| `python tracker.py` | Full ROI report |
| `python tracker.py --add` | Interactive bet logger |
| `python tracker.py --auto-settle` | Settle pending bets from results |

## Project Structure

```
soccer-value-bets/
├── app.py                  # Flask app + daily pipeline runner
├── config.example.py       # Template config (safe to commit)
├── config.py               # Your config with API key (gitignored)
├── fetch_data.py           # Bzzoiro fetch + xG enrichment
├── model.py                # Dixon-Coles Poisson engine
├── scan.py                 # Value bet scanner + all-fixtures scanner
├── tune.py                 # Grid-search calibrator
├── backtest.py             # Confidence-band verifier
├── tracker.py              # Bet tracker with auto-settle
├── requirements.txt
├── .gitignore
├── templates/
│   ├── index.html          # Main UI
│   └── tracker.html        # ROI dashboard
└── data/                   # Auto-created (gitignored)
    ├── matches.csv
    ├── matches_with_xg.csv
    └── xg_cache.csv
```

## How the Model Works

For each match:

1. Compute league averages of home/away xG
2. Compute each team's attack & defence strength relative to league average
   (weighted 60% recent form / 40% full season)
3. Apply multiplicative Poisson:
   ```
   λ_home = home_attack × away_defence × league_avg_home_xg × LAMBDA_BOOST
   λ_away = away_attack × home_defence × league_avg_away_xg × LAMBDA_BOOST
   ```
4. Compute the probability of every scoreline from 0-0 to 10-10
5. Sum to get Over/Under 2.5 probabilities
6. Rank scorelines to find the top 2 most likely correct scores

**Value detection:** compare model probability vs bookmaker implied
probability (1 / decimal odds). If `model_prob × odds − 1 > 3%` and the
pick is high-confidence, flag as a value bet.

## Disclaimer

This is a personal research tool, not financial advice. Sports betting
carries risk of loss. Never bet more than you can afford to lose. Historical
backtest performance does not guarantee future results. Bet responsibly.

## License

MIT
