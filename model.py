# model.py
import os
import pandas as pd
from scipy.stats import poisson

from config import (
    MATCHES_XG_CSV, MATCHES_CSV,
    RECENT_WEIGHT, SEASON_WEIGHT, LAST_N_GAMES,
    LAMBDA_BOOST, OVER_THRESHOLD,
    HIGH_CONF_OVER_PROB, HIGH_CONF_UNDER_PROB,
)


def load_data():
    if os.path.exists(MATCHES_XG_CSV):
        df = pd.read_csv(MATCHES_XG_CSV)
        print(f"Loaded {len(df)} matches from {MATCHES_XG_CSV}")
    elif os.path.exists(MATCHES_CSV):
        df = pd.read_csv(MATCHES_CSV)
        if "home_xg" not in df.columns:
            df["home_xg"] = df["home_goals"]
        if "away_xg" not in df.columns:
            df["away_xg"] = df["away_goals"]
        print(f"Loaded {len(df)} matches from {MATCHES_CSV} (no xG)")
    else:
        raise FileNotFoundError("No data found. Run python app.py first.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    return df.sort_values("date").reset_index(drop=True)


def _weighted_mean(series, w_recent=RECENT_WEIGHT, n_recent=LAST_N_GAMES):
    if series is None or len(series) == 0:
        return None
    recent = series.tail(n_recent)
    all_mean = series.mean()
    recent_mean = recent.mean() if len(recent) else all_mean
    return w_recent * recent_mean + (1 - w_recent) * all_mean


def predict_lambdas(df, home_team_id, away_team_id, league_id=None):
    if league_id:
        scope = df[df["league_id"] == league_id]
    else:
        scope = df
    if len(scope) < 20:
        scope = df[df["league_id"] == league_id] if league_id else df

    lg_home_xg = scope["home_xg"].mean()
    lg_away_xg = scope["away_xg"].mean()
    if not lg_home_xg or lg_home_xg <= 0:
        lg_home_xg = 1.40
    if not lg_away_xg or lg_away_xg <= 0:
        lg_away_xg = 1.10

    home_home = scope[scope["home_team_id"] == home_team_id].sort_values("date")
    home_scored_home   = _weighted_mean(home_home["home_xg"])
    home_conceded_home = _weighted_mean(home_home["away_xg"])

    away_away = scope[scope["away_team_id"] == away_team_id].sort_values("date")
    away_scored_away   = _weighted_mean(away_away["away_xg"])
    away_conceded_away = _weighted_mean(away_away["home_xg"])

    if home_scored_home is None or home_scored_home <= 0:
        home_scored_home = lg_home_xg
    if home_conceded_home is None or home_conceded_home <= 0:
        home_conceded_home = lg_away_xg
    if away_scored_away is None or away_scored_away <= 0:
        away_scored_away = lg_away_xg
    if away_conceded_away is None or away_conceded_away <= 0:
        away_conceded_away = lg_home_xg

    home_attack  = home_scored_home   / lg_home_xg
    home_defence = home_conceded_home / lg_away_xg
    away_attack  = away_scored_away   / lg_away_xg
    away_defence = away_conceded_away / lg_home_xg

    lambda_home = home_attack * away_defence * lg_home_xg
    lambda_away = away_attack * home_defence * lg_away_xg

    return max(lambda_home, 0.20), max(lambda_away, 0.20)


def over_under_2_5(lambda_home, lambda_away, max_goals=10):
    prob_over = prob_under = 0.0
    score_matrix = {}

    for h in range(max_goals + 1):
        ph = poisson.pmf(h, lambda_home)
        for a in range(max_goals + 1):
            pa = poisson.pmf(a, lambda_away)
            p = ph * pa
            score_matrix[(h, a)] = p
            if h + a <= 2:
                prob_under += p
            else:
                prob_over += p

    total = prob_over + prob_under
    if total > 0:
        prob_over /= total
        prob_under /= total

    top_scores = sorted(score_matrix.items(), key=lambda x: x[1], reverse=True)[:6]
    return prob_over, prob_under, top_scores


def classify_confidence(prob_over):
    if prob_over >= HIGH_CONF_OVER_PROB:
        return "HIGH", "Over 2.5", "green"
    if prob_over <= HIGH_CONF_UNDER_PROB:
        return "HIGH", "Under 2.5", "green"
    if 0.60 <= prob_over < HIGH_CONF_OVER_PROB:
        return "MEDIUM", "Lean Over 2.5", "yellow"
    if HIGH_CONF_UNDER_PROB < prob_over <= 0.40:
        return "MEDIUM", "Lean Under 2.5", "yellow"
    return "LOW", "No confident pick", "gray"


def full_prediction(df, home_team_id, away_team_id, league_id=None):
    lh_raw, la_raw = predict_lambdas(df, home_team_id, away_team_id, league_id)

    lh = lh_raw * LAMBDA_BOOST
    la = la_raw * LAMBDA_BOOST

    prob_over, prob_under, top_scores = over_under_2_5(lh, la)
    tier, recommendation, color = classify_confidence(prob_over)

    return {
        "lambda_home":      round(lh, 2),
        "lambda_away":      round(la, 2),
        "prob_over_2_5":    round(prob_over * 100, 1),
        "prob_under_2_5":   round(prob_under * 100, 1),
        "recommendation":   recommendation,
        "confidence":       tier,
        "confidence_color": color,
        "top_scores": [
            {"score": f"{h}-{a}", "prob": round(p * 100, 1)}
            for (h, a), p in top_scores
        ],
    }
