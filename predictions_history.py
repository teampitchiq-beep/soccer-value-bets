# predictions_history.py
"""
Retro-compute and cache the model's predictions for finished matches.

For each match, uses only training data with an earlier date.
Results are cached in data/predictions_history.csv so we don't
recompute on every page load.
"""
import os
import pandas as pd

CACHE_FILE = "data/predictions_history.csv"

COLUMNS = [
    "event_id", "po", "pu", "pred_over_pick",
    "top1_score", "top1_prob", "top2_score", "top2_prob",
]


def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            return pd.read_csv(CACHE_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=COLUMNS)


def save_cache(df):
    os.makedirs("data", exist_ok=True)
    df.to_csv(CACHE_FILE, index=False)


def compute_prediction(df, event_id):
    """
    Given the full dataset and an event_id, return the prediction the
    model WOULD have made before that match, using only earlier data.
    """
    from model import predict_lambdas, over_under_2_5
    from config import LAMBDA_BOOST

    rows = df[df["event_id"] == event_id]
    if rows.empty:
        return None
    match = rows.iloc[0]
    match_date = match["date"]
    league_id = int(match["league_id"])

    history = df[df["date"] < match_date]
    if len(history) < 50:
        return None

    try:
        lh, la = predict_lambdas(
            history,
            match["home_team_id"],
            match["away_team_id"],
            league_id,
        )
    except Exception:
        return None

    lh *= LAMBDA_BOOST
    la *= LAMBDA_BOOST
    po, pu, top = over_under_2_5(lh, la)

    top1 = top[0] if len(top) > 0 else ((0, 0), 0.0)
    top2 = top[1] if len(top) > 1 else ((0, 0), 0.0)

    return {
        "event_id":       int(event_id),
        "po":             round(po * 100, 1),
        "pu":             round(pu * 100, 1),
        "pred_over_pick": "Over 2.5" if po >= pu else "Under 2.5",
        "top1_score":     f"{top1[0][0]}-{top1[0][1]}",
        "top1_prob":      round(top1[1] * 100, 1),
        "top2_score":     f"{top2[0][0]}-{top2[0][1]}",
        "top2_prob":      round(top2[1] * 100, 1),
    }


def ensure_predictions(df, event_ids, progress=True):
    """Compute any missing predictions, cache them, return full cache."""
    cache = load_cache()
    existing = set(cache["event_id"].astype(int).tolist()) if len(cache) else set()
    missing = [int(e) for e in event_ids if int(e) not in existing]

    if not missing:
        return cache

    if progress:
        print(f"  Computing predictions for {len(missing)} finished matches...")

    new_rows = []
    for i, eid in enumerate(missing, 1):
        p = compute_prediction(df, eid)
        if p:
            new_rows.append(p)
        if progress and i % 25 == 0:
            print(f"    {i}/{len(missing)}")

    if new_rows:
        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        save_cache(cache)
        if progress:
            print(f"  Cached {len(new_rows)} predictions.")

    return cache


def get_predictions_for(event_ids):
    """Fast lookup from cache for a list of event_ids."""
    cache = load_cache()
    if len(cache) == 0:
        return {}
    filtered = cache[cache["event_id"].isin(event_ids)]
    return {int(r["event_id"]): r.to_dict() for _, r in filtered.iterrows()}
