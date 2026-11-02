# results_extra.py
"""
Extra column for the Results tab: what the BTTS model WOULD have predicted before each finished
match, next to what actually happened. (Over/Under 2.5 is graded by predictions_history.py.)

Point-in-time: each day's predictions use a model fitted only on matches before that day, so
nothing leaks. Predictions are cached (data/btts_pred_history.csv) so each match is computed once;
new finished matches are added as they appear.
"""
import os

import numpy as np
import pandas as pd

import config

BT_PRED = "data/btts_pred_history.csv"
COLS = ["event_id", "ok", "pick", "prob", "mu_t", "top1_score", "top1_prob", "top2_score", "top2_prob"]


def _load(path, cols=None):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.DataFrame(columns=cols or COLS)


def _match_info(event_ids):
    m = pd.read_csv(config.MATCHES_XG_CSV,
                    usecols=["event_id", "date", "league_id", "home_team_id", "away_team_id"])
    m["date"] = pd.to_datetime(m["date"], errors="coerce", utc=True, format="ISO8601")
    return m[m["event_id"].isin(event_ids)].dropna(subset=["date"])


_FIT_MEMO = {}


def _fit_cached(hist, day):
    """One goals-model fit per day, used by the BTTS cache."""
    from model import fit_model
    key = (str(day), len(hist))
    if key not in _FIT_MEMO:
        if len(_FIT_MEMO) > 40:
            _FIT_MEMO.clear()
        _FIT_MEMO[key] = fit_model(hist, now=day)
    return _FIT_MEMO[key]


def _ensure(path, event_ids, train, fit_fn, predict_fn, min_matches, cols=None):
    """Compute + cache predictions for event_ids that are not cached yet."""
    cols = cols or COLS
    cache = _load(path, cols)
    have = set(cache["event_id"].astype(int).tolist()) if len(cache) else set()
    missing = [int(e) for e in event_ids if int(e) not in have]
    if not missing or train is None:
        return cache
    info = _match_info(missing)
    new = []
    for day, grp in info.groupby(info["date"].dt.normalize()):
        hist = train[train["date"] < day]
        fit = None
        if len(hist) >= 100:
            fit = fit_fn(hist, day)
            counts = pd.concat([hist["home_team_id"], hist["away_team_id"]]).value_counts()
        for r in grp.itertuples():
            row = {"event_id": int(r.event_id), "ok": 0}
            if fit is not None and counts.get(r.home_team_id, 0) >= min_matches \
                    and counts.get(r.away_team_id, 0) >= min_matches:
                try:
                    row.update(predict_fn(fit, hist, r))
                    row["ok"] = 1
                except Exception:
                    pass
            new.append(row)
    if new:
        cache = pd.concat([cache, pd.DataFrame(new, columns=cols)], ignore_index=True)
        os.makedirs("data", exist_ok=True)
        cache.to_csv(path, index=False)
    return cache


# ---------------------------------------------------------------- BTTS
def _bt_predict(fit, hist, r):
    from config import LAMBDA_BOOST
    from model import lambdas_from_fit
    import btts_model as B
    lh, la = lambdas_from_fit(fit, r.home_team_id, r.away_team_id, r.league_id)
    p = B.single(lh * LAMBDA_BOOST, la * LAMBDA_BOOST)
    return {"pick": p["pick"], "prob": p["prob"], "mu_t": round((lh + la) * LAMBDA_BOOST, 2),
            "top1_score": f"{p['p_yes']:.4f}"}          # top1_score slot stores P(BTTS Yes)


def _bt(event_ids):
    from model import load_data
    train = load_data()
    cache = _ensure(BT_PRED, event_ids, train, _fit_cached, _bt_predict, 10)
    g = pd.read_csv(config.MATCHES_XG_CSV, usecols=["event_id", "home_goals", "away_goals"])
    g = g[g["event_id"].isin(event_ids)].dropna()
    actual = {int(r.event_id): (int(r.home_goals), int(r.away_goals)) for r in g.itertuples()}
    return cache, actual


# ---------------------------------------------------------------- public
def build_extras(event_ids):
    """{event_id: {"btts": {...} | None}}"""
    ids = [int(e) for e in event_ids]
    out = {e: {"btts": None} for e in ids}
    try:
        cache, actual = _bt(ids)
        pred = {int(r.event_id): r for r in cache[cache["ok"] == 1].itertuples()}
        for e in ids:
            p, a = pred.get(e), actual.get(e)
            if p is None:
                continue
            d = {"has_pred": True, "has_actual": a is not None, "pick": p.pick,
                 "prob": round(p.prob * 100, 1)}
            if a is not None:
                d.update(ah=a[0], aa=a[1], actual_yes=a[0] > 0 and a[1] > 0)
                d["hit"] = (p.pick == "BTTS Yes") == d["actual_yes"]
            out[e]["btts"] = d
    except Exception as ex:
        print(f"Results (BTTS) failed: {ex}")
    return out


def summary(results):
    """Hit counts over the shown results, as (hits, picks) tuples, split by the side the model PICKED:
    by_side["ou25"]["Over"] = (Over 2.5 picks that won, Over 2.5 picks made). Same for BTTS Yes / No.
    top1 / top2 = how often the most likely / second most likely scoreline was the real one."""
    def by_side(pairs, sides):
        out = {side: [0, 0] for side in sides}
        for side, hit in pairs:
            if side in out:
                out[side][1] += 1
                out[side][0] += 1 if hit else 0
        return {side: (v[0], v[1]) for side, v in out.items()}

    ou25 = [(r["prediction"]["over_pick"].split()[0], r["prediction"]["over_hit"])
            for r in results if r.get("prediction")]
    btts = [(r["btts"]["pick"].replace("BTTS ", ""), r["btts"]["hit"])
            for r in results if r.get("btts") and "hit" in r["btts"]]
    pred = [r["prediction"] for r in results if r.get("prediction")]
    t1 = sum(1 for p in pred if p.get("top1_hit"))
    t2 = t1 + sum(1 for p in pred if p.get("top2_hit"))      # top-1 and alt are always different scores
    return {"top1": (t1, len(pred)), "top2": (t2, len(pred)),
            "by_side": {"ou25": by_side(ou25, ("Over", "Under")), "btts": by_side(btts, ("Yes", "No"))}}
