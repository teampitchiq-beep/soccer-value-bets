# results_extra.py
"""
Extra columns for the Results tab: what the corners model and the first-half goals model
WOULD have predicted before each finished match, next to what actually happened.

Point-in-time: each day's predictions use a model fitted only on matches before that day, so
nothing leaks. Predictions are cached (data/ht_pred_history.csv, data/corner_pred_history.csv) so
each match is computed once; new finished matches are added as they appear.
"""
import os

import numpy as np
import pandas as pd

import config

HT_PRED = "data/ht_pred_history.csv"
CO_PRED = "data/corner_pred_history.csv"
COLS = ["event_id", "ok", "pick", "prob", "mu_t", "top1_score", "top1_prob", "top2_score", "top2_prob"]


def _load(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.DataFrame(columns=COLS)


def _match_info(event_ids):
    m = pd.read_csv(config.MATCHES_XG_CSV,
                    usecols=["event_id", "date", "league_id", "home_team_id", "away_team_id"])
    m["date"] = pd.to_datetime(m["date"], errors="coerce", utc=True, format="ISO8601")
    return m[m["event_id"].isin(event_ids)].dropna(subset=["date"])


def _ensure(path, event_ids, train, fit_fn, predict_fn, min_matches):
    """Compute + cache predictions for event_ids that are not cached yet."""
    cache = _load(path)
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
        cache = pd.concat([cache, pd.DataFrame(new, columns=COLS)], ignore_index=True)
        os.makedirs("data", exist_ok=True)
        cache.to_csv(path, index=False)
    return cache


# ---------------------------------------------------------------- half-time
def _ht_predict(fit, hist, r):
    import ht_model as H
    mh, ma = H.expected_ht(fit, r.home_team_id, r.away_team_id, r.league_id)
    po = H.prob_over(mh + ma)
    over = po >= 0.5
    top = H.top_scores(mh, ma)
    return {"pick": f"{'Over' if over else 'Under'} {H.LINE:g} HT", "prob": round((po if over else 1 - po), 4),
            "mu_t": round(mh + ma, 2),
            "top1_score": top[0]["score"], "top1_prob": top[0]["prob"],
            "top2_score": top[1]["score"], "top2_prob": top[1]["prob"]}


def _ht(event_ids):
    import ht_model as H
    train = H.load_training()
    cache = _ensure(HT_PRED, event_ids, train, lambda hist, day: H.fit_ht(hist, now=day),
                    _ht_predict, H.MIN_MATCHES)
    actual = {}
    if os.path.exists(H.HT_CSV):
        a = pd.read_csv(H.HT_CSV).dropna(subset=["home_ht", "away_ht"])
        actual = {int(r.event_id): (int(r.home_ht), int(r.away_ht)) for r in a.itertuples()}
    return cache, actual, H.LINE


# ---------------------------------------------------------------- corners
def _co_predict(fit, hist, r):
    import corners_model as C
    mh, ma = C.expected_corners(fit, r.home_team_id, r.away_team_id, r.league_id)
    phi = float(C.DISPERSION) if C.DISPERSION else C.estimate_phi(fit, hist)
    po = C.prob_over(mh + ma, phi)
    over = po >= 0.5
    return {"pick": f"{'Over' if over else 'Under'} {C.LINE:g}", "prob": round((po if over else 1 - po), 4),
            "mu_t": round(mh + ma, 1)}


def _co(event_ids):
    import corners_model as C
    train = C.load_training()
    cache = _ensure(CO_PRED, event_ids, train, lambda hist, day: C.fit_corners(hist, now=day),
                    _co_predict, C.MIN_MATCHES)
    actual = {}
    if os.path.exists(C.CORNERS_CSV):
        a = pd.read_csv(C.CORNERS_CSV).dropna(subset=["home_corners", "away_corners"])
        actual = {int(r.event_id): (int(r.home_corners), int(r.away_corners)) for r in a.itertuples()}
    return cache, actual, C.LINE


# ---------------------------------------------------------------- public
def build_extras(event_ids):
    """{event_id: {"ht": {...} | None, "corners": {...} | None}} for the Results tab."""
    ids = [int(e) for e in event_ids]
    out = {e: {"ht": None, "corners": None} for e in ids}

    try:
        cache, actual, line = _ht(ids)
        pred = {int(r.event_id): r for r in cache[cache["ok"] == 1].itertuples()}
        for e in ids:
            p, a = pred.get(e), actual.get(e)
            if p is None and a is None:
                continue
            d = {"has_pred": p is not None, "has_actual": a is not None, "line": line}
            if a:
                d.update(ah=a[0], aa=a[1], total=a[0] + a[1], actual_over=(a[0] + a[1]) > line,
                         score=f"{a[0]}-{a[1]}")
            if p is not None:
                d.update(pick=p.pick, prob=round(p.prob * 100, 1), mu_t=p.mu_t,
                         top1_score=p.top1_score, top1_prob=p.top1_prob,
                         top2_score=p.top2_score, top2_prob=p.top2_prob)
                if a:
                    d["hit"] = p.pick.startswith("Over") == d["actual_over"]
                    d["top1_hit"] = p.top1_score == d["score"]
                    d["top2_hit"] = p.top2_score == d["score"]
            out[e]["ht"] = d
    except Exception as ex:
        print(f"Results (half-time) failed: {ex}")

    try:
        cache, actual, line = _co(ids)
        pred = {int(r.event_id): r for r in cache[cache["ok"] == 1].itertuples()}
        for e in ids:
            p, a = pred.get(e), actual.get(e)
            if p is None and a is None:
                continue
            d = {"has_pred": p is not None, "has_actual": a is not None, "line": line}
            if a:
                d.update(ah=a[0], aa=a[1], total=a[0] + a[1], actual_over=(a[0] + a[1]) > line)
            if p is not None:
                d.update(pick=p.pick, prob=round(p.prob * 100, 1), mu_t=p.mu_t)
                if a:
                    d["hit"] = p.pick.startswith("Over") == d["actual_over"]
            out[e]["corners"] = d
    except Exception as ex:
        print(f"Results (corners) failed: {ex}")
    return out


def summary(results):
    """Hit counts over the shown results: {'ht_ou': (hits, n), 'ht_top1': .., 'ht_top2': .., 'corners': ..}"""
    def tally(key, sub, flag):
        vals = [r[key][flag] for r in results if r.get(key) and flag in r[key]]
        return (sum(1 for v in vals if v), len(vals))
    s = {"ht_ou": tally("ht", "ht", "hit"), "ht_top1": tally("ht", "ht", "top1_hit"),
         "corners": tally("corners", "corners", "hit")}
    t1, n = s["ht_top1"]
    t2 = sum(1 for r in results if r.get("ht") and r["ht"].get("top2_hit"))
    s["ht_top2"] = (t1 + t2, n)           # top-1 and alt are always different scores
    return s
