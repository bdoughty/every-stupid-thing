"""Predict which songs the Mountain Goats play at a given show.

Frames each show as ~800 binary outcomes (one per known song) and walks the
show history chronologically, so every feature uses only information
available before that night:

  - ewma_rate:   exponentially decayed play rate (half-life in shows, tuned
                 on a validation split) — this alone is the BASELINE model
  - recency:     shows since the song was last played, played-last-show
  - career_rate: plays per show since the song's live debut
  - song_age, new_material (debuted < 1.5 yrs ago — album-cycle proxy)
  - tour_rate / in_tour_pool: how often the song has appeared on the
                 current tour so far (captures per-tour rotations)
  - show_type:   radio session / festival / tv (much shorter sets)

Model: logistic regression on those features; compared against the EWMA
baseline on a strict temporal split (test = shows from TEST_START on).
Songs debuting on the night are unpredictable in principle and excluded
from candidate sets (reported separately).

Usage:
    /Users/bdoughty/opt/miniconda3/envs/tmg-scrape/bin/python predict.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis"

HALF_LIVES = [10, 25, 50, 100, 200]  # in shows; tuned on validation
TEST_START = "2023-01-01"
VAL_FRACTION = 0.2  # tail of the training era used to pick the half-life
NEW_MATERIAL_YEARS = 1.5
EPS = 1e-4


def load_shows_and_plays():
    shows = pd.read_csv(ROOT / "data" / "shows.csv", parse_dates=["date"])
    perfs = pd.read_csv(ROOT / "data" / "performances.csv", parse_dates=["date"])
    shows = (
        shows[(shows.n_songs > 0) & shows.date.notna()]
        .sort_values(["date", "show_id"])
        .reset_index(drop=True)
    )
    played = perfs.groupby("show_id")["song_key"].agg(set).to_dict()
    return shows, played


def build_rows(shows, played):
    """One row per (show, previously-debuted song), features as of that night."""
    songs = sorted({k for s in played.values() for k in s})
    idx = {k: i for i, k in enumerate(songs)}
    n = len(songs)

    alphas = {h: 0.5 ** (1.0 / h) for h in HALF_LIVES}
    ewma_count = {h: np.zeros(n) for h in HALF_LIVES}
    ewma_norm = {h: np.zeros(n) for h in HALF_LIVES}
    debuted = np.zeros(n, bool)
    first_t = np.zeros(n, int)
    first_date = np.full(n, np.datetime64("NaT"), "datetime64[ns]")
    last_t = np.zeros(n, int)
    n_plays = np.zeros(n, int)
    tour_plays = np.zeros(n, int)

    current_tour = object()
    tour_show_count = 0
    chunks = []

    for t, show in enumerate(shows.itertuples()):
        tonight = played.get(show.show_id, set())
        y_idx = np.array([idx[k] for k in tonight], int)

        # NaN tour = untoured one-off; treat as its own tour (reset rotation).
        tour = show.tour if isinstance(show.tour, str) else f"__oneoff_{t}"
        if tour != current_tour:
            current_tour = tour
            tour_show_count = 0
            tour_plays[:] = 0

        cand = np.flatnonzero(debuted)
        if cand.size:
            y = np.zeros(cand.size, bool)
            y[np.isin(cand, y_idx)] = True
            feats = {
                "show_id": show.show_id,
                "date": show.date,
                "t": t,
                "song_key": np.array(songs, object)[cand],
                "played": y,
                "shows_since_last": t - last_t[cand],
                "played_last_show": (last_t[cand] == t - 1).astype(float),
                "career_rate": n_plays[cand] / np.maximum(t - first_t[cand], 1),
                "song_age_years": ((np.datetime64(show.date) - first_date[cand]) / np.timedelta64(1, "D")) / 365.25,
                "new_material": None,  # filled below
                "tour_rate": tour_plays[cand] / max(tour_show_count, 1),
                "in_tour_pool": (tour_plays[cand] > 0).astype(float),
                "is_special_show": float(isinstance(show.show_type, str)),
            }
            for h in HALF_LIVES:
                with np.errstate(invalid="ignore"):
                    feats[f"ewma_{h}"] = np.where(
                        ewma_norm[h][cand] > 0, ewma_count[h][cand] / ewma_norm[h][cand], 0.0
                    )
            df = pd.DataFrame(feats)
            df["new_material"] = (df.song_age_years < NEW_MATERIAL_YEARS).astype(float)
            chunks.append(df)

        # --- update state with tonight's setlist (after features) ---
        for h in HALF_LIVES:
            a = alphas[h]
            ewma_count[h][debuted] *= a
            ewma_norm[h][debuted] = ewma_norm[h][debuted] * a + 1.0
        if y_idx.size:
            newly = y_idx[~debuted[y_idx]]
            old = y_idx[debuted[y_idx]]
            for h in HALF_LIVES:
                ewma_count[h][old] += 1.0
                # debut play seeds the decayed counters
                ewma_count[h][newly] = 1.0
                ewma_norm[h][newly] = 1.0
            debuted[newly] = True
            first_t[newly] = t
            first_date[newly] = np.datetime64(show.date)
            last_t[y_idx] = t
            n_plays[y_idx] += 1
            tour_plays[y_idx] += 1
        tour_show_count += 1

    return pd.concat(chunks, ignore_index=True)


def topn_overlap(df, prob_col):
    """Mean fraction of the actual setlist recovered by the top-n predictions."""
    def one(g):
        k = int(g.played.sum())
        if k == 0:
            return np.nan
        top = g.nlargest(k, prob_col)
        return top.played.sum() / k

    return df.groupby("show_id", sort=False).apply(one, include_groups=False).mean()


def evaluate(df, prob_col):
    p = df[prob_col].clip(EPS, 1 - EPS)
    return {
        "log_loss": log_loss(df.played, p),
        "brier": brier_score_loss(df.played, p),
        "top_n_overlap": topn_overlap(df, prob_col),
    }


def main():
    shows, played = load_shows_and_plays()
    print(f"{len(shows)} shows with dated setlists")
    rows = build_rows(shows, played)
    print(f"{len(rows):,} (show, song) rows, {rows.played.mean():.1%} positive")

    test_mask = rows.date >= TEST_START
    train = rows[~test_mask]
    cutoff_t = train.t.quantile(1 - VAL_FRACTION)
    fit, val = train[train.t < cutoff_t], train[train.t >= cutoff_t]
    test = rows[test_mask]
    print(f"fit {fit.show_id.nunique()} shows | val {val.show_id.nunique()} | test {test.show_id.nunique()}\n")

    # --- baseline: pick EWMA half-life on validation ---
    best_h = min(HALF_LIVES, key=lambda h: evaluate(val, f"ewma_{h}")["log_loss"])
    print(f"baseline: EWMA play rate, half-life {best_h} shows (tuned on validation)")

    # --- logistic model ---
    feat_cols = [
        f"ewma_{best_h}", "shows_since_last", "played_last_show", "career_rate",
        "song_age_years", "new_material", "tour_rate", "in_tour_pool", "is_special_show",
    ]
    Xf = rows[feat_cols].copy()
    Xf["shows_since_last"] = np.log1p(Xf["shows_since_last"])
    scaler = StandardScaler().fit(Xf[~test_mask])
    model = LogisticRegression(max_iter=2000)
    model.fit(scaler.transform(Xf[~test_mask]), rows.played[~test_mask])
    rows["p_model"] = model.predict_proba(scaler.transform(Xf))[:, 1]
    rows["p_baseline"] = rows[f"ewma_{best_h}"]

    print("\n=== Test-era performance (shows from %s) ===" % TEST_START)
    test = rows[test_mask]
    res = pd.DataFrame({
        "baseline (EWMA)": evaluate(test, "p_baseline"),
        "logistic model": evaluate(test, "p_model"),
    }).T
    print(res.round(4).to_string())

    print("\n=== Coefficients (standardized) ===")
    coefs = pd.Series(model.coef_[0], index=feat_cols).sort_values(key=abs, ascending=False)
    print(coefs.round(3).to_string())

    # --- example: most recent show ---
    last_id = test.sort_values("date").show_id.iloc[-1]
    g = test[test.show_id == last_id]
    k = int(g.played.sum())
    print(f"\n=== Example: {last_id} (setlist of {k}) ===")
    top = g.nlargest(k, "p_model")[["song_key", "p_model", "played"]]
    hit = top.played.sum()
    print(top.to_string(index=False, formatters={"p_model": "{:.2f}".format}))
    print(f"top-{k} recovered {hit}/{k} of the actual setlist")

    keep = ["show_id", "date", "song_key", "played", "p_baseline", "p_model"]
    rows.loc[test_mask, keep].to_csv(OUT / "model_test_predictions.csv", index=False)
    print(f"\nWrote analysis/model_test_predictions.csv "
          f"({int(test_mask.sum()):,} test rows)")


if __name__ == "__main__":
    main()
