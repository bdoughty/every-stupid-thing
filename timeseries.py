"""Rolling play-rate trajectories: p(song played) over calendar time.

For every song, computes a trailing-window play rate sampled on a regular
date grid: rate(d) = (shows containing the song in (d - WINDOW, d]) /
(shows with any setlist in that same window). This is the closest analogue
to a Google-Ngram-style frequency curve for setlists — it shows songs
appear at 0 before they're written, rise as they enter rotation, plateau,
decline, and sometimes resurge.

Two outputs:
    analysis/song_timeseries.csv        monthly grid, full precision, all songs
    analysis/song_timeseries_compact.csv quarterly grid, rounded, for the webapp

Usage:
    /Users/bdoughty/opt/miniconda3/envs/tmg-scrape/bin/python timeseries.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis"

WINDOW_DAYS = 730  # 2-year trailing window
MIN_OPPORTUNITIES = 5  # shows required in-window to report a rate (else NaN)
MIN_TOTAL_PLAYS = 3  # skip songs too rare to produce a meaningful curve


def load():
    shows = pd.read_csv(ROOT / "data" / "shows.csv", parse_dates=["date"])
    perfs = pd.read_csv(ROOT / "data" / "performances.csv", parse_dates=["date"])
    shows = shows[(shows.n_songs > 0) & shows.date.notna()].sort_values("date")
    perfs = perfs.dropna(subset=["date"])
    return shows, perfs


def rolling_rates(show_dates, song_dates_by_key, grid, window_days, min_opportunities):
    """Vectorized trailing-window rate per song over a shared date grid."""
    window = np.timedelta64(window_days, "D")
    lo_bound = grid - window
    denom = np.searchsorted(show_dates, grid, side="right") - np.searchsorted(
        show_dates, lo_bound, side="right"
    )
    valid = denom >= min_opportunities

    out = {}
    for key, dates in song_dates_by_key.items():
        numer = np.searchsorted(dates, grid, side="right") - np.searchsorted(dates, lo_bound, side="right")
        rate = np.full(grid.shape, np.nan)
        rate[valid] = numer[valid] / denom[valid]
        out[key] = rate
    return out, denom


def _with_asof_point(grid, asof):
    """Append the most recent show date to a quarter/month grid, if it's
    later than the grid's own last point -- otherwise a brand-new song's
    trailing rate is stuck at 0% until the *next* calendar quarter/month
    boundary rolls around, even though "as of the last show" it's already
    nonzero (e.g. a song that debuted two weeks ago, played several times
    since: genuinely 0% as of the last quarter-start, which predates it,
    but that's a stale snapshot, not today's actual rate).
    """
    if grid[-1] < asof:
        return np.append(grid, asof)
    return grid


def main():
    shows, perfs = load()
    show_dates = shows.date.values.astype("datetime64[D]")

    counts = perfs.groupby("song_key").size()
    keep_songs = counts[counts >= MIN_TOTAL_PLAYS].index
    song_dates_by_key = {
        k: np.sort(g.date.values.astype("datetime64[D]"))
        for k, g in perfs[perfs.song_key.isin(keep_songs)].groupby("song_key")
    }
    print(f"{len(song_dates_by_key)} songs with >={MIN_TOTAL_PLAYS} plays (of {perfs.song_key.nunique()} total)")

    asof = show_dates.max()
    monthly_grid = pd.date_range(shows.date.min(), shows.date.max(), freq="MS").values.astype("datetime64[D]")
    monthly_grid = _with_asof_point(monthly_grid, asof)
    rates, denom = rolling_rates(show_dates, song_dates_by_key, monthly_grid, WINDOW_DAYS, MIN_OPPORTUNITIES)

    long_rows = []
    for key, rate in rates.items():
        for d, r in zip(monthly_grid, rate):
            if not np.isnan(r):
                long_rows.append((key, d, r))
    ts = pd.DataFrame(long_rows, columns=["song_key", "date", "play_rate"])
    ts.to_csv(OUT / "song_timeseries.csv", index=False)
    print(f"Wrote analysis/song_timeseries.csv ({len(ts):,} rows, monthly, {WINDOW_DAYS}d trailing window)")

    # --- compact quarterly version for the webapp (smaller payload) ---
    quarterly_grid = pd.date_range(shows.date.min(), shows.date.max(), freq="QS").values.astype("datetime64[D]")
    quarterly_grid = _with_asof_point(quarterly_grid, asof)
    rates_q, _ = rolling_rates(show_dates, song_dates_by_key, quarterly_grid, WINDOW_DAYS, MIN_OPPORTUNITIES)
    compact_rows = []
    for key, rate in rates_q.items():
        for d, r in zip(quarterly_grid, rate):
            if not np.isnan(r):
                compact_rows.append((key, str(d), round(float(r), 3)))
    compact = pd.DataFrame(compact_rows, columns=["song_key", "date", "play_rate"])
    compact.to_csv(OUT / "song_timeseries_compact.csv", index=False)
    print(f"Wrote analysis/song_timeseries_compact.csv ({len(compact):,} rows, quarterly)")

    # --- auto-pick illustrative trajectories for the report plot ---
    span = ts.groupby("song_key").play_rate.agg(["max", "idxmax", "count"])
    total_plays = counts.reindex(span.index)

    classics = total_plays.nlargest(4).index.tolist()

    # "riser/plateau": debuted in roughly the back half of the catalog's
    # history and reached meaningful rotation since.
    first_seen = {k: d.min() for k, d in song_dates_by_key.items()}
    catalog_mid = shows.date.min() + (shows.date.max() - shows.date.min()) / 2
    late_debut = [k for k, d in first_seen.items() if pd.Timestamp(d) > catalog_mid]
    risers = (
        ts[ts.song_key.isin(late_debut)]
        .sort_values("date")
        .groupby("song_key")
        .play_rate.last()
        .nlargest(4)
        .index.tolist()
    )

    # "decline": high early-career rate, much lower in the most recent window.
    early = ts[ts.date < ts.date.min() + pd.Timedelta(days=365 * 8)]
    late = ts[ts.date > ts.date.max() - pd.Timedelta(days=365 * 3)]
    early_peak = early.groupby("song_key").play_rate.max()
    late_avg = late.groupby("song_key").play_rate.mean()
    decline = (early_peak - late_avg.reindex(early_peak.index).fillna(0)).nlargest(4).index.tolist()

    # "revival": a sustained near-zero trough between two active periods.
    def has_revival(g):
        g = g.sort_values("date")
        if len(g) < 12 or g.play_rate.max() < 0.1:
            return -1
        mid = g.play_rate.iloc[len(g) // 4 : 3 * len(g) // 4]
        return (g.play_rate.iloc[: len(g) // 4].max() > 0.08) and (mid.min() < 0.02) and (
            g.play_rate.iloc[3 * len(g) // 4 :].max() > 0.08
        )

    revival_flags = ts.groupby("song_key").apply(has_revival, include_groups=False)
    revivals = total_plays.reindex(revival_flags[revival_flags == True].index).nlargest(4).index.tolist()  # noqa: E712

    picks = {
        "classics": classics,
        "risers": risers,
        "decline": decline,
        "revivals": revivals,
    }
    print("\nAuto-picked example trajectories:")
    for cat, keys in picks.items():
        print(f"  {cat}: {keys}")

    import json

    (OUT / "timeseries_examples.json").write_text(json.dumps(picks, indent=1))


if __name__ == "__main__":
    main()
