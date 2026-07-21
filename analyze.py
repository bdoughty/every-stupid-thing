"""Starter analyses on the scraped Mountain Goats live-show data.

Reads data/shows.csv + data/performances.csv, writes analysis/*.csv and
prints a short report. Methodology for deep cuts follows
docs/deep_cut_notes.md: opportunity-adjusted play rates with empirical
Bayes shrinkage, where a song's "opportunities" are the shows *with
recorded setlists* since its first appearance.

Usage:
    /Users/bdoughty/opt/miniconda3/envs/tmg-scrape/bin/python analyze.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis"

PRIOR_WEIGHT = 20  # pseudo-opportunities for shrinkage
MIN_OPPORTUNITIES = 20  # eligibility filter for deep-cut flagging
DEEP_CUT_QUANTILE = 0.25

ENCORE_SHRINKAGE_K = 8  # pseudo-plays, empirical-Bayes toward the global encore rate
MIN_PLAYS_FOR_ENCORE_RANK = 10  # eligibility filter for encore-rate ranking


def load():
    shows = pd.read_csv(ROOT / "data" / "shows.csv", parse_dates=["date"])
    perfs = pd.read_csv(ROOT / "data" / "performances.csv", parse_dates=["date"])
    return shows, perfs


def song_stats(shows, perfs):
    """Per-song play counts, opportunities, and shrunk play rates."""
    # Only dated shows with at least one recorded song count as opportunities.
    setlist_shows = (
        shows[(shows.n_songs > 0) & shows.date.notna()]
        .sort_values("date")
        .reset_index(drop=True)
    )
    setlist_shows["t"] = np.arange(1, len(setlist_shows) + 1)
    total = len(setlist_shows)

    p = perfs.dropna(subset=["date"]).drop_duplicates(subset=["show_id", "song_key"])
    p = p.merge(setlist_shows[["show_id", "t"]], on="show_id", how="inner")

    g = p.groupby("song_key")
    stats = pd.DataFrame(
        {
            "song_title": g["song_canonical"].first(),
            "n_plays": g["show_id"].nunique(),
            "first_t": g["t"].min(),
            "first_played": g["date"].min(),
            "last_played": g["date"].max(),
            "album": g["album"].agg(lambda s: s.dropna().mode().iat[0] if s.dropna().size else None),
            "is_cover": g["is_cover"].any(),
        }
    ).reset_index()
    stats["opportunities"] = total - stats["first_t"] + 1

    global_rate = stats["n_plays"].sum() / stats["opportunities"].sum()
    alpha = global_rate * PRIOR_WEIGHT
    beta = (1 - global_rate) * PRIOR_WEIGHT
    stats["play_rate"] = stats["n_plays"] / stats["opportunities"]
    stats["play_rate_shrunk"] = (stats["n_plays"] + alpha) / (stats["opportunities"] + alpha + beta)

    # Covers are excluded from deep-cut eligibility: a Thin Lizzy cover
    # played once isn't a "passed-over" original the way a rarely-played
    # canonical song is — it was never really in the rotation to begin with.
    eligible = (stats["opportunities"] >= MIN_OPPORTUNITIES) & ~stats["is_cover"]
    threshold = stats.loc[eligible, "play_rate_shrunk"].quantile(DEEP_CUT_QUANTILE)
    stats["deep_cut"] = eligible & (stats["play_rate_shrunk"] <= threshold)
    stats["surprisal"] = -np.log(stats["play_rate_shrunk"].clip(lower=1e-6))
    return stats.sort_values("n_plays", ascending=False), total


def encore_stats(perfs):
    """Per-song main-set vs. encore play counts, restricted to shows where
    the encore breakout is actually known.

    `encore` defaults to 0 for every performance, including in the 888 of
    1317 setlisted shows whose Notes never called out an encore at all — so
    a song's raw main-set/encore split would just be diluted noise unless
    we first filter down to the shows that DO carry a real encore/main-set
    label (encore_map() found at least one "the encore was songs ..." note).
    """
    tracked_shows = perfs.groupby("show_id")["encore"].max()
    tracked_shows = tracked_shows[tracked_shows >= 1].index
    p = perfs[perfs.show_id.isin(tracked_shows)].copy()
    p["is_encore"] = p["encore"] >= 1

    g = p.groupby("song_key")
    stats = pd.DataFrame(
        {
            "song_title": g["song_canonical"].first(),
            "n_main": g["is_encore"].agg(lambda s: (~s).sum()),
            "n_encore": g["is_encore"].sum(),
            "is_cover": g["is_cover"].any(),
        }
    ).reset_index()
    stats["n_tracked_plays"] = stats["n_main"] + stats["n_encore"]

    global_rate = stats["n_encore"].sum() / stats["n_tracked_plays"].sum()
    alpha = global_rate * ENCORE_SHRINKAGE_K
    beta = (1 - global_rate) * ENCORE_SHRINKAGE_K
    stats["encore_rate"] = stats["n_encore"] / stats["n_tracked_plays"]
    stats["encore_rate_shrunk"] = (
        (stats["n_encore"] + alpha) / (stats["n_tracked_plays"] + alpha + beta)
    )
    stats["eligible"] = stats["n_tracked_plays"] >= MIN_PLAYS_FOR_ENCORE_RANK
    return stats.sort_values("n_tracked_plays", ascending=False), len(tracked_shows), global_rate


POSITION_SHRINKAGE_K = 8  # pseudo-plays, empirical-Bayes toward the global mean position
MIN_PLAYS_FOR_POSITION_RANK = 5
POSITION_DENSITY_MIN_PLAYS = 20  # need real sample size before a KDE means anything
POSITION_DENSITY_POINTS = 41  # grid resolution for the exported density curve


def _position_density(positions, grid):
    """Gaussian KDE of a song's position samples, boundary-corrected by
    reflecting the data across 0 and 1 before fitting (plain KDE badly
    underestimates density right at the edges -- exactly where openers and
    closers pile up) and renormalized so it integrates to 1 over [0, 1].
    """
    positions = np.asarray(positions, dtype=float)
    reflected = np.concatenate([positions, -positions, 2 - positions])
    density = gaussian_kde(reflected)(grid)
    return density / np.trapz(density, grid)


def set_position_stats(shows, perfs):
    """Per-song average position in the running order, 0 = opener, 1 =
    closer, shrunk toward the global mean. Position is (seq-1)/(n_songs-1)
    within that show's *full* running order (encores included -- seq counts
    straight through them, same as encore_stats() above relies on). Shows
    with only one song are excluded, since position is undefined there.

    Also attaches a KDE `density` (pipe-separated floats, POSITION_DENSITY_POINTS
    points evenly spaced over [0, 1]) for songs with enough samples -- a mean
    alone hides whether a song is consistently mid-set or bimodal (say, a
    frequent opener that's also sometimes the encore).
    """
    eligible_shows = shows.loc[shows.n_songs > 1, ["show_id", "n_songs"]]
    p = perfs.merge(eligible_shows, on="show_id", how="inner")
    p = p.assign(position=(p["seq"] - 1) / (p["n_songs"] - 1))

    g = p.groupby("song_key")
    stats = pd.DataFrame(
        {
            "song_title": g["song_canonical"].first(),
            "n_plays": g["position"].count(),
            "mean_position": g["position"].mean(),
        }
    ).reset_index()

    global_mean = p["position"].mean()
    k = POSITION_SHRINKAGE_K
    stats["shrunk_position"] = (
        stats["n_plays"] * stats["mean_position"] + k * global_mean
    ) / (stats["n_plays"] + k)
    stats["eligible"] = stats["n_plays"] >= MIN_PLAYS_FOR_POSITION_RANK

    grid = np.linspace(0, 1, POSITION_DENSITY_POINTS)
    density_by_key = {
        key: _position_density(sub.values, grid)
        for key, sub in p.groupby("song_key")["position"]
        if len(sub) >= POSITION_DENSITY_MIN_PLAYS
    }
    stats["density"] = stats["song_key"].map(
        lambda k: "|".join(f"{v:.3f}" for v in density_by_key[k]) if k in density_by_key else None
    )
    return stats.sort_values("n_plays", ascending=False), global_mean


def main():
    OUT.mkdir(exist_ok=True)
    shows, perfs = load()

    n_setlists = int((shows.n_songs > 0).sum())
    print(f"{len(shows)} shows, {n_setlists} with setlists, {len(perfs)} performances")
    print(f"dates {shows.date.min():%Y-%m-%d} .. {shows.date.max():%Y-%m-%d}\n")

    # Shows per year + coverage
    yearly = (
        shows.assign(has_setlist=shows.n_songs > 0)
        .groupby("year")
        .agg(shows=("show_id", "size"), with_setlist=("has_setlist", "sum"), songs=("n_songs", "sum"))
        .reset_index()
    )
    yearly.to_csv(OUT / "shows_per_year.csv", index=False)

    # Tour summary
    tours = (
        shows[shows.tour.notna()]
        .groupby("tour")
        .agg(shows=("show_id", "size"), first=("date", "min"), last=("date", "max"), avg_songs=("n_songs", "mean"))
        .sort_values("first")
        .round({"avg_songs": 1})
        .reset_index()
    )
    tours.to_csv(OUT / "tours.csv", index=False)

    stats, total_opportunities = song_stats(shows, perfs)
    stats.to_csv(OUT / "song_stats.csv", index=False)

    print("=== Top 15 most-played songs ===")
    top = stats.head(15)[["song_title", "n_plays", "opportunities", "play_rate"]]
    print(top.to_string(index=False, formatters={"play_rate": "{:.1%}".format}), "\n")

    deep = stats[stats.deep_cut].sort_values("play_rate_shrunk")
    print(f"=== Deep cuts: {len(deep)} songs (bottom {DEEP_CUT_QUANTILE:.0%} of shrunk play rate, "
          f">={MIN_OPPORTUNITIES} opportunities) — 15 deepest ===")
    print(
        deep.head(15)[["song_title", "n_plays", "opportunities", "play_rate_shrunk", "last_played"]]
        .to_string(index=False, formatters={"play_rate_shrunk": "{:.2%}".format})
    )

    print(f"\n=== Biggest tours ===")
    print(tours.sort_values("shows", ascending=False).head(10).to_string(index=False))

    encore, n_tracked, global_encore_rate = encore_stats(perfs)
    encore.to_csv(OUT / "encore_stats.csv", index=False)
    print(
        f"\n=== Main set vs. encore ({n_tracked} shows with a known encore breakout, "
        f"global encore rate {global_encore_rate:.1%}) ==="
    )
    elig = encore[encore.eligible]
    print("--- Most-played non-encore (main set) songs ---")
    print(
        elig.sort_values("n_main", ascending=False).head(10)[["song_title", "n_main", "n_encore"]]
        .to_string(index=False)
    )
    print("--- Most-played encore songs ---")
    print(
        elig.sort_values("n_encore", ascending=False).head(10)[["song_title", "n_main", "n_encore"]]
        .to_string(index=False)
    )
    print(f"--- Biggest encore-leaners (shrunk encore rate, >={MIN_PLAYS_FOR_ENCORE_RANK} tracked plays) ---")
    print(
        elig.sort_values("encore_rate_shrunk", ascending=False).head(10)
        [["song_title", "n_main", "n_encore", "encore_rate_shrunk"]]
        .to_string(index=False, formatters={"encore_rate_shrunk": "{:.1%}".format})
    )

    position, global_position = set_position_stats(shows, perfs)
    position.to_csv(OUT / "set_position.csv", index=False)
    pos_elig = position[position.eligible]
    print(f"\n=== Where in the set (0=opener, 1=closer; global mean {global_position:.2f}) ===")
    print("--- Earliest-leaning songs ---")
    print(
        pos_elig.sort_values("shrunk_position").head(10)[["song_title", "n_plays", "shrunk_position"]]
        .to_string(index=False, formatters={"shrunk_position": "{:.2f}".format})
    )
    print("--- Latest-leaning songs ---")
    print(
        pos_elig.sort_values("shrunk_position", ascending=False).head(10)
        [["song_title", "n_plays", "shrunk_position"]]
        .to_string(index=False, formatters={"shrunk_position": "{:.2f}".format})
    )

    print(
        "\nWrote analysis/shows_per_year.csv, analysis/tours.csv, analysis/song_stats.csv, "
        "analysis/encore_stats.csv, analysis/set_position.csv"
    )


if __name__ == "__main__":
    main()
