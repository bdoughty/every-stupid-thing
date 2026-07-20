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

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis"

PRIOR_WEIGHT = 20  # pseudo-opportunities for shrinkage
MIN_OPPORTUNITIES = 20  # eligibility filter for deep-cut flagging
DEEP_CUT_QUANTILE = 0.25


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
        }
    ).reset_index()
    stats["opportunities"] = total - stats["first_t"] + 1

    global_rate = stats["n_plays"].sum() / stats["opportunities"].sum()
    alpha = global_rate * PRIOR_WEIGHT
    beta = (1 - global_rate) * PRIOR_WEIGHT
    stats["play_rate"] = stats["n_plays"] / stats["opportunities"]
    stats["play_rate_shrunk"] = (stats["n_plays"] + alpha) / (stats["opportunities"] + alpha + beta)

    eligible = stats["opportunities"] >= MIN_OPPORTUNITIES
    threshold = stats.loc[eligible, "play_rate_shrunk"].quantile(DEEP_CUT_QUANTILE)
    stats["deep_cut"] = eligible & (stats["play_rate_shrunk"] <= threshold)
    stats["surprisal"] = -np.log(stats["play_rate_shrunk"].clip(lower=1e-6))
    return stats.sort_values("n_plays", ascending=False), total


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

    print("\nWrote analysis/shows_per_year.csv, analysis/tours.csv, analysis/song_stats.csv")


if __name__ == "__main__":
    main()
