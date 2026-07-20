"""Does geography predict how surprising a setlist is? City-level surprisal,
empirical-Bayes shrunk so cities with only one or two shows don't swing
wildly on noise.

Two-level cascade, pooling by actual METRO PROXIMITY, not administrative
state/country boundaries: cities within METRO_RADIUS_KM of each other (via
real lat/lon, DBSCAN + haversine distance) are treated as one scene and
pooled together (Boston/Cambridge/Somerville; Durham/Chapel Hill/Carrboro).
A state is not a scene -- pooling Pittsboro toward North Carolina's average
just because they share a state line pulls in unrelated small towns 150mi
away that have nothing to do with Durham's local character. Cities with NO
real geographic neighbor in the tour history (no other place within the
radius) are correctly left ungrouped, singleton "clusters of one" --
they fall back to a single shrink straight to the global mean, weighted
only by their OWN n_shows, so an isolated place with a real, distinct
signal (Watkins Glen's Farm Sanctuary benefit shows, genuinely unlike nearby
NY touring) never gets diluted by a fake regional average built from
nothing but itself.

shrunk = (n * raw_mean + k * prior) / (n + k)

k is in "equivalent shows" of prior strength -- a place with n << k stays
close to its prior; n >> k is dominated by its own history. Same k as
predict.py's CITY_SHRINKAGE, for consistency, applied at both levels.

Also computes a separate region_surprisal.csv (US state / country) purely
as its own descriptive table -- no longer used as anyone's shrinkage prior.

Requires analysis/show_surprisal.csv (predict.py) and
analysis/city_coordinates.csv (geocode_cities.py).

Usage:
    /Users/bdoughty/opt/miniconda3/envs/tmg-scrape/bin/python geography.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis"
D = ROOT / "data"

SHRINKAGE_K = 8  # equivalent shows; matches predict.py's CITY_SHRINKAGE
MIN_SHOWS_TO_RANK = 3  # cities/regions below this are shrunk hard enough to be uninformative to list
METRO_RADIUS_KM = 25  # ~15.5 miles -- deliberately conservative ("undergroup,
# not overgroup": Boston/Cambridge/Somerville and Durham/Chapel Hill/
# Carrboro are a few miles apart and cluster easily; a satellite town
# 25-40 miles out (Pittsboro, Watkins Glen) stays its own thing rather
# than getting smeared into a big regional blob)
EARTH_RADIUS_KM = 6371.0088


def shrink(df, group_cols, prior, k=SHRINKAGE_K):
    """prior: a scalar (shrink every group toward one value) or a Series
    sharing the post-groupby index (shrink each group toward its own
    prior -- pandas aligns the assignment below by index automatically)."""
    g = df.groupby(group_cols).agg(
        n_shows=("mean_surprisal_played_bits", "size"),
        raw_mean_bits=("mean_surprisal_played_bits", "mean"),
    )
    g["prior_bits"] = prior
    g["shrunk_mean_bits"] = (g.n_shows * g.raw_mean_bits + k * g.prior_bits) / (g.n_shows + k)
    return g.reset_index()


def cluster_metros(places):
    """places: DataFrame of distinct (city, region, lat, lon). Returns it
    with a `cluster` column -- shared for genuine geographic neighbors,
    a unique singleton id for anywhere with none."""
    coords = np.radians(places[["lat", "lon"]].to_numpy())
    eps = METRO_RADIUS_KM / EARTH_RADIUS_KM
    labels = DBSCAN(eps=eps, min_samples=1, metric="haversine").fit_predict(coords)
    places = places.copy()
    places["cluster"] = labels
    return places


def main():
    surprisal = pd.read_csv(OUT / "show_surprisal.csv", parse_dates=["date"]).dropna(
        subset=["mean_surprisal_played_bits"]
    )
    shows = pd.read_csv(D / "shows.csv")[["show_id", "city", "region"]]
    df = surprisal.merge(shows, on="show_id", how="left").dropna(subset=["city"])
    global_mean = df.mean_surprisal_played_bits.mean()
    print(f"global mean surprisal: {global_mean:.2f} bits/song, over {len(df)} shows with a known city")

    # --- region (state/country): its own descriptive table, not a prior ---
    region = shrink(df.dropna(subset=["region"]), "region", global_mean).sort_values(
        "shrunk_mean_bits", ascending=False
    )
    region.to_csv(OUT / "region_surprisal.csv", index=False)

    # --- metro clusters, from real distance, not administrative borders ---
    coords = pd.read_csv(OUT / "city_coordinates.csv").dropna(subset=["lat", "lon"])
    places = cluster_metros(coords[["city", "region", "lat", "lon"]].drop_duplicates(["city", "region"]))
    cluster_size = places.groupby("cluster")["city"].transform("size")
    places["is_singleton"] = cluster_size == 1

    df_place = df.merge(places[["city", "region", "cluster", "is_singleton"]], on=["city", "region"], how="left")
    # No coordinates at all -> treated exactly like a singleton (no known
    # neighbor to pool with): prior is the global mean, one shrink step.
    df_place["is_singleton"] = df_place["is_singleton"].fillna(True).astype(bool)

    multi = df_place[df_place.is_singleton == False]  # noqa: E712
    metro = shrink(multi, "cluster", global_mean).sort_values("shrunk_mean_bits", ascending=False)
    metro_prior = metro.set_index("cluster").shrunk_mean_bits
    df_place["_prior"] = np.where(
        df_place.is_singleton, global_mean, df_place["cluster"].map(metro_prior)
    )

    # Human-readable metro labels: the member city with the most shows.
    metro_label = (
        df_place[df_place.is_singleton == False]  # noqa: E712
        .groupby(["cluster", "city"]).size().reset_index(name="n")
        .sort_values("n", ascending=False).drop_duplicates("cluster")
        .set_index("cluster")["city"]
    )
    metro["label"] = metro.cluster.map(metro_label)
    members = (
        places[places.cluster.isin(metro.cluster)]
        .groupby("cluster")["city"].apply(lambda s: ", ".join(sorted(s)))
    )
    metro["members"] = metro.cluster.map(members)
    metro.to_csv(OUT / "metro_surprisal.csv", index=False)

    # --- city-level, grouped by (city, region) -- see module docstring for
    # why not city name alone (Portland OR vs ME, Durham NC vs NH, ...) ---
    city_prior = df_place.groupby(["city", "region"])["_prior"].first()
    city = shrink(df_place, ["city", "region"], city_prior).sort_values("shrunk_mean_bits", ascending=False)
    city["shrunk_flat_bits"] = (city.n_shows * city.raw_mean_bits + SHRINKAGE_K * global_mean) / (
        city.n_shows + SHRINKAGE_K
    )
    city = city.merge(coords[["city", "region", "lat", "lon"]], on=["city", "region"], how="left")
    city.to_csv(OUT / "city_surprisal.csv", index=False)

    print(f"\n=== Metro clusters found (>=2 places within {METRO_RADIUS_KM}km) ===")
    for r in metro.sort_values("n_shows", ascending=False).itertuples():
        print(f"  {r.label + ' area':22s} {r.members}  ({int(r.n_shows)} shows, "
              f"raw {r.raw_mean_bits:.2f}, shrunk {r.shrunk_mean_bits:.2f})")
    n_singleton = int(places.is_singleton.sum())
    print(f"{n_singleton} of {len(places)} geocoded places have no real neighbor within "
          f"{METRO_RADIUS_KM}km -- shrink straight to global, weighted only by their own n_shows.")

    eligible = city[city.n_shows >= MIN_SHOWS_TO_RANK]
    cols = ["city", "region", "n_shows", "raw_mean_bits", "prior_bits", "shrunk_mean_bits"]
    fmt = {c: "{:.2f}".format for c in ("raw_mean_bits", "prior_bits", "shrunk_mean_bits")}
    print(f"\n=== Most surprising CITIES (metro-cascaded, >={MIN_SHOWS_TO_RANK} shows, k={SHRINKAGE_K}) ===")
    print(eligible.head(12)[cols].to_string(index=False, formatters=fmt))
    print(f"\n=== Most predictable CITIES ===")
    print(eligible.tail(8)[cols].to_string(index=False, formatters=fmt))

    for city_name, region_code in [("San Francisco", "CA"), ("Watkins Glen", "NY")]:
        row = city[(city.city == city_name) & (city.region == region_code)]
        if row.empty:
            continue
        r = row.iloc[0]
        elig_sorted = eligible.sort_values("shrunk_mean_bits", ascending=False).reset_index(drop=True)
        match = elig_sorted[(elig_sorted.city == city_name) & (elig_sorted.region == region_code)].index
        rank = int(match[0]) + 1 if len(match) else None
        place_row = places[(places.city == city_name) & (places.region == region_code)]
        singleton = place_row.is_singleton.iloc[0] if len(place_row) else True  # ungeocoded == no known neighbor
        print(f"\n=== {city_name}, {region_code} specifically ===")
        print(f"{int(r.n_shows)} shows, raw {r.raw_mean_bits:.2f} bits, "
              f"{'no metro neighbor -- ' if singleton else ''}prior {r.prior_bits:.2f} bits "
              f"(global {global_mean:.2f}) -- shrunk {r.shrunk_mean_bits:.2f}"
              + (f", rank {rank} of {len(eligible)} places with >={MIN_SHOWS_TO_RANK} shows" if rank else ""))

    print(f"\nWrote analysis/city_surprisal.csv, analysis/region_surprisal.csv, analysis/metro_surprisal.csv")


if __name__ == "__main__":
    main()
