"""Does geography predict how surprising a setlist is? City- and
region-level surprisal, empirical-Bayes shrunk so cities with only one or
two shows don't swing wildly on noise.

shrunk_mean = (n * raw_mean + k * global_mean) / (n + k)

k is in "equivalent shows" of prior strength -- a city with n << k stays
close to the global average; a city with n >> k is dominated by its own
history. Same k as predict.py's CITY_SHRINKAGE, for consistency.

Requires analysis/show_surprisal.csv (predict.py) and
analysis/city_coordinates.csv (geocode_cities.py).

Usage:
    /Users/bdoughty/opt/miniconda3/envs/tmg-scrape/bin/python geography.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis"
D = ROOT / "data"

SHRINKAGE_K = 8  # equivalent shows; matches predict.py's CITY_SHRINKAGE
MIN_SHOWS_TO_RANK = 3  # cities/regions below this are shrunk hard enough to be uninformative to list


def shrink(df, group_col, global_mean):
    g = df.groupby(group_col).agg(
        n_shows=("mean_surprisal_played_bits", "size"),
        raw_mean_bits=("mean_surprisal_played_bits", "mean"),
    )
    g["shrunk_mean_bits"] = (g.n_shows * g.raw_mean_bits + SHRINKAGE_K * global_mean) / (g.n_shows + SHRINKAGE_K)
    return g.reset_index()


def main():
    surprisal = pd.read_csv(OUT / "show_surprisal.csv", parse_dates=["date"]).dropna(
        subset=["mean_surprisal_played_bits"]
    )
    shows = pd.read_csv(D / "shows.csv")[["show_id", "city", "region"]]
    df = surprisal.merge(shows, on="show_id", how="left").dropna(subset=["city"])
    global_mean = df.mean_surprisal_played_bits.mean()
    print(f"global mean surprisal: {global_mean:.2f} bits/song, over {len(df)} shows with a known city")

    city = shrink(df, "city", global_mean).sort_values("shrunk_mean_bits", ascending=False)
    coords = pd.read_csv(OUT / "city_coordinates.csv")[["city", "region", "n_shows", "lat", "lon"]]
    # A city name can repeat across regions (Durham NC vs Durham NH) --
    # coords is already deduped on (city, region); collapse to one row per
    # city name (weighted by shows) for the surprisal join, which operates
    # on city name alone.
    coords_by_city = (
        coords.sort_values("n_shows", ascending=False).drop_duplicates("city")[["city", "lat", "lon"]]
    )
    city = city.merge(coords_by_city, on="city", how="left")
    city.to_csv(OUT / "city_surprisal.csv", index=False)

    region = shrink(df.dropna(subset=["region"]), "region", global_mean).sort_values(
        "shrunk_mean_bits", ascending=False
    )
    region.to_csv(OUT / "region_surprisal.csv", index=False)

    eligible = city[city.n_shows >= MIN_SHOWS_TO_RANK]
    print(f"\n=== Most surprising CITIES (shrunk, >={MIN_SHOWS_TO_RANK} shows, k={SHRINKAGE_K}) ===")
    print(eligible.head(12)[["city", "n_shows", "raw_mean_bits", "shrunk_mean_bits"]]
          .to_string(index=False, formatters={"raw_mean_bits": "{:.2f}".format, "shrunk_mean_bits": "{:.2f}".format}))
    print(f"\n=== Most predictable CITIES ===")
    print(eligible.tail(8)[["city", "n_shows", "raw_mean_bits", "shrunk_mean_bits"]]
          .to_string(index=False, formatters={"raw_mean_bits": "{:.2f}".format, "shrunk_mean_bits": "{:.2f}".format}))

    sf = city[city.city == "San Francisco"].iloc[0]
    rank = int((city.shrunk_mean_bits > sf.shrunk_mean_bits).sum()) + 1
    print(f"\n=== San Francisco specifically ===")
    print(f"{int(sf.n_shows)} shows, raw mean {sf.raw_mean_bits:.2f} bits, "
          f"shrunk mean {sf.shrunk_mean_bits:.2f} bits -- rank {rank} of {len(city)} cities "
          f"(global mean {global_mean:.2f})")

    eligible_r = region[region.n_shows >= MIN_SHOWS_TO_RANK]
    print(f"\n=== Most/least surprising REGIONS (shrunk, >={MIN_SHOWS_TO_RANK} shows) ===")
    print(pd.concat([eligible_r.head(6), eligible_r.tail(6)])
          [["region", "n_shows", "raw_mean_bits", "shrunk_mean_bits"]]
          .to_string(index=False, formatters={"raw_mean_bits": "{:.2f}".format, "shrunk_mean_bits": "{:.2f}".format}))

    print(f"\nWrote analysis/city_surprisal.csv, analysis/region_surprisal.csv")


if __name__ == "__main__":
    main()
