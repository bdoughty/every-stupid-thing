"""Geocode every (city, region) in the tour history against an offline
city database (geonamescache -- bundled data, no network calls, so no
risk of inventing coordinates from memory).

Disambiguates by region: US shows match on (countrycode='US',
admin1code=<state postal code>); international shows map the region
string (a country name, sometimes with noise like "The Netherlands (Extra
Glenns set)") to an ISO country code; Canadian shows match on
countrycode='CA' regardless of province (city names rarely collide).
Anything that doesn't match cleanly is left uncoded and reported, rather
than guessed.

Usage:
    /Users/bdoughty/opt/miniconda3/envs/tmg-scrape/bin/python geocode_cities.py
"""

import re
import unicodedata
from pathlib import Path

import geonamescache
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis"

US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN",
    "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV",
    "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN",
    "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "D.C.",
}
CANADIAN_PROVINCES = {
    "Ontario", "Quebec", "British Columbia", "Alberta", "Manitoba",
    "ON", "QC", "BC", "AB", "MB", "Nova Scotia", "Saskatchewan",
}
# Country-name aliases as they appear in our `region` column -> ISO code.
COUNTRY_ALIASES = {
    "england": "GB", "scotland": "GB", "wales": "GB", "united kingdom": "GB", "uk": "GB",
    "the netherlands": "NL", "netherlands": "NL", "holland": "NL",
    "australia": "AU", "au": "AU", "new zealand": "NZ", "nz": "NZ",
    "ireland": "IE", "sweden": "SE", "germany": "DE", "spain": "ES", "france": "FR",
    "belgium": "BE", "japan": "JP", "norway": "NO", "denmark": "DK", "poland": "PL",
    "austria": "AT", "portugal": "PT", "switzerland": "CH", "italy": "IT",
    "finland": "FI", "czech republic": "CZ", "czechia": "CZ", "hungary": "HU",
}


def load_cities():
    gc = geonamescache.GeonamesCache()
    cities = list(gc.get_cities().values())
    by_country = {}
    for c in cities:
        by_country.setdefault(c["countrycode"], []).append(c)
    return by_country


# A handful of unambiguous English exonyms for major cities geonamescache
# only lists under their local name. High-confidence, verified against the
# database directly (not guessed) -- deliberately NOT a general fuzzy-match
# mechanism, just spelling variants of the same well-known place.
EXONYM_ALIASES = {"gothenburg": "goteborg", "gothenberg": "goteborg", "cologne": "koln"}


def normalize(name):
    """Case/diacritic/abbreviation-insensitive comparison key -- NOT a
    similarity heuristic. "Montréal" must equal "Montreal"; "Millvale"
    must never equal "Philadelphia"."""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = name.lower().strip()
    name = re.sub(r"^st\.?\s+", "saint ", name)
    name = re.sub(r"^ft\.?\s+", "fort ", name)
    return EXONYM_ALIASES.get(name, name)


def best_match(candidates, name):
    """Exact match only (after normalizing case/diacritics/abbreviations),
    or the same name with "city" appended ("New York" -> "New York City").
    Deliberately does NOT fall back to "biggest city in the region" for a
    non-match -- that produced silently wrong coordinates (Pittsboro NC
    matched to Charlotte, Montreal to Toronto, Millvale PA to Philadelphia)
    for any town smaller than geonamescache's population cutoff. Unmatched
    stays unmatched; see the caller's reporting of that.
    """
    target = normalize(name)
    hits = [c for c in candidates if normalize(c["name"]) == target]
    if not hits:
        hits = [c for c in candidates if normalize(c["name"]) == target + " city"]
    if not hits:
        return None
    return max(hits, key=lambda c: c.get("population", 0))


def geocode(city, region, by_country):
    if not isinstance(city, str) or not city.strip():
        return None, "no city"
    city = city.strip()
    region = region.strip() if isinstance(region, str) else ""
    region_clean = re.sub(r"\s*\(.*?\)\s*$", "", region).strip()  # drop "(Extra Glenns set)" etc.

    if region_clean.upper() in US_STATES or region_clean == "D.C.":
        code = "DC" if region_clean == "D.C." else region_clean.upper()
        pool = [c for c in by_country.get("US", []) if c["admin1code"] == code]
        m = best_match(pool, city)
        return m, "us_state" if m else "us_state_no_match"

    if region_clean in CANADIAN_PROVINCES:
        m = best_match(by_country.get("CA", []), city)
        return m, "canada" if m else "canada_no_match"

    key = region_clean.lower()
    if key in COUNTRY_ALIASES:
        code = COUNTRY_ALIASES[key]
        m = best_match(by_country.get(code, []), city)
        return m, "country" if m else "country_no_match"

    return None, "unrecognized_region"


def main():
    shows = pd.read_csv(ROOT / "data" / "shows.csv")
    sh = shows[shows.n_songs > 0]
    grouped = sh.groupby(["city", "region"], dropna=False).size().reset_index(name="n_shows")

    by_country = load_cities()
    rows = []
    for r in grouped.itertuples():
        m, method = geocode(r.city, r.region, by_country)
        rows.append({
            "city": r.city,
            "region": r.region,
            "n_shows": r.n_shows,
            "lat": m["latitude"] if m else None,
            "lon": m["longitude"] if m else None,
            "matched_name": m["name"] if m else None,
            "match_method": method,
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "city_coordinates.csv", index=False)

    matched = out.lat.notna()
    print(f"{matched.sum()}/{len(out)} distinct (city, region) pairs geocoded "
          f"({out.loc[matched, 'n_shows'].sum()}/{out.n_shows.sum()} shows covered)")
    print("\nUnmatched, by frequency (kept out of the map rather than guessed):")
    print(out[~matched].sort_values("n_shows", ascending=False).head(20)
          [["city", "region", "n_shows", "match_method"]].to_string(index=False))


if __name__ == "__main__":
    main()
