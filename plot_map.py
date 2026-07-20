"""Map of shrunk per-city setlist surprise. Uses plotly's built-in
scattergeo basemap (no tile server, no API key, no internet at render
time beyond the one-time kaleido/plotly install) so this is fully
reproducible offline once dependencies are installed.

Requires analysis/city_surprisal.csv (geography.py).

Usage:
    /Users/bdoughty/opt/miniconda3/envs/tmg-scrape/bin/python plot_map.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
A = ROOT / "analysis"
PLOTS = A / "plots"
PLOTS.mkdir(exist_ok=True)

# Diverging blue<->red pair from the house palette (dataviz skill), gray
# midpoint -- "surprise" has a natural center (the global mean), so this is
# a polarity encoding, not a magnitude ramp.
BLUE, GRAY, RED = "#2a78d6", "#f0efec", "#e34948"
INK, INK_SECONDARY, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def main():
    city = pd.read_csv(A / "city_surprisal.csv").dropna(subset=["lat", "lon"])
    global_mean = pd.read_csv(A / "show_surprisal.csv").mean_surprisal_played_bits.mean()

    span = max(city.shrunk_mean_bits.max() - global_mean, global_mean - city.shrunk_mean_bits.min())
    lo, hi = global_mean - span, global_mean + span

    fig = go.Figure(go.Scattergeo(
        lon=city.lon, lat=city.lat,
        text=city.apply(lambda r: f"{r.city}<br>{int(r.n_shows)} shows<br>"
                                   f"{r.shrunk_mean_bits:.2f} bits (raw {r.raw_mean_bits:.2f})", axis=1),
        hoverinfo="text",
        marker=dict(
            size=np.clip(np.sqrt(city.n_shows) * 5.5, 5, 34),
            color=city.shrunk_mean_bits,
            colorscale=[[0, BLUE], [0.5, GRAY], [1, RED]],
            cmin=lo, cmax=hi,
            line=dict(width=0.6, color="rgba(11,11,11,0.25)"),
            colorbar=dict(
                title=dict(text="bits of<br>surprise", font=dict(size=11, color=INK_SECONDARY)),
                thickness=14, len=0.55, tickfont=dict(size=10, color=INK_SECONDARY),
            ),
        ),
    ))
    fig.update_geos(
        projection_type="natural earth",
        showland=True, landcolor="#eeece2", showocean=True, oceancolor=SURFACE,
        showcountries=True, countrycolor="#d8d0ba", showcoastlines=True, coastlinecolor="#d8d0ba",
        showframe=False, bgcolor=SURFACE,
    )
    fig.update_layout(
        title=dict(
            text="Where the surprising shows happen (marker size = shows played)",
            font=dict(size=17, color=INK, family="Helvetica, Arial, sans-serif"), x=0.01, xanchor="left",
        ),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        margin=dict(l=10, r=10, t=55, b=10),
        width=1250, height=650,
        font=dict(family="Helvetica, Arial, sans-serif"),
    )
    fig.write_image(str(PLOTS / "surprise_map_world.png"), scale=2)

    # US-focused inset -- most of the touring history is domestic, and the
    # world view compresses the interesting city-to-city variation.
    fig.update_geos(scope="usa", projection_type="albers usa")
    fig.update_layout(title=dict(text="Same data, continental US"))
    fig.write_image(str(PLOTS / "surprise_map_us.png"), scale=2)

    print(f"Wrote {PLOTS / 'surprise_map_world.png'} and surprise_map_us.png "
          f"({len(city)} cities plotted, global mean {global_mean:.2f} bits)")


if __name__ == "__main__":
    main()
