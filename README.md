# every-stupid-thing

A scraped, tidy dataset of (nearly) every documented Mountain Goats live
performance, built from the fan-maintained wiki at
[themountaingoats.fandom.com](https://themountaingoats.fandom.com/wiki/Category:Live_Shows).

**[Live site](https://bdoughty.github.io/every-stupid-thing/)** (setlist/song
search, the "next show" predictor, and the surprise/geography analysis) ·
**[Full report (PDF)](https://bdoughty.github.io/every-stupid-thing/report.pdf)**

## Quickstart

```bash
PY=/Users/bdoughty/opt/miniconda3/envs/tmg-scrape/bin/python

$PY scrape.py fetch      # download/refresh the raw page cache (resumable, ~15 min cold)
$PY scrape.py build      # parse cache -> data/*.csv (seconds, offline)
$PY analyze.py           # starter analyses -> analysis/*.csv + printed report
$PY predict.py           # setlist-prediction model -> analysis/*.csv
$PY timeseries.py        # song popularity-over-time -> analysis/song_timeseries*.csv
$PY geocode_cities.py    # offline city geocoding -> analysis/city_coordinates.csv
$PY geography.py         # shrunk city/region surprisal -> analysis/{city,region}_surprisal.csv
$PY plot_report.py       # report figures -> analysis/plots/*.png
$PY plot_map.py          # surprise maps -> analysis/plots/surprise_map_*.png
$PY build_pdf.py         # report.pdf (mirrors report.md)
$PY export_webapp.py     # bundle webapp/data.json from data/ + analysis/
$PY build_webapp.py      # splice data.json into webapp/index.html
```

Re-running `fetch` later only downloads new or edited pages (it compares
wiki revision ids), so keeping the dataset current is cheap.

## The data

Load with pandas and join on `show_id`:

```python
import pandas as pd
shows = pd.read_csv("data/shows.csv", parse_dates=["date"])
perfs = pd.read_csv("data/performances.csv", parse_dates=["date"])
df = perfs.merge(shows, on="show_id", suffixes=("", "_show"))
```

### `data/shows.csv` — one row per show

| column | meaning |
|---|---|
| `show_id` | wiki page slug, primary key |
| `date`, `date_raw`, `year`, `month`, `day` | `date` is ISO and null when the wiki only knows a partial date (`1992-xx-xx`) |
| `venue`, `location`, `city`, `region` | parsed from the page title; `region` is usually a US state code or country |
| `tour`, `all_tours` | from wiki categories (e.g. "Jenny From Thebes Tour 2023"); null for one-offs |
| `show_type` | "radio session", "festival", "tv appearance", or null for a regular show |
| `act` | "The Mountain Goats" or a side project ("The Extra Glenns") |
| `region_category` | the wiki's `<State> live shows` category |
| `has_video`, `has_audio`, `incomplete_setlist`, `incomplete_article` | wiki flags |
| `is_solo` | whole-show John Darnielle solo set; conservative/high-precision, not exhaustive — see CLAUDE.md |
| `n_songs` | setlist length as scraped (0 = no setlist recorded) |
| `pageid`, `revid`, `url`, `title` | wiki bookkeeping |

### `data/performances.csv` — one row per song per show

| column | meaning |
|---|---|
| `show_id`, `date`, `year` | join keys / convenience |
| `seq` | position in the full show (1..n, continues through encores) |
| `order_raw` | the number printed in the wiki's setlist table |
| `encore` | 0 = main set, 1 = first encore, ... |
| `song_title` | display text for this performance |
| `song_key`, `song_canonical` | canonical song identity — group by these |
| `album` | album column from the setlist table ("Unreleased" is meaningful) |
| `note` | leftover annotations: guests, cover attribution, alternate titles |
| `video_urls` | pipe-separated YouTube/Vimeo links for this performance |
| `is_cover` | from the wiki's own `Category:Covers` — see below |
| `solo_segment` | John solo for this song — whole-show `is_solo`, or a full-band show's parsed within-show solo break; display-only, not a model feature |
| `raw_text` | untouched cell text, for auditing |

### `data/songs.csv` — one row per song

Play counts, first/last played dates, modal album, video counts, `is_cover`.

### `data/show_notes.csv` — one row per Notes bullet per show

Free-text show notes from the wiki (guests, solo segments, encore
descriptions, banter). The `encore` column in performances.csv is derived
from these ("The encore was songs 19 through 23").

## Setlist prediction (`predict.py`)

Predicts P(song is played) for every (show, song) pair, walking history
chronologically so features only use pre-show information: decayed play
rate, recency, career rate, song age/new-material, current-tour rotation,
special-show type, whole-show solo set (`is_solo`), and a shrunk city-level
play rate (`city_song_rate` — empirical-Bayes toward the song's own recent
rate, so cities with little history collapse to "no city effect"; see
`geography.py` below for the descriptive version of this same question).
Covers are excluded from the candidate universe entirely (predictions are
scoped to the canonical catalog). Logistic regression vs. an EWMA-play-rate
baseline, evaluated on a strict temporal split (test = 2023+): the model
recovers ~59% of each setlist in its top-n predictions vs ~51% for the
baseline. Per-show test predictions land in
`analysis/model_test_predictions.csv`.

Also exports: `model_coefficients.csv`; `surprising_plays.csv` (individual
songs the model least expected); `next_show_snapshot.csv` (prediction for
"the next show, continuing the current tour" — feeds the webapp);
`historical_tour_example.csv` (a data-driven, non-album-cycle prediction
example, picked automatically from 2014 tours, for checking the model
isn't just riding album hype); `show_surprisal.csv` / `tour_surprisal.csv`
(per-show and per-tour "how surprising was this setlist," in bits, from
each song's pre-show probability); `most_surprising_concerts.csv` (same,
filtered to real setlists of 10+ songs, so a 1-song guest cameo can't
trivially top the list).

## Geography (`geocode_cities.py`, `geography.py`, `plot_map.py`)

Does where a show happens predict how surprising the setlist is? Cities are
geocoded against an offline database ([geonamescache](https://pypi.org/project/geonamescache/),
matched by exact name only — not fabricated, not fuzzy-matched to "the
nearest big city"), covering 91.6% of setlisted shows. Places too small for
that database can be filled in by hand in `data/city_overrides.csv`
(city, region, lat, lon) — human-verified coordinates always win over the
automatic match. `geography.py`
computes empirical-Bayes-shrunk mean surprisal per city, pooled by real
geographic proximity (DBSCAN + haversine distance, 25km radius) rather
than state/country lines — Boston/Cambridge/Somerville and Durham/Chapel
Hill/Carrboro pool as genuine metro scenes, while a place with no real
neighbor in the tour history (Watkins Glen, NY; Pittsboro, NC) keeps its
own distinct signal instead of borrowing a same-state city's reputation.
Most city-level samples are thin (median 2 shows), so shrinkage matters a
lot. Outputs: `analysis/city_surprisal.csv` (city-level, both the
metro-cascaded and flat-to-global values), `analysis/metro_surprisal.csv`
(the clusters themselves), `analysis/region_surprisal.csv` (state/country,
a separate descriptive table). `plot_map.py` renders the city-level result
as a world map and a US inset (`analysis/plots/surprise_map_*.png`), using
plotly's built-in basemap (no tile server, no API key).

## Analyses (`analyze.py`)

- shows and setlist coverage per year (`analysis/shows_per_year.csv`)
- tour summaries (`analysis/tours.csv`)
- per-song opportunity-adjusted play rates with empirical-Bayes shrinkage and
  a deep-cut flag (`analysis/song_stats.csv`) — methodology in
  [docs/deep_cut_notes.md](docs/deep_cut_notes.md)
- main-set vs. encore play counts per song, with a shrunk `encore_rate`
  (`analysis/encore_stats.csv`) — restricted to the 429 shows whose Notes
  actually spell out the encore breakout, so it's not diluted by shows where
  the split just isn't recorded

## Song popularity over time (`timeseries.py`)

Trailing 2-year play-rate curves per song, sampled monthly — a Google-Ngram
analogue for setlists. Auto-picks illustrative trajectories (steady
classics, rise-and-plateau, decline, fall-and-revival) into
`analysis/timeseries_examples.json`; full series in
`analysis/song_timeseries.csv` (monthly) and `_compact.csv` (quarterly,
rounded, used by the webapp).

## Report (`report.md` / `report.pdf`)

A written summary of the whole project — the scraping gotchas, the deep-cut
and trajectory analyses, and the prediction model's results — with the
figures from `plot_report.py` embedded. `build_pdf.py` renders the PDF
directly via reportlab (no pandoc/LaTeX dependency); keep it in sync with
`report.md` by hand if you edit either.

## Webapp (`webapp/`)

A self-contained static page (data embedded inline, no server) with three
tools: search a song for its video links, cover flag, and popularity
trajectory; browse a show's setlist by encore with its wiki notes and
per-show surprise score, or jump straight to one of the most surprising
concerts; and see the prediction model's guess at the next show plus
city-level "local favorites." Rebuild with `export_webapp.py` then
`build_webapp.py` after refreshing the data; `webapp/app_template.html` is
the hand-edited source, `webapp/index.html` is the generated, publishable
output. (Not yet in the webapp: the geography map — it's a report figure
for now, not an embedded interactive one.)

## Caveats

- The wiki lists most but not all shows; early-90s coverage is spotty and
  many pages have no setlist. Use shows-with-setlists as the denominator.
- Song titles are canonicalized via wiki link targets, but unlinked entries
  (mostly rarities) may still have spelling variants.
- `is_solo` is conservative/high-precision, not exhaustive — it likely
  under-flags true solo shows the wiki didn't call out as exceptional,
  especially pre-2002.
- City-level analyses cover 91.6% of setlisted shows and are inherently
  thin per city (median 2 shows) — that's what the empirical-Bayes
  shrinkage in `geography.py` and the `city_song_rate` model feature are
  for. `city` is normalized for known same-place naming variants
  (`CITY_ALIASES` in scrape.py — "New York City" → "New York"); a new
  variant slipping through would silently split that place's history.
- `encore` (main set / which encore) and `solo_segment` (whole-show or
  within-show) are both real per-song data, parsed from Notes prose — but
  neither is a feature in the prediction model, since encore/solo status
  isn't known before a show happens (it's an outcome, not a predictor).
- `legacy/` holds two earlier scraper attempts and their outputs, superseded
  by this pipeline.
