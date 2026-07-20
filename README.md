# The Mountain Goats live-show database

A scraped, tidy dataset of (nearly) every documented Mountain Goats live
performance, built from the fan-maintained wiki at
[themountaingoats.fandom.com](https://themountaingoats.fandom.com/wiki/Category:Live_Shows).

## Quickstart

```bash
PY=/Users/bdoughty/opt/miniconda3/envs/tmg-scrape/bin/python

$PY scrape.py fetch    # download/refresh the raw page cache (resumable, ~15 min cold)
$PY scrape.py build    # parse cache -> data/*.csv (seconds, offline)
$PY analyze.py         # starter analyses -> analysis/*.csv + printed report
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
| `raw_text` | untouched cell text, for auditing |

### `data/songs.csv` — one row per song

Play counts, first/last played dates, modal album, video counts.

### `data/show_notes.csv` — one row per Notes bullet per show

Free-text show notes from the wiki (guests, solo segments, encore
descriptions, banter). The `encore` column in performances.csv is derived
from these ("The encore was songs 19 through 23").

## Setlist prediction (`predict.py`)

Predicts P(song is played) for every (show, song) pair, walking history
chronologically so features only use pre-show information: decayed play
rate, recency, career rate, song age/new-material, current-tour rotation,
and special-show type. Logistic regression vs. an EWMA-play-rate baseline,
evaluated on a strict temporal split (test = 2023+): the model recovers
~59% of each setlist in its top-n predictions vs ~51% for the baseline.
Per-show test predictions land in `analysis/model_test_predictions.csv`.

## Analyses (`analyze.py`)

- shows and setlist coverage per year (`analysis/shows_per_year.csv`)
- tour summaries (`analysis/tours.csv`)
- per-song opportunity-adjusted play rates with empirical-Bayes shrinkage and
  a deep-cut flag (`analysis/song_stats.csv`) — methodology in
  [docs/deep_cut_notes.md](docs/deep_cut_notes.md)

## Caveats

- The wiki lists most but not all shows; early-90s coverage is spotty and
  many pages have no setlist. Use shows-with-setlists as the denominator.
- Song titles are canonicalized via wiki link targets, but unlinked entries
  (mostly covers and rarities) may still have spelling variants.
- `legacy/` holds two earlier scraper attempts and their outputs, superseded
  by this pipeline.
