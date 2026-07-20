# The Mountain Goats live-show database

Scrapes every page in `Category:Live_Shows` on themountaingoats.fandom.com
into tidy, pandas-friendly tables for setlist analysis.

## Environment

Use the `tmg-scrape` conda env (python 3.11, requests/bs4/pandas):

    /Users/bdoughty/opt/miniconda3/envs/tmg-scrape/bin/python

## Layout

- `scrape.py` — the scraper. Two stages:
  - `scrape.py fetch` downloads each show page (rendered HTML + categories +
    revid via the MediaWiki API) into `data/raw/<pageid>.json`. Resumable and
    incremental: pages whose revid is unchanged are skipped, so re-running to
    pick up new shows is cheap. Polite by default (0.4 s delay).
  - `scrape.py build` parses the cached JSON into `data/*.csv`. Never needs
    the network — iterate on parsing freely.
  - `fetch` also caches `data/raw/_covers.json`, the wiki's
    `Category:Covers` membership — the source of the `is_cover` flag.
- `analyze.py` — starter analyses; writes `analysis/*.csv` and prints a report.
- `predict.py` — the setlist-prediction model (logistic regression vs. an
  EWMA baseline), including `is_solo` and a shrunk `city_song_rate` feature.
  Also exports: `model_coefficients.csv`, `surprising_plays.csv` (biggest
  individual misses), `next_show_snapshot.csv` ("if the next show continues
  the current tour" — feeds the webapp predictor), `historical_tour_example.csv`
  (a data-driven non-album-cycle example, for sanity-checking the model
  isn't just riding album hype), `show_surprisal.csv` / `tour_surprisal.csv`
  (per-show and per-tour "how surprising was this setlist" in bits, from
  each song's pre-show probability), `most_surprising_concerts.csv` (same,
  filtered to real setlists so a 1-song guest cameo can't top the list).
- `geocode_cities.py` — offline city geocoding (geonamescache; no network,
  no fabricated coordinates) → `analysis/city_coordinates.csv`.
- `geography.py` — empirical-Bayes-shrunk surprisal per city/region (does
  geography predict how surprising a setlist is — the SF/NC finding) →
  `analysis/city_surprisal.csv` / `region_surprisal.csv`.
- `plot_map.py` — world + US surprise maps via plotly's built-in basemap
  (no tile server/API key; static PNG export needs `kaleido`).
- `timeseries.py` — rolling 2yr play-rate per song (Ngram-style), monthly +
  quarterly-compact versions, plus auto-picked illustrative trajectories.
- `plot_report.py` / `build_pdf.py` — report figures and the PDF build
  (reportlab, no pandoc/LaTeX dependency). `report.md` is hand-written and
  must be kept in sync with `build_pdf.py` by hand if either changes.
- `export_webapp.py` / `build_webapp.py` — bundle `data/` + `analysis/` into
  `webapp/data.json`, then splice it into `webapp/app_template.html` (the
  hand-edited source) to produce the publishable `webapp/index.html`.
- `data/shows.csv` — one row per show. Key: `show_id` (wiki page slug).
  `is_solo` is sourced from a strict "solo show" note pattern or a
  tour name containing "Solo" — conservative, not exhaustive (see Gotchas).
- `data/performances.csv` — one row per song per show, join on `show_id`.
  Use `song_canonical` (or `song_key`) for grouping; `song_title` is the raw
  display text of that performance. `is_cover` is sourced from the wiki's
  own `Category:Covers` (see below), not text-sniffed from notes.
- `data/songs.csv` — per-song aggregates, including `is_cover`.
- `docs/deep_cut_notes.md` — methodology notes on defining "deep cuts"
  (opportunity-adjusted shrunk play rates); `analyze.py` implements this.
  Deep-cut eligibility excludes covers (a cover played once isn't a
  "passed-over" original).
- `legacy/` — earlier scraper attempts and their outputs. Superseded; kept
  for reference only. Don't extend these.

## Wiki page structure (what the parser relies on)

- Page titles encode metadata: `YYYY-MM-DD - Venue - City, ST`. Dates can be
  partial (`1992-xx-xx`); the venue segment is optional and may itself
  contain ` - `.
- **Tour lives in the page categories**, not the page body (e.g.
  `Peter_Hughes_Farewell_Tour_2024`). Other categories: `<State>_live_shows`,
  bare years, and flags (`Shows_with_video`, `Incomplete_Setlists`,
  `Incomplete_articles`). `classify_categories()` sorts these out; anything
  unrecognized lands in `tour`, so when the wiki grows a new non-tour
  category, add it to `FLAG_CATEGORIES` / `SHOW_TYPE_CATEGORIES` /
  `ACT_CATEGORIES` / `IGNORE_CATEGORIES` and re-run `build` (no re-fetch
  needed).
- Setlists are usually a `wikitable` with Order/Song/Album columns (some
  tables omit Order — the parser reads the header row); pre-~2000 pages
  sometimes use plain `<ol>`/`<ul>` lists (both handled).
- **Encores are NOT marked in the setlist table.** They live as prose in the
  Notes section ("The encore was songs 19 through 23"), keyed to the printed
  order numbers; `encore_map()` parses this into the `encore` column. Notes
  bullets are also exported to `data/show_notes.csv`.
- Song cells: title is quoted with a wiki link; parentheticals hold guest
  info, cover attribution, and external video links (YouTube/Vimeo links go
  to `video_urls`, the rest of the parenthetical to `note`; `raw_text`
  preserves everything). If the quoted text and the wiki link disagree
  entirely, the link is a performer/album, not the song — trust the quotes.
  **A song's link `title=` attribute can be disambiguated** (e.g. link text
  "Get Lonely", `title="Get Lonely (Song)"`, because a same-named album page
  exists) — 20 songs do this. `clean_song_cell()`'s note-construction must
  strip what's actually in the raw cell text (the quoted string / the
  anchor's own visible text), never the resolved `title`, or the original
  link text gets stranded as a bogus note (a real bug, caught and fixed —
  it had been happening on every performance of all 20 songs).
- Song identity: `song_key` = link slug with underscores→spaces (else display
  text), case-unified across wiki redirect variants. Always group on
  `song_key`, never raw `song_title`.
- The wiki has a standalone `Category:Covers` (174 song pages, 156 of which
  match a song we've actually seen played live) — far more complete than
  sniffing "(cover)" out of setlist notes (~12 hits). This is the source of
  `is_cover`; `predict.py` filters covers out of the prediction candidate
  universe entirely, not just post-hoc.

## Gotchas

- Many shows legitimately have no setlist (`n_songs == 0`,
  often `incomplete_article=True`) — filter before rate analyses, and use
  "shows with setlists" as the denominator for play-rate opportunities.
- `date` is null when the wiki only knows the year/month; `date_raw`/`year`
  are still populated.
- Song identity: prefer `song_key` (wiki link slug) over display text —
  display titles vary ("Standard Bitter Love Song #5" vs "... Number 5").
  Unlinked songs fall back to display text and may still have variants.
- The wiki is incomplete and community-maintained: it lists most but not all
  shows, and setlist coverage is much better after ~2005.
- `is_solo` (`SOLO_NOTE_RE`) only matches an explicit "solo show" note or a
  tour named "...Solo...". It deliberately does NOT match "John's solo SET
  was songs 7-9" (a segment of an otherwise full-band show) — that's a
  different, much more common note pattern (480 shows) meaning something
  else. Treat `is_solo` as "known-solo," not exhaustive — pre-2002 shows in
  particular are likely under-flagged, since a solo show wasn't noteworthy
  before a full-time backing band was the norm.
- City/region matching for geocoding and `city_song_rate` is exact-string on
  `shows.city` — "New York" and "New York City" are different keys unless
  normalized upstream; check `analysis/city_coordinates.csv`'s unmatched
  rows after any change to city/venue parsing.
