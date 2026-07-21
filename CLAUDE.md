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
  Includes `encore_stats.csv`: per-song main-set vs. encore play counts and an
  empirical-Bayes-shrunk `encore_rate`, restricted to the 429 shows whose
  Notes actually specify an encore breakout (see Gotchas) — an unrestricted
  version would dilute every song's rate with shows where the split just
  isn't known.
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
- `geocode_cities.py` — offline city geocoding (geonamescache; no network).
  Matches by exact name only (after normalizing case/diacritics/
  abbreviations) — deliberately no fuzzy/population fallback, since an
  earlier version of that produced silently wrong coordinates (see
  Gotchas) → `analysis/city_coordinates.csv`. `data/city_overrides.csv`
  (city,region,lat,lon) is hand-maintained ground truth for places too
  small for geonamescache — always wins over the automatic match, since a
  human-supplied coordinate isn't the same risk as an algorithmic guess.
  Pre-populated with every currently-unmatched (city,region) pair and
  blank lat/lon; fill in whichever you care about, leave the rest blank.
- `geography.py` — empirical-Bayes-shrunk surprisal per city, pooled by
  real geographic proximity (DBSCAN + haversine distance on the geocoded
  coordinates, 25km radius), not administrative state/country lines — a
  city with no real neighbor in the tour history shrinks straight to the
  global mean instead of a fake "region of one." Also computes
  region_surprisal.csv (state/country) as its own separate, unrelated
  table (not used as anyone's prior). → `analysis/city_surprisal.csv`
  (`shrunk_mean_bits` = metro-cascaded, `shrunk_flat_bits` = flat
  shrink-to-global, for comparison), `analysis/region_surprisal.csv`,
  `analysis/metro_surprisal.csv` (the clusters themselves, with members).
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
  `city` is normalized through `CITY_ALIASES` (parse_title() in scrape.py)
  for known same-place variants ("New York City" → "New York") — add new
  entries there if `geography.py`'s output shows an obvious near-duplicate
  city splitting one place's history (verify against actual show counts
  first, like the existing entries' comments do; not every look-alike is
  a duplicate — "Columbia"/"West Columbia", SC are genuinely different
  towns, checked and correctly NOT aliased).
- `data/performances.csv` — one row per song per show, join on `show_id`.
  Use `song_canonical` (or `song_key`) for grouping; `song_title` is the raw
  display text of that performance. `is_cover` is sourced from the wiki's
  own `Category:Covers` (see below), not text-sniffed from notes.
  `solo_segment` (bool) is True for a whole-show `is_solo`, or for a song
  inside a full-band show's within-show solo break, parsed from Notes prose
  by `solo_segment_map()` — see Wiki page structure below. It's for display
  only; not a model feature (see Gotchas).
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
- **Within-show solo segments are also prose-only**, in a full-band show's
  Notes ("Songs 7 through 9 were played by John Darnielle solo", "John's
  solo set was songs 7-9") — `solo_segment_map()` parses these into the
  `solo_segment` column, same position-keying as `encore_map()`. A single
  note frequently packs a solo range together with a second, non-solo range
  in the same sentence ("...solo, and songs 10-14 were played by Kaki King"
  / "John played songs 1-5 solo. Peter Hughes joined him for songs 6-11.")
  so matching is done per-clause (split on sentence/semicolon boundaries and
  ", and ") — a clause counts as solo only if "solo" appears in that same
  clause, so the guest-only range doesn't get swept in too.
- Song cells: title is quoted with a wiki link; parentheticals hold guest
  info, cover attribution, and external video links (YouTube/Vimeo links go
  to `video_urls`, the rest of the parenthetical to `note`; `raw_text`
  preserves everything). If the quoted text and the wiki link disagree
  entirely, the link is a performer/album, not the song — trust the quotes.
  The disagreement check normalizes "#N" / "No. N" / spelled-out numerals
  ("Number One") to a common "number N" form first (`numeral_normalize()`),
  so a numbering-convention difference alone doesn't look like a real
  disagreement and strand the song without its wiki slug (a real bug, caught
  and fixed — "Sax Rohmer #1" and the wiki's own redirect-page spelling
  "Sax Rohmer Number One" were splitting into two separate songs).
  **A song's link `title=` attribute can be disambiguated** (e.g. link text
  "Get Lonely", `title="Get Lonely (Song)"`, because a same-named album page
  exists) — 20 songs do this. `clean_song_cell()`'s note-construction must
  strip what's actually in the raw cell text (the quoted string / the
  anchor's own visible text), never the resolved `title`, or the original
  link text gets stranded as a bogus note (a real bug, caught and fixed —
  it had been happening on every performance of all 20 songs).
- Song identity: `song_key` = link slug with underscores→spaces (else display
  text), case-unified across wiki redirect variants, *and* numeral-spelling
  variants (`numeral_normalize()` again, in `build()` this time — catches
  same-song wiki redirects the per-cell check above doesn't, like "Sax
  Rohmer Number 1" vs. the redirect page "Sax Rohmer Number One", which have
  genuinely different slugs). Always group on `song_key`, never raw
  `song_title`.
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
  `shows.city` (post-`CITY_ALIASES` normalization) — check
  `analysis/city_coordinates.csv`'s unmatched rows after any change to
  city/venue parsing, and scan for new same-region near-duplicate city
  names (see geography.py's docstring for the detection query used to
  find the ones already fixed).
- `geocode_cities.py`'s matcher used to fall back to "the biggest city in
  the state/country" when there was no exact name match — silently wrong
  coordinates for 74 towns (Pittsboro NC → Charlotte; Montreal → Toronto;
  population-based, completely ignoring actual distance). Fixed to
  exact-match-only; a town too small for geonamescache is now correctly
  left uncoded rather than mislocated. Don't reintroduce a fuzzy fallback
  without a real distance check.
- **Known limitation, deliberately left as-is for now**: `geography.py`'s
  DBSCAN metro-clustering has no cluster-diameter bound — it only checks
  that *adjacent* hops are ≤25km, not how far apart a chain's endpoints
  end up (a standard single-linkage/density-chaining property, not a bug).
  After adding manual coordinate overrides, this became real: Durham's
  cluster now reaches Pittsboro (39.5km away, direct) and Graham (45.9km)
  through Carrboro and Saxapahaw as stepping-stones. Graham's own raw rate
  (3.12 bits, near the global mean) inherits Durham's much higher cluster
  prior and now outranks San Francisco's robust 57-show estimate on a
  4-show sample — exactly the "random unsurprising town borrows a bigger
  city's reputation" failure mode flagged when this design was chosen (see
  the geography.py docstring). Currently concentrated around Durham
  specifically (JD's home turf, so genuinely denser local coverage than
  the rest of the tour history) rather than a global problem, but it will
  keep happening as more small towns get geocoded there. The principled
  fix, if/when this gets revisited: switch to a **cluster-diameter cap**
  (every pair inside a cluster ≤25km, i.e. complete-linkage) instead of
  DBSCAN's adjacent-hop-only criterion — Boston/Cambridge/Somerville-style
  tight scenes still merge fine under an all-pairs test, but Graham could
  never join Durham's cluster no matter how many stepping-stones exist.
- Both `encore` and `solo_segment` are real per-song data (parsed from Notes
  prose, see above), but neither is currently a feature in predict.py's
  model — it predicts "is this song played," not "where in the show," and
  encore/solo-segment status isn't known before the show happens anyway (it's
  an outcome, not a pre-show predictor) — so it's descriptive/display-only,
  same as `city_surprisal.csv` vs. the predictive `city_song_rate`.
  `encore_stats.csv` (analyze.py) is restricted to the 429 (of 1317
  setlisted) shows with a known encore breakout — most shows' Notes never
  call out an encore at all, so an unrestricted rate would just be diluted
  by shows where the split is unknown, not shows with no encore.
