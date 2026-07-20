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
- `analyze.py` — starter analyses; writes `analysis/*.csv` and prints a report.
- `data/shows.csv` — one row per show. Key: `show_id` (wiki page slug).
- `data/performances.csv` — one row per song per show, join on `show_id`.
  Use `song_canonical` (or `song_key`) for grouping; `song_title` is the raw
  display text of that performance.
- `data/songs.csv` — per-song aggregates.
- `docs/deep_cut_notes.md` — methodology notes on defining "deep cuts"
  (opportunity-adjusted shrunk play rates); `analyze.py` implements this.
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
- Song identity: `song_key` = link slug with underscores→spaces (else display
  text), case-unified across wiki redirect variants. Always group on
  `song_key`, never raw `song_title`.

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
