# The Mountain Goats, Every Night: Scraping and Modeling 34 Years of Setlists

*A data project built on [themountaingoats.fandom.com](https://themountaingoats.fandom.com/wiki/Category:Live_Shows), the fan-run wiki documenting (nearly) every Mountain Goats live show.*

## The dataset

The wiki lists 1,579 live shows from 1992-05-31 through 2026-05-26 — solo John
Darnielle sets, full-band tours, radio sessions, festival slots, and Extra
Glenns/Extra Lens side-project shows. 1,317 of those have a documented
setlist: 22,699 individual song performances across 845 distinct songs,
spanning 97 named tours. 1,739 performances have at least one linked
YouTube/Vimeo video.

Getting from "a wiki category page" to those numbers took a few real fixes:

- **Tour isn't in the page text — it's a wiki category** (e.g. `Peter Hughes
  Farewell Tour 2024`). Two earlier scraper attempts both missed this and
  scraped `tour: null` for every show.
- **Encores aren't marked in the setlist table.** They're stated in prose
  in the Notes section — *"The encore was songs 19 through 23"* — referencing
  the table's printed order numbers. A small parser turns ~30 recurring
  phrasings of that sentence into a per-song encore number.
- **Song identity needed real normalization.** The wiki links each song to
  its own page, but the same song shows up under underscore-joined slugs,
  space-joined display text, and case variants of both (`Southwood_Plantation_Road`
  vs. `Southwood Plantation Road`, `Broken To Begin With` vs. `Broken to
  Begin With`). Left unmerged, this silently splits a song's play count
  across multiple rows — a real bug caught mid-project, where "Southwood
  Plantation Road" initially showed 1 play instead of its real count, 165.
- **Covers needed their own source of truth.** Sniffing setlist notes for
  the word "cover" only catches about a dozen of them — the wiki actually
  maintains its own `Category:Covers`, which tags 156 of the songs played
  live here (Thin Lizzy, Bowie, Fall Out Boy, and 153 others). Every song
  and performance carries an `is_cover` flag from that category, and the
  prediction model excludes covers from its candidate universe entirely —
  a Thin Lizzy cover played once isn't a "deep cut" in the same sense a
  rarely-played original is.

The pipeline is two-stage: `fetch` downloads and locally caches every wiki
page via the MediaWiki API (resumable, and incremental on later runs — it
only re-fetches pages whose revision id changed), and `build` parses the
cached HTML into `shows.csv` / `performances.csv` / `songs.csv` entirely
offline. Re-running `fetch` after a new tour is a few-minute job.

## What gets played, and what doesn't

<img src="analysis/plots/song_frequency.png" width="100%">

"This Year" and "No Children" anchor the top of the list — both a majority
of all shows since they entered rotation. The deep-cuts panel isn't just
"least played ever" (most of the 845 songs have only 1–2 plays and aren't
interesting on their own); it's the bottom of an **opportunity-adjusted,
shrinkage-smoothed play rate** among songs that have had at least 20 chances
to be played since their live debut — so a song is only called a deep cut if
it's been consistently passed over, not just recently written. Full
methodology in [docs/deep_cut_notes.md](docs/deep_cut_notes.md); results in
[analysis/song_stats.csv](analysis/song_stats.csv).

## Song popularity over time

Treating each song's live history like a word's frequency in Google Ngrams —
a trailing 2-year play rate sampled monthly — turns 34 years of setlists into
trajectories: songs sit at zero before they're written, rise as they enter
rotation, plateau, decline, and sometimes come back.

<img src="analysis/plots/song_trajectories.png" width="100%">

These four groups were picked automatically from the computed trajectories,
not hand-selected: the four most-played songs overall (*steady classics*);
songs that debuted in the back half of the catalog's history and have
climbed straight into rotation since (*rise and plateau* — this is the
"allele sweep" pattern, a song appearing from nothing and taking hold); the
biggest gap between early-career peak and recent play rate (*decline*); and
songs with a real trough between two active periods (*fall and revival*).

The decline panel caught something the wiki itself corroborates: **"Going to
Georgia"** was a fixture through the mid-2000s, then drops off hard — a 2012
show note has John Darnielle explaining, mid-show, that he considers it a
"bullshit song" with an "asshole" narrator and doesn't want to play it
again. The data shows the falloff independent of that anecdote; the anecdote
just explains it.

## Predicting the setlist

Framed as one binary outcome per (show, song) pair — is this the night song
X gets played? — every one of the 1,310 dated, setlisted shows becomes a
training example, using only information available *before* that show:
how recently and how often the song's been played, how it's fared on the
current tour, its age, and whether it's a shortened-set special appearance
(radio/festival/TV). Full feature list and code in
[predict.py](predict.py).

A logistic regression on those features is compared against a simple but
strong baseline — an exponentially-decayed play rate alone — on a strict
temporal split (train through 2022, test on the 207 shows from 2023
onward, so nothing in the test set could leak backward into training):

| model | log-loss | Brier score | top-*n* setlist recovery |
|---|---|---|---|
| Baseline (decayed play rate) | 0.0822 | 0.0195 | 51.1% |
| Logistic regression | **0.0673** | **0.0166** | **59.4%** |

*Top-*n* setlist recovery*: for each show, take the model's *n* highest-probability
songs, where *n* is the actual number of songs played that night, and measure
what fraction were right. On the most recent show in the data:

<img src="analysis/plots/example_prediction.png" width="100%">

### What actually predicts a setlist

<img src="analysis/plots/model_weights.png" width="100%">

Recency dominates everything else combined — if a song was played a few
shows ago, it's overwhelmingly likely to come back soon; if it's been a
while, it's overwhelmingly likely to sit out. That's the visible mechanism
behind the tour "rotation": John Darnielle appears to work from a
live pool of songs and cycle through it rather than drawing independently
from the whole catalog each night, which is exactly what `tour_rate`
(how often a song's been played *this tour specifically*) picking up a real,
independent signal on top of recency confirms. Two smaller and genuinely
interesting effects: conditional on recency, brand-new material is actually
*less* sticky than catalog average (album-cycle songs burn hot then get
rotated out faster than an old favorite would), and radio/festival/TV
appearances reliably favor hits over deep cuts, as you'd expect from a
shorter set.

### Is this just tracking album hype?

The 2026 example above is a new-album tour, and `new_material` is a real
feature in the model — worth checking whether the strong recovery number is
mostly "predict whatever's newest" in disguise. To test that, I picked a
tour the model never got to see labeled as an album cycle: among all 2014
tours with at least 8 shows, the one whose *actually-played* songs had the
lowest share of new material — automatically, not by hand — is the **Twin
Inhuman Highway Fiends Tour 2014**, squarely in the two-and-a-half-year gap
between *Transcendental Youth* (2012) and *Beat the Champ* (2015):

<img src="analysis/plots/example_comparison.png" width="100%">

The between-albums tour is visibly *harder* to predict — 55.1% top-*n*
recovery tour-wide, versus 59.4% for the full 2023+ test era — despite 2014
being in-sample (the model trained on it directly, which should make it
look artificially *easier*, not harder). That's a reasonable answer: without
a record to promote, the setlist draws more evenly across a wider pool of
similarly-loved catalog songs, so there's genuinely more entropy to predict,
not less. The model isn't just riding hype; if anything, album cycles are
the *easy* case, because they concentrate probability mass onto a
predictable set of newly-written songs.

### When the model gets surprised

The flip side of a good model is a good list of its misses — nights the
model gave a song almost no chance, and it got played anyway:

<img src="analysis/plots/surprising_plays.png" width="100%">

These are basically a machine-generated "rarities and one-offs" list:
songs that came back from years of dormancy for a single night, often at a
show with some specific occasion (last night of a tour leg, a hometown show,
a request). [analysis/surprising_plays.csv](analysis/surprising_plays.csv)
has the full ranked list if you want to go looking for what made those
nights special.

### How surprising was each *setlist* as a whole?

The plays above are the most surprising individual songs; zooming out, the
same pre-show probabilities give a surprise score for an entire night's
setlist — the average bits of information (−log₂ *p*) across the songs
actually played, using only what the model knew walking in. Low = a
thoroughly expected set; high = a night that defied the rotation:

<img src="analysis/plots/surprisal_over_time.png" width="100%">

Touring is bursty — weeks of shows, then months of silence — so this is
aggregated per tour (the natural unit) rather than smoothed over a fixed
show-count window, which would saw up and down at tour boundaries. The
tours with the highest average surprise are almost all **solo, stripped-down
tours** — Winter Solo Tour 2024, the Ghost Cave Incubation Chamber solo
non-tour, All Roads Lead to Lincoln Solo Mini-Tour — which make sense: a
solo acoustic set draws from a noticeably different, more idiosyncratic pool
than a full-band show, and the model has no explicit "solo show" feature to
account for it (only the coarser `is_special_show` flag for radio/
festival/TV). That's a concrete, data-backed suggestion for the next
modeling iteration. Full per-show numbers in
[analysis/show_surprisal.csv](analysis/show_surprisal.csv), per-tour
rollups in [analysis/tour_surprisal.csv](analysis/tour_surprisal.csv), and
every show's surprise score is browsable live in the
[webapp](webapp/index.html)'s Show Browser tab.

## Caveats

- The wiki is fan-maintained and lists *most*, not all, shows — coverage is
  noticeably better from the mid-2000s on. Analyses here condition on
  "shows with a documented setlist," not "shows that happened."
- The predictive model has no access to anything outside setlist history —
  no lyrics, no album themes, no explicit "songs currently being toured."
  `tour_rate` and `song_age` are proxies for that; a real album↔tour mapping
  would likely improve on new-material handling.
- Song identity is deduplicated via wiki link slugs where available; a
  handful of never-linked rarities may still have unmerged spelling
  variants.
- The model has no "solo show" feature, which the surprise-over-time
  analysis suggests it should — solo/stripped-down tours are consistently
  the hardest to predict.

## Reproducing this

```bash
PY=/Users/bdoughty/opt/miniconda3/envs/tmg-scrape/bin/python
$PY scrape.py fetch && $PY scrape.py build   # refresh the raw data
$PY analyze.py                                # deep cuts, tour summaries
$PY predict.py                                # the setlist model
$PY timeseries.py                             # popularity-over-time series
$PY plot_report.py                            # this report's figures
```

Full data dictionary in [README.md](README.md); wiki-scraping gotchas and
repo layout in [CLAUDE.md](CLAUDE.md).
