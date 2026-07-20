# Notes and robustness tips
# - Fandom infobox labels vary across wikis; the script captures common fields but you may need to adapt label names after inspecting a few pages.
# - Setlists can vary: multiple encores, subheadings, or notes. The parser marks an encore when it sees a heading containing “Encore”, and continues numbering positions across the whole show.
# - If some pages don’t use a “Setlist” heading, you may need to add additional heuristics (e.g., search for the first ordered list after headings like “Set list”, “Songs”, etc.).
# - If you want to parse raw wikitext instead, you can switch to action=query&prop=revisions&rvprop=content and parse with mwparserfromhell; that can be more precise if the wiki uses consistent templates.

# After scraping: modeling ideas
# - Baseline: estimate song probabilities as frequency over the last N shows (or exponential decay on recency).
# - Venue- or city-specific effects: add hierarchical structure by city/venue and tour.
# - Time effects: include AR terms for “streaks,” or model logistic regression with time-varying coefficients per song.
# - Conditioning: include whether an album is being toured; tour-level random intercepts/slopes.

# Great project. The Mountain Goats wiki runs on Fandom (MediaWiki), so the cleanest, most robust way to scrape is via the MediaWiki API rather than scraping raw HTML pages. You can still parse the HTML rendered by the API to extract setlists and infobox fields. Below is a complete, rate-limited Python scraper that:

# - Enumerates all pages in Category:Live_Shows via the API
# - Fetches each page’s rendered HTML and last revision ID
# - Extracts show metadata from the infobox (date, tour, venue, location if present)
# - Extracts setlists, supports multiple encores, captures track order and notes
# - Writes a normalized SQLite database (shows, songs, show_songs), plus CSV exports
# - Respects API rate-limits, includes a proper User-Agent, and supports resuming by lastrevid

# Before you run it
# - Be considerate and throttle requests (built-in delay).
# - It’s best to use the MediaWiki API (as in this script).
# - Install dependencies:
#   pip install requests beautifulsoup4 python-dateutil requests-cache tqdm

# Script: scrape_tmg_fandom.py



Short answer

Common English meaning: A “deep cut” is a non-obvious song from an artist’s catalog—usually not a single, seldom played live, and mainly known to dedicated fans.
Functional definition (plays-based): A song in the low tail of live-play likelihood. For example:
Lifetime deep cut: songs in the bottom 20–30% by play rate across the artist’s career (after adjusting for how long they’ve had the chance to be played).
Tour-era deep cut: songs in the bottom 20–30% by play rate within a given tour/era.
Yes, you can identify them programmatically. Percentile thresholds on an opportunity-adjusted play rate (with some shrinkage) work well. You can also use “surprise” (−log probability) for show-by-show identification.
Looking at the frequency distribution is useful for characterization and cross-band comparisons. Song play counts are typically heavy-tailed (often close to Zipf/power-law or lognormal). Fitting the tail and comparing tail exponents, Gini, or “top-K coverage” across bands is informative, though you don’t need full distribution fitting to tag deep cuts.
Why “opportunity-adjusted” matters

Raw counts bias toward older songs. A song released last year can’t have many lifetime plays yet.
Define opportunities as the number of shows since a song first appeared (or since album release).
Use a rate r = plays / opportunities instead of raw counts.
Apply shrinkage so rates for very rare/new songs don’t look extreme just due to noise.
A practical, programmatic definition

Inputs: your scraped setlist data with one row per song per show and the show dates.
For each song i:
n_i = number of unique shows where the song was played.
o_i = number of shows since its first appearance (opportunities).
Posterior play rate (empirical Bayes): r̂_i = (n_i + α) / (o_i + α + β), with α, β chosen by empirical Bayes or simple defaults (e.g., α=1, β=19 implies a soft prior around 5%).
Lifetime deep cut flag:
Minimum opportunities filter: o_i ≥ 20 (or a threshold that suits your data volume).
Deep cut if r̂_i is below the τ-th percentile (e.g., τ=25) of r̂ among songs passing the filter, or if P(rate < 0.05) > 0.9 under the Beta posterior.
Tour-specific deep cuts:
Recompute n_i and o_i within a tour and repeat the same logic. This captures “deep on this tour” even if the song was common in earlier eras.
Show-level surprise:
If your predictive model yields per-song probabilities p_i for a given show, define surprisal S_i = −log p_i. Tag a song as a deep cut in that show if S_i exceeds a threshold (e.g., top 20% most surprising within that tour or above a fixed S threshold). This is useful for evaluating “wow factor” per set.
A simple pandas recipe
Assume a DataFrame plays with columns: show_id, date (datetime), song_title. One row per song-play per show.

Build a show index (opportunities timeline):
Sort all shows by date and assign an index t.
For each song, find its first index t0.
opportunities o_i = total_shows − t0 + 1.
Compute plays, opportunities, and shrunk rates, then flag deep cuts.
Example code (concise and customizable):

import pandas as pd
import numpy as np

plays: columns ['show_id','date','song_title']
Ensure dates are datetime
plays = plays.copy()
plays['date'] = pd.to_datetime(plays['date'])

One row per song per show (dedupe in case of duplicates/medleys)
plays = plays.drop_duplicates(subset=['show_id','song_title'])

Build show timeline
shows = plays[['show_id','date']].drop_duplicates().sort_values('date')
shows['t'] = np.arange(1, len(shows)+1)
total_shows = len(shows)

plays = plays.merge(shows[['show_id','t']], on='show_id', how='left')

First appearance index per song (t0), plays per song (n), opportunities (o)
first_t = plays.groupby('song_title')['t'].min().rename('t0')
n = plays.groupby('song_title')['show_id'].nunique().rename('n')
song_stats = pd.concat([first_t, n], axis=1).reset_index()
song_stats['o'] = total_shows - song_stats['t0'] + 1

Empirical Bayes shrinkage for rate r = n/o with Beta(α,β) prior
Simple symmetric prior centered near global mean; tune α,β as needed
global_mean = (song_stats['n'].sum() / song_stats['o'].sum())

Choose pseudo-counts so prior weight ~ 20 opportunities
prior_weight = 20
alpha = global_mean * prior_weight
beta = (1 - global_mean) * prior_weight

song_stats['r_hat'] = (song_stats['n'] + alpha) / (song_stats['o'] + alpha + beta)

Apply opportunity filter
min_o = 20
eligible = song_stats[song_stats['o'] >= min_o].copy()

Percentile threshold for deep cuts (e.g., bottom 25%)
tau = 0.25
threshold = eligible['r_hat'].quantile(tau)
eligible['deep_cut'] = eligible['r_hat'] <= threshold

Merge flags back if you want a column for all songs (False for ineligible by default)
song_stats = song_stats.merge(eligible[['song_title','deep_cut']], on='song_title', how='left')
song_stats['deep_cut'] = song_stats['deep_cut'].fillna(False)

Optional: compute a per-song “surprise weight” for current tour modeling: surprisal = -log(r_hat)
song_stats['surprisal'] = -np.log(song_stats['r_hat'].clip(lower=1e-6))

Notes on choices and pitfalls

New material: Exclude songs with very low opportunities or treat “current album cycle” separately so you don’t call the whole new record “deep cuts.”
Covers and medleys: Decide whether to include, exclude, or group them. Covers are often rare by design.
Title variants: Normalize aliases and punctuation to avoid splitting the same song.
Missing shows: If many setlists are missing, your “opportunities” (denominator) might be overstated. You can instead use “observed opportunities” = number of shows with recorded setlists since first appearance.
Time decay: If you care about current deep cuts, use time-decayed counts/opportunities (e.g., weight by exp(−Δt/half_life)) so old plays don’t dominate.
Tour- or venue-conditional deep: Compute r̂ within strata (tour, venue, city) for more nuanced definitions.
Distribution fitting and cross-band comparisons

Usefulness: Good for understanding repertoire concentration and comparing artists/eras, less necessary for tagging deep cuts day-to-day.
What to measure:
Tail fit: power law vs lognormal on play counts (Clauset–Shalizi–Newman method). The tail exponent (alpha) indicates how hit-dominated the catalog is.
Concentration: Gini coefficient, Herfindahl–Hirschman Index, or “top-K coverage” (e.g., fraction of total plays covered by the top 10 songs).
Repertoire breadth: N50-style metric (minimum number of songs that account for 50% of plays).
Interpreting: Lower alpha, higher Gini, and higher top-K coverage imply a more hit-focused setlist strategy; heavier tails often coincide with a larger deep-cut reservoir.
An alternative, model-based definition

Fit a mixture model on log play rates (or time-decayed rates) across songs (e.g., 2–3 Gaussian components). The lowest-mean component typically corresponds to deep cuts. This avoids picking an arbitrary percentile and adapts to the band’s distribution.
Or define deep cut as “statistically surprising on this tour”: given your tour-level predictive model, label songs with p_i below a fixed small threshold (e.g., p_i < 2%) or in the bottom decile of predicted probabilities.
Bottom line

A practical, defensible approach is:
Use opportunity-adjusted, shrunk play rates.
Impose a minimum opportunity threshold.
Define deep cuts via a percentile or a mixture component.
For per-show analysis, use surprisal based on your current predictive baseline.
Distribution fitting won’t improve the tagging directly, but it’s valuable for describing and comparing catalog usage across artists and eras.