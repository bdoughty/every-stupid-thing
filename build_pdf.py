"""Render report.md's content as report.pdf via reportlab (no pandoc/LaTeX
needed). Not a generic markdown-to-PDF converter -- mirrors report.md by
hand so headings/images/tables get real PDF layout.

Usage:
    /Users/bdoughty/opt/miniconda3/envs/tmg-scrape/bin/python build_pdf.py
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parent
PLOTS = ROOT / "analysis" / "plots"

INK = colors.HexColor("#0b0b0b")
SECONDARY = colors.HexColor("#52514e")
MUTED = colors.HexColor("#898781")
BLUE = colors.HexColor("#2a78d6")
GRID = colors.HexColor("#e1e0d9")

styles = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=19, leading=23,
                             textColor=INK, spaceAfter=4),
    "subtitle": ParagraphStyle("subtitle", fontName="Helvetica-Oblique", fontSize=10.5,
                                leading=14, textColor=SECONDARY, spaceAfter=16),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=14.5, leading=18,
                          textColor=INK, spaceBefore=18, spaceAfter=8),
    "h3": ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=11.5, leading=15,
                          textColor=INK, spaceBefore=10, spaceAfter=5),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.8, leading=14.5,
                            textColor=INK, spaceAfter=8),
    "caption": ParagraphStyle("caption", fontName="Helvetica-Oblique", fontSize=8.3,
                               leading=11.5, textColor=MUTED, spaceAfter=14),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=8.3, leading=11.5,
                            textColor=INK, backColor=colors.HexColor("#f4f3ef"),
                            borderPadding=8, spaceAfter=10),
}

B = lambda t: f"<b>{t}</b>"
I = lambda t: f"<i>{t}</i>"


def img(name, width=6.6 * inch):
    path = PLOTS / name
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    return Image(str(path), width=width, height=width * h / w)


def metrics_table():
    data = [
        ["model", "log-loss", "Brier score", "top-n setlist recovery"],
        ["Baseline (decayed play rate)", "0.0822", "0.0195", "51.1%"],
        ["Logistic regression", "0.0673", "0.0166", "59.4%"],
    ]
    t = Table(data, colWidths=[2.3 * inch, 1.2 * inch, 1.2 * inch, 1.7 * inch])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#c3c2b7")),
        ("LINEBELOW", (0, 1), (-1, 1), 0.4, GRID),
        ("FONTNAME", (1, 2), (2, 2), "Helvetica-Bold"),
        ("FONTNAME", (3, 2), (3, 2), "Helvetica-Bold"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def bullets(items):
    return ListFlowable(
        [ListItem(Paragraph(t, styles["body"]), bulletColor=BLUE) for t in items],
        bulletType="bullet", start="circle", leftIndent=14, bulletFontSize=6,
    )


def build():
    doc = SimpleDocTemplate(
        str(ROOT / "report.pdf"), pagesize=LETTER,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=0.8 * inch, rightMargin=0.8 * inch,
        title="The Mountain Goats, Every Night", author="tmg-live-shows project",
    )
    S = []
    P = lambda t, s="body": S.append(Paragraph(t, styles[s]))

    P("The Mountain Goats, Every Night", "title")
    P("Scraping and modeling 34 years of live setlists from "
      "themountaingoats.fandom.com", "subtitle")

    P("The dataset", "h2")
    P("The wiki lists 1,579 live shows from 1992-05-31 through 2026-05-26 &mdash; solo "
      "John Darnielle sets, full-band tours, radio sessions, festival slots, and Extra "
      "Glenns/Extra Lens side-project shows. 1,317 of those have a documented setlist: "
      "22,699 individual song performances across 845 distinct songs, spanning 97 named "
      "tours. 1,739 performances have at least one linked YouTube/Vimeo video.")
    P("Getting from a wiki category page to those numbers took a few real fixes:")
    S.append(bullets([
        f"{B('Tour isn&rsquo;t in the page text &mdash; it&rsquo;s a wiki category')} "
        "(e.g. <i>Peter Hughes Farewell Tour 2024</i>). Two earlier scraper attempts both "
        "missed this and scraped a null tour for every show.",
        f"{B('Encores aren&rsquo;t marked in the setlist table.')} They&rsquo;re stated in "
        'prose in the Notes section &mdash; &ldquo;The encore was songs 19 through 23&rdquo; '
        "&mdash; referencing the table&rsquo;s printed order numbers. A small parser turns "
        "~30 recurring phrasings of that sentence into a per-song encore number.",
        f"{B('Song identity needed real normalization.')} The wiki links each song to its "
        "own page, but the same song shows up under underscore-joined slugs, space-joined "
        "display text, and case variants of both. Left unmerged, this silently splits a "
        "song&rsquo;s play count across rows &mdash; a real bug caught mid-project, where "
        '&ldquo;Southwood Plantation Road&rdquo; initially showed 1 play instead of its '
        "real count, 165.",
        f"{B('Covers needed their own source of truth.')} Sniffing setlist notes for the "
        "word &ldquo;cover&rdquo; only catches about a dozen of them &mdash; the wiki "
        "actually maintains its own Category:Covers, which tags 156 of the songs played "
        "live here. Every song carries an is_cover flag from that category, and the "
        "prediction model excludes covers from its candidate universe entirely.",
    ]))
    P("The pipeline is two-stage: <b>fetch</b> downloads and locally caches every wiki page "
      "via the MediaWiki API (resumable, incremental on later runs), and <b>build</b> parses "
      "the cached HTML into tidy tables entirely offline.")

    P("What gets played, and what doesn't", "h2")
    S.append(img("song_frequency.png"))
    P('&ldquo;This Year&rdquo; and &ldquo;No Children&rdquo; anchor the top of the list &mdash; '
      "both played at a majority of all shows since entering rotation. The deep-cuts panel "
      "isn't just least-played-ever (most of the 845 songs have only 1&ndash;2 plays and "
      "aren't interesting on their own); it's the bottom of an opportunity-adjusted, "
      "shrinkage-smoothed play rate among songs with at least 20 chances to be played since "
      "their live debut &mdash; so a song only counts as a deep cut if it's been "
      "consistently passed over, not just recently written.", "caption")

    P("Song popularity over time", "h2")
    P("Treating each song's live history like a word's frequency in Google Ngrams &mdash; a "
      "trailing 2-year play rate sampled monthly &mdash; turns 34 years of setlists into "
      "trajectories: songs sit at zero before they're written, rise as they enter rotation, "
      "plateau, decline, and sometimes come back.")
    S.append(img("song_trajectories.png"))
    P("These four groups were picked automatically from the computed trajectories, not "
      "hand-selected: the most-played songs overall (steady classics); songs that debuted "
      "in the back half of the catalog's history and have climbed straight into rotation "
      "since (rise and plateau &mdash; a song appearing from nothing and taking hold); the "
      "biggest gap between early-career peak and recent play rate (decline); and songs with "
      "a real trough between two active periods (fall and revival).", "caption")
    P('The decline panel caught something the wiki itself corroborates: &ldquo;Going to '
      'Georgia&rdquo; was a fixture through the mid-2000s, then drops off hard &mdash; a 2012 '
      "show note has John Darnielle explaining, mid-show, that he considers it a "
      '&ldquo;bullshit song&rdquo; with an &ldquo;asshole&rdquo; narrator and doesn&rsquo;t '
      "want to play it again. The data shows the falloff independent of that anecdote; the "
      "anecdote just explains it.")

    P("Predicting the setlist", "h2")
    P("Framed as one binary outcome per (show, song) pair &mdash; is this the night song X "
      "gets played? &mdash; every one of the 1,310 dated, setlisted shows becomes a training "
      "example, using only information available before that show: how recently and how "
      "often the song's been played, how it's fared on the current tour, its age, and "
      "whether it's a shortened-set special appearance (radio/festival/TV).")
    P("A logistic regression on those features is compared against a simple but strong "
      "baseline &mdash; an exponentially-decayed play rate alone &mdash; on a strict "
      "temporal split (train through 2022, test on the 207 shows from 2023 onward):")
    S.append(Spacer(1, 4))
    S.append(metrics_table())
    S.append(Spacer(1, 10))
    P("<i>Top-n setlist recovery</i>: for each show, take the model's n highest-probability "
      "songs, where n is the actual number played that night, and measure what fraction "
      "were right. On the most recent show in the data:")
    S.append(img("example_prediction.png", width=5.6 * inch))

    P("What actually predicts a setlist", "h3")
    S.append(img("model_weights.png"))
    P("Recency dominates everything else combined &mdash; if a song was played a few shows "
      "ago, it's overwhelmingly likely to come back soon. That's the visible mechanism "
      "behind the tour &ldquo;rotation&rdquo;: John Darnielle appears to work from a live "
      "pool of songs and cycle through it, which is exactly what tour_rate (play rate so "
      "far, this tour) picking up an independent signal on top of recency confirms. Two "
      "smaller, genuinely interesting effects: conditional on recency, brand-new material "
      "is actually less sticky than catalog average, and radio/festival/TV appearances "
      "reliably favor hits over deep cuts.", "caption")

    P("Is this just tracking album hype?", "h3")
    P("The 2026 example above is a new-album tour, and new_material is a real feature in the "
      "model &mdash; worth checking whether the strong recovery number is mostly &ldquo;predict "
      "whatever's newest&rdquo; in disguise. Among all 2014 tours with at least 8 shows, the "
      "one whose actually-played songs had the lowest share of new material &mdash; picked "
      "automatically, not by hand &mdash; is the Twin Inhuman Highway Fiends Tour 2014, "
      "squarely between Transcendental Youth (2012) and Beat the Champ (2015):")
    S.append(img("example_comparison.png"))
    P("The between-albums tour is visibly harder to predict &mdash; 55.1% top-n recovery "
      "tour-wide, versus 59.4% for the full 2023+ test era &mdash; despite 2014 being "
      "in-sample (which should make it look artificially easier, not harder). Without a "
      "record to promote, the setlist draws more evenly across a wider pool of similarly-loved "
      "catalog songs, so there's genuinely more entropy to predict. The model isn't just "
      "riding hype; if anything, album cycles are the easy case.", "caption")

    P("When the model gets surprised", "h3")
    P("The flip side of a good model is a good list of its misses &mdash; nights the model "
      "gave a song almost no chance, and it got played anyway:")
    S.append(img("surprising_plays.png", width=6.2 * inch))
    P("These are basically a machine-generated &ldquo;rarities and one-offs&rdquo; list: "
      "songs that came back from years of dormancy for a single night, often at a show with "
      "some specific occasion.", "caption")

    P("How surprising was each setlist as a whole?", "h3")
    P("Zooming out from individual songs: the same pre-show probabilities give a surprise "
      "score for an entire night &mdash; the average bits of information across the songs "
      "actually played. Aggregated per tour (touring is bursty, so a fixed show-count window "
      "saws at tour boundaries):")
    S.append(img("surprisal_over_time.png"))
    P("The tours with the highest average surprise are almost all solo, stripped-down tours "
      "&mdash; Winter Solo Tour 2024, the Ghost Cave Incubation Chamber solo non-tour, All "
      "Roads Lead to Lincoln Solo Mini-Tour &mdash; which makes sense: a solo acoustic set "
      "draws from a different pool than a full-band show, and the model has no explicit "
      "&ldquo;solo show&rdquo; feature to account for it. A concrete lead for the next "
      "modeling iteration.", "caption")

    P("Caveats", "h2")
    S.append(bullets([
        "The wiki is fan-maintained and lists most, not all, shows &mdash; coverage is "
        "noticeably better from the mid-2000s on.",
        "The model has no access to anything outside setlist history &mdash; no lyrics, no "
        "album themes, no explicit &ldquo;songs currently being toured.&rdquo; tour_rate "
        "and song_age are proxies for that.",
        "Song identity is deduplicated via wiki link slugs where available; a handful of "
        "never-linked rarities may still have unmerged spelling variants.",
        "The model has no &ldquo;solo show&rdquo; feature, which the surprise analysis "
        "suggests it should &mdash; solo/stripped-down tours are consistently hardest to "
        "predict.",
    ]))

    P("Reproducing this", "h2")
    S.append(Paragraph(
        "PY=/Users/bdoughty/opt/miniconda3/envs/tmg-scrape/bin/python<br/>"
        "$PY scrape.py fetch &amp;&amp; $PY scrape.py build &nbsp;# refresh the raw data<br/>"
        "$PY analyze.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# deep cuts, tour summaries<br/>"
        "$PY predict.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# the setlist model<br/>"
        "$PY timeseries.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# popularity-over-time series<br/>"
        "$PY plot_report.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# this report's figures",
        styles["code"],
    ))
    P("Full data dictionary in README.md; wiki-scraping gotchas and repo layout in CLAUDE.md.",
      "caption")

    doc.build(S)
    print(f"Wrote {ROOT / 'report.pdf'}")


if __name__ == "__main__":
    build()
