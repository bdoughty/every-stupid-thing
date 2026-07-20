"""Scrape The Mountain Goats fandom wiki live shows into tidy tables.

Two-stage design so parsing tweaks never require re-downloading:

  1. fetch  -- enumerate Category:Live_Shows via the MediaWiki API and cache
               each page's rendered HTML + categories + revid as JSON in
               data/raw/<pageid>.json. Resumable: already-cached pages are
               skipped unless their revid changed (or --force).
  2. build  -- parse every cached JSON into data/shows.csv,
               data/performances.csv, data/songs.csv.

Usage:
    python scrape.py fetch [--limit N] [--delay 0.4] [--force]
    python scrape.py build

Run with the tmg-scrape conda env:
    /Users/bdoughty/opt/miniconda3/envs/tmg-scrape/bin/python scrape.py fetch
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote, unquote

import requests
from bs4 import BeautifulSoup, Tag

BASE_URL = "https://themountaingoats.fandom.com"
API_URL = f"{BASE_URL}/api.php"
CATEGORY = "Category:Live_Shows"
COVERS_CATEGORY = "Category:Covers"
USER_AGENT = "TMG-Live-Show-Scraper/2.0 (+benjamin.doughty@gmail.com)"

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data"

# Categories that are flags rather than tours, lowercased.
FLAG_CATEGORIES = {
    "shows_with_video": "has_video",
    "shows_with_audio": "has_audio",
    "incomplete_setlists": "incomplete_setlist",
    "shows_with_incomplete_setlists": "incomplete_setlist",
    "incomplete_articles": "incomplete_article",
    "video": "has_video",
}

# One-off region category variants, lowercased -> region name.
REGION_ALIASES = {"north_carolina_shows": "North Carolina"}

# Categories describing what kind of show it was, lowercased.
SHOW_TYPE_CATEGORIES = {
    "radio_sessions": "radio session",
    "festival_shows": "festival",
    "television_appearances": "tv appearance",
}

# Categories naming the act rather than a tour, lowercased.
ACT_CATEGORIES = {"the_extra_glenns", "the_extra_lens"}

# Meta categories to ignore entirely, lowercased.
IGNORE_CATEGORIES = {"live_shows", "live_shows_by_country", "live_shows_by_state"}


# ---------------------------------------------------------------- fetch stage

def make_session():
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def api_get(session, params, delay):
    params = dict(params, format="json", formatversion=2)
    resp = session.get(API_URL, params=params, timeout=30)
    time.sleep(delay)
    resp.raise_for_status()
    return resp.json()


def list_category_members(session, delay, category=CATEGORY):
    members, cont = [], None
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmnamespace": 0,
            "cmlimit": "max",
        }
        if cont:
            params["cmcontinue"] = cont
        j = api_get(session, params, delay)
        members.extend(j["query"]["categorymembers"])
        cont = j.get("continue", {}).get("cmcontinue")
        if not cont:
            break
    return members


def fetch(limit=None, delay=0.4, force=False):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()

    print("Listing category members...", flush=True)
    members = list_category_members(session, delay)
    print(f"{len(members)} pages in {CATEGORY}", flush=True)
    (RAW_DIR / "_members.json").write_text(json.dumps(members, indent=1))

    # Song pages tagged as covers on the wiki -- titles are the song's own
    # page title, which matches song_key's space-joined form directly.
    covers = list_category_members(session, delay, category=COVERS_CATEGORY)
    print(f"{len(covers)} pages in {COVERS_CATEGORY}", flush=True)
    (RAW_DIR / "_covers.json").write_text(json.dumps(covers, indent=1))

    # Grab current revids in bulk (50 titles/request) so unchanged cached
    # pages can be skipped without a per-page parse call.
    revids = {}
    if not force:
        for i in range(0, len(members), 50):
            chunk = members[i : i + 50]
            j = api_get(
                session,
                {
                    "action": "query",
                    "pageids": "|".join(str(m["pageid"]) for m in chunk),
                    "prop": "revisions",
                    "rvprop": "ids",
                },
                delay,
            )
            for p in j["query"]["pages"]:
                revs = p.get("revisions") or [{}]
                revids[p["pageid"]] = revs[0].get("revid")

    if limit:
        members = members[:limit]

    fetched = skipped = failed = 0
    for i, m in enumerate(members):
        pageid, title = m["pageid"], m["title"]
        out = RAW_DIR / f"{pageid}.json"
        if out.exists() and not force:
            cached = json.loads(out.read_text())
            if cached.get("revid") == revids.get(pageid):
                skipped += 1
                continue
        try:
            j = api_get(
                session,
                {"action": "parse", "pageid": pageid, "prop": "text|revid|categories"},
                delay,
            )
            p = j["parse"]
            out.write_text(
                json.dumps(
                    {
                        "pageid": pageid,
                        "title": title,
                        "revid": p.get("revid"),
                        "categories": [c["category"] for c in p.get("categories", [])],
                        "html": p.get("text", ""),
                    }
                )
            )
            fetched += 1
        except Exception as e:  # noqa: BLE001 - log and continue the crawl
            print(f"  FAILED {title}: {e}", flush=True)
            failed += 1
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(members)} (fetched {fetched}, cached {skipped}, failed {failed})", flush=True)

    print(f"Done: fetched {fetched}, already cached {skipped}, failed {failed}", flush=True)


# ---------------------------------------------------------------- build stage

def parse_title(title):
    """'2014-04-16 - Somerville Theatre - Somerville, MA' -> fields.

    Dates may be partial: '1992-xx-xx', '1994-08-xx'. Venue may be absent
    ('1994-08-xx - New Paltz, NY') or contain ' - ' itself.
    """
    parts = [p.strip() for p in title.split(" - ")]
    date_raw = parts[0]
    venue = location = None
    if len(parts) == 2:
        location = parts[1]
    elif len(parts) >= 3:
        venue = " - ".join(parts[1:-1])
        location = parts[-1]

    m = re.match(r"^(\d{4})-(\d{2}|xx)-(\d{2}|xx)", date_raw)
    year = month = day = None
    date = None
    if m:
        year = int(m.group(1))
        if m.group(2) != "xx":
            month = int(m.group(2))
        if m.group(3) != "xx":
            day = int(m.group(3))
        if month and day:
            date = f"{year:04d}-{month:02d}-{day:02d}"

    city = region = None
    if location and "," in location:
        city, region = [p.strip() for p in location.rsplit(",", 1)]
    elif location:
        city = location
    return {
        "date_raw": date_raw,
        "date": date,
        "year": year,
        "month": month,
        "day": day,
        "venue": venue,
        "location": location,
        "city": city,
        "region": region,
    }


def classify_categories(categories):
    tours, regions, show_types, acts = [], [], [], []
    flags = set()
    for c in categories:
        low = c.lower()
        if low in IGNORE_CATEGORIES or re.fullmatch(r"\d{4}", low):
            continue  # meta categories / bare years duplicating the date
        if low in FLAG_CATEGORIES:
            flags.add(FLAG_CATEGORIES[low])
        elif low in SHOW_TYPE_CATEGORIES:
            show_types.append(SHOW_TYPE_CATEGORIES[low])
        elif low in ACT_CATEGORIES:
            acts.append(c.replace("_", " "))
        elif low in REGION_ALIASES:
            regions.append(REGION_ALIASES[low])
        elif low.endswith("_live_shows"):
            regions.append(c[: -len("_live_shows")].replace("_", " "))
        else:
            tours.append(c.replace("_", " "))
    return tours, regions, flags, show_types, acts


VIDEO_HOST_RE = re.compile(r"https?://(?:www\.|m\.)?(?:youtube\.com|youtu\.be|vimeo\.com)/", re.IGNORECASE)


def clean_song_cell(cell):
    """Extract song title/slug/note/videos from a setlist-table song cell."""
    cell = BeautifulSoup(str(cell), "html.parser")  # private copy to mutate
    # External links are video/audio references, not part of the song name.
    videos = []
    for a in cell.select("a.external"):
        href = a.get("href", "")
        if VIDEO_HOST_RE.match(href):
            videos.append(href)
        a.decompose()
    for sup in cell.select("sup"):
        sup.decompose()

    slug = None
    link_title = None
    link_text = None
    for a in cell.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/wiki/") and ":" not in unquote(href[len("/wiki/"):]):
            slug = unquote(href[len("/wiki/"):]).split("#")[0]
            link_text = a.get_text(" ", strip=True)
            link_title = a.get("title") or link_text
            break

    raw = re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).strip()
    # Prefer the wiki-link title (canonical spelling), else quoted text, else
    # the cleaned raw text. But if quoted text and link disagree entirely, the
    # link is something else (a performer, an album) — trust the quotes.
    qm = re.search(r"[\"“]([^\"”]+)[\"”]", raw)
    quoted = qm.group(1).strip() if qm else None
    if quoted and link_title:
        a, b = quoted.lower(), link_title.lower()
        if a not in b and b not in a:
            link_title = slug = None
    title = link_title or quoted or raw.strip('"').strip()

    # Whatever remains outside the quoted/linked title (guests, "cover", ...).
    # Strip what actually appears in `raw` (the quoted text, or else the
    # link's own visible text) -- NOT `title`, which may have been swapped
    # for the wiki link's `title=` attribute on a disambiguated page (link
    # text "Get Lonely", title "Get Lonely (Song)" because a same-named
    # album page exists). Stripping `title` there wouldn't match anything
    # in `raw`, leaving the original link text stranded as a bogus note.
    note = raw
    strip_text = quoted or link_text or title
    if strip_text:
        note = note.replace(strip_text, "")
    note = re.sub(r"[\"“”]", "", note)
    note = re.sub(r"\(\s*\)", "", note)
    note = re.sub(r"\s+", " ", note).strip(" ,;-").strip()
    if not re.search(r"\w", note):
        note = None
    return title, slug, note, raw, videos


def parse_album_cell(cell):
    a = cell.find("a", href=True)
    if a and a["href"].startswith("/wiki/"):
        return a.get("title") or a.get_text(strip=True)
    text = cell.get_text(" ", strip=True)
    return text or None


ENCORE_RE = re.compile(r"\bencore\b", re.IGNORECASE)

# "This was a John Darnielle solo show." -- distinct from "John's solo SET
# was songs 7 through 9," which describes a segment of an otherwise
# full-band show, not the whole night.
SOLO_NOTE_RE = re.compile(r"\bsolo show\b", re.IGNORECASE)

ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4}

# "The encore was songs 19 through 23." / "songs 18-21" / "song 22 and 23" /
# "The first encore was ... and the second encore was ..."
ENCORE_NOTE_RE = re.compile(
    r"(first|second|third|fourth)?\s*encore\s+(?:was|includes|were)\s+songs?\s+"
    r"(\d+)(?:\s*(through|to|and|[-–—])\s*(\d+))?",
    re.IGNORECASE,
)


def parse_notes(soup):
    """Bullet lines under the Notes heading."""
    notes = []
    for h in soup.find_all("h2"):
        if "notes" in h.get_text(" ", strip=True).lower():
            for sib in h.next_siblings:
                if isinstance(sib, Tag):
                    if sib.name == "h2":
                        break
                    if sib.name in ("ul", "ol"):
                        notes.extend(
                            li.get_text(" ", strip=True) for li in sib.find_all("li")
                        )
            break
    return [n for n in notes if n]


def encore_map(notes):
    """Map setlist position -> encore number, from Notes prose."""
    mapping = {}
    for note in notes:
        for m in ENCORE_NOTE_RE.finditer(note):
            ordinal, start, conn, end = m.groups()
            n = ORDINALS.get((ordinal or "").lower(), 1)
            start = int(start)
            if end is None:
                mapping[start] = n
            elif conn and conn.lower() == "and" and int(end) > start + 1:
                mapping[start] = n  # "songs 18 and 21" lists exactly two
                mapping[int(end)] = n
            else:
                for pos in range(start, int(end) + 1):
                    mapping[pos] = n
    return mapping


def parse_setlist(soup):
    """Return list of performance dicts from a show page's parsed HTML."""
    content = soup.select_one("div.mw-parser-output") or soup

    heading = None
    for h in content.find_all(["h2", "h3"]):
        if "setlist" in h.get_text(" ", strip=True).lower():
            heading = h
            break

    songs = []
    seq = 0
    encore = 0

    def eat_table(table):
        nonlocal seq, encore
        rows = table.find_all("tr")
        # Column layout comes from the header row (cells with bgcolor);
        # most tables are Order/Song/Album but some omit the Order column.
        order_col, song_col, album_col = 0, 1, 2
        for row in rows:
            texts = [c.get_text(" ", strip=True).lower() for c in row.find_all("td")]
            if texts:
                if "song" in texts:
                    song_col = texts.index("song")
                    order_col = texts.index("order") if "order" in texts else None
                    album_col = texts.index("album") if "album" in texts else None
                break
        for row in rows:
            cells = row.find_all("td")
            if not cells or len(cells) <= song_col:
                continue
            first = cells[0].get_text(" ", strip=True)
            if first.lower() in ("order", "#", "no.", "song") or cells[0].get("bgcolor"):
                continue
            if ENCORE_RE.search(first) and not first[:1].isdigit():
                encore += 1
                continue
            order_raw = None
            if order_col is not None:
                m = re.match(r"(\d+)", cells[order_col].get_text(" ", strip=True))
                order_raw = int(m.group(1)) if m else None
            title, slug, note, raw, videos = clean_song_cell(cells[song_col])
            if not title or not re.search(r"\w", title):
                continue
            seq += 1
            songs.append(
                {
                    "seq": seq,
                    "order_raw": order_raw,
                    "encore": encore,
                    "song_title": title,
                    "song_slug": slug,
                    "album": (
                        parse_album_cell(cells[album_col])
                        if album_col is not None and len(cells) > album_col
                        else None
                    ),
                    "note": note,
                    "video_urls": "|".join(videos) or None,
                    "raw_text": raw,
                }
            )

    def eat_list(lst):
        nonlocal seq, encore
        for li in lst.find_all("li", recursive=False):
            text = li.get_text(" ", strip=True)
            if not text:
                continue
            if ENCORE_RE.match(text):
                encore += 1
                continue
            title, slug, note, raw, videos = clean_song_cell(li)
            title = re.sub(r"^\s*\d+\s*[\.\)]\s*", "", title).strip()
            if not title or not re.search(r"\w", title):
                continue
            seq += 1
            songs.append(
                {
                    "seq": seq,
                    "order_raw": None,
                    "encore": encore,
                    "song_title": title,
                    "song_slug": slug,
                    "album": None,
                    "note": note,
                    "video_urls": "|".join(videos) or None,
                    "raw_text": raw,
                }
            )

    if heading is not None:
        for sib in heading.next_siblings:
            if not isinstance(sib, Tag):
                continue
            if sib.name == "h2":
                break
            if sib.name in ("h3", "h4"):
                if ENCORE_RE.search(sib.get_text(" ", strip=True)):
                    encore += 1
                continue
            for table in ([sib] if sib.name == "table" else sib.find_all("table")):
                eat_table(table)
            if sib.name in ("ol", "ul"):
                eat_list(sib)
            else:
                for lst in sib.find_all(["ol", "ul"], recursive=False):
                    eat_list(lst)

    # Fallback for pages without a Setlist heading: first wikitable that
    # looks like a setlist (Order/Song header).
    if not songs:
        for table in content.find_all("table"):
            head = table.get_text(" ", strip=True).lower()
            if "order" in head and "song" in head:
                eat_table(table)
                if songs:
                    break

    return songs


def build():
    import pandas as pd

    raw_files = sorted(p for p in RAW_DIR.glob("*.json") if not p.name.startswith("_"))
    if not raw_files:
        sys.exit("No cached pages in data/raw/ - run `scrape.py fetch` first.")
    print(f"Parsing {len(raw_files)} cached pages...", flush=True)

    shows, perfs, note_rows = [], [], []
    unclassified_tours = {}
    for path in raw_files:
        page = json.loads(path.read_text())
        title = page["title"]
        slug = title.replace(" ", "_")
        meta = parse_title(title)
        tours, regions, flags, show_types, acts = classify_categories(page["categories"])
        soup = BeautifulSoup(page["html"], "html.parser")
        setlist = parse_setlist(soup)
        notes = parse_notes(soup)
        note_rows.extend(
            {"show_id": slug, "note_seq": i + 1, "note": n} for i, n in enumerate(notes)
        )
        # Whole-show solo flag: "solo show" in notes (distinct from a
        # partial "solo set" within an otherwise full-band show), or a tour
        # explicitly branded as solo. Conservative/high-precision by
        # design -- likely under-flags true solo shows the wiki didn't
        # bother to call out (especially pre-2002, before a full-time
        # backing band was the norm), so treat this as "known-solo," not
        # an exhaustive classifier.
        is_solo = any(SOLO_NOTE_RE.search(n) for n in notes) or any(
            "solo" in t.lower() for t in tours
        )

        # Encores live in Notes prose ("The encore was songs 19 through 23"),
        # keyed to the wiki's printed order numbers.
        enc = encore_map(notes)
        if enc:
            for s in setlist:
                pos = s["order_raw"] if s["order_raw"] is not None else s["seq"]
                s["encore"] = enc.get(pos, 0)
        for t in tours:
            unclassified_tours[t] = unclassified_tours.get(t, 0) + 1

        show = {
            "show_id": slug,
            "pageid": page["pageid"],
            "title": title,
            "url": f"{BASE_URL}/wiki/{quote(slug)}",
            **meta,
            "tour": tours[0] if tours else None,
            "all_tours": "|".join(tours) or None,
            "show_type": "|".join(show_types) or None,
            "act": "|".join(acts) or "The Mountain Goats",
            "region_category": "|".join(regions) or None,
            "has_video": "has_video" in flags,
            "has_audio": "has_audio" in flags,
            "incomplete_setlist": "incomplete_setlist" in flags,
            "incomplete_article": "incomplete_article" in flags,
            "is_solo": is_solo,
            "n_songs": len(setlist),
            "revid": page["revid"],
        }
        shows.append(show)
        for s in setlist:
            perfs.append({"show_id": slug, "date": meta["date"], "year": meta["year"], **s})

    shows_df = pd.DataFrame(shows).sort_values(["date_raw", "show_id"]).reset_index(drop=True)
    perfs_df = pd.DataFrame(perfs)

    # Canonical song key: the wiki link slug when present (normalizes typos
    # and display variants), else the display title. Slugs use underscores
    # where display text uses spaces — normalize so linked and unlinked
    # plays of the same song unify.
    slug_spaced = perfs_df["song_slug"].str.replace("_", " ", regex=False).str.strip()
    perfs_df["song_key"] = slug_spaced.fillna(perfs_df["song_title"].str.strip())
    # Case-variant wiki redirects ("Broken To/to Begin With") — unify each
    # case-insensitive group under its most common casing.
    key_lower = perfs_df["song_key"].str.lower()
    majority_case = perfs_df.groupby(key_lower)["song_key"].agg(lambda s: s.mode().iat[0])
    perfs_df["song_key"] = key_lower.map(majority_case)
    canonical = (
        perfs_df.dropna(subset=["song_slug"])
        .groupby("song_key")["song_title"]
        .agg(lambda s: s.mode().iat[0])
    )
    perfs_df["song_canonical"] = perfs_df["song_key"].map(canonical).fillna(perfs_df["song_title"])

    # Cover flag, from the wiki's own Category:Covers (authoritative — far
    # more complete than sniffing "(cover)" out of setlist notes, which only
    # catches ~12 of the ~156 wiki-tagged covers we've actually played live).
    covers_path = RAW_DIR / "_covers.json"
    if covers_path.exists():
        cover_titles = {m["title"] for m in json.loads(covers_path.read_text())}
        perfs_df["is_cover"] = perfs_df["song_key"].isin(cover_titles)
    else:
        print("warning: data/raw/_covers.json missing (run `fetch` to refresh) — is_cover left False", flush=True)
        perfs_df["is_cover"] = False

    perfs_df = perfs_df.sort_values(["date", "show_id", "seq"]).reset_index(drop=True)

    songs_df = (
        perfs_df.groupby("song_key")
        .agg(
            song_title=("song_canonical", "first"),
            is_cover=("is_cover", "any"),
            n_plays=("show_id", "size"),
            n_shows=("show_id", "nunique"),
            first_played=("date", lambda s: s.dropna().min()),
            last_played=("date", lambda s: s.dropna().max()),
            album=("album", lambda s: s.dropna().mode().iat[0] if s.dropna().size else None),
            n_videos=("video_urls", lambda s: sum(len(v.split("|")) for v in s.dropna())),
        )
        .sort_values("n_plays", ascending=False)
        .reset_index()
    )

    OUT_DIR.mkdir(exist_ok=True)
    shows_df.to_csv(OUT_DIR / "shows.csv", index=False)
    perfs_df.to_csv(OUT_DIR / "performances.csv", index=False)
    songs_df.to_csv(OUT_DIR / "songs.csv", index=False)
    pd.DataFrame(note_rows).to_csv(OUT_DIR / "show_notes.csv", index=False)

    n_with = int((shows_df["n_songs"] > 0).sum())
    print(f"shows:        {len(shows_df)} ({n_with} with setlists, {len(shows_df) - n_with} without)")
    print(f"performances: {len(perfs_df)} ({int((perfs_df.encore > 0).sum())} in encores)")
    print(f"songs:        {len(songs_df)}")
    print(f"notes:        {len(note_rows)}")
    print(f"date range:   {shows_df['date'].dropna().min()} .. {shows_df['date'].dropna().max()}")
    top_tours = shows_df["tour"].value_counts().head(5)
    print("top tours:", dict(top_tours))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download/refresh raw page cache")
    f.add_argument("--limit", type=int, default=None)
    f.add_argument("--delay", type=float, default=0.4)
    f.add_argument("--force", action="store_true", help="re-fetch even if revid unchanged")
    sub.add_parser("build", help="parse raw cache into data/*.csv")
    args = ap.parse_args()
    if args.cmd == "fetch":
        fetch(limit=args.limit, delay=args.delay, force=args.force)
    else:
        build()


if __name__ == "__main__":
    main()
