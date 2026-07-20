"""Bundle a compact JSON payload for the static webapp mockup.

Pulls from data/*.csv and analysis/*.csv (both must be built already —
run scrape.py build, predict.py, and timeseries.py first) and writes a
single minified JSON to webapp/data.json, sized to embed directly in a
self-contained HTML page (no server, no network calls at runtime).

Usage:
    /Users/bdoughty/opt/miniconda3/envs/tmg-scrape/bin/python export_webapp.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
A = ROOT / "analysis"
D = ROOT / "data"
OUT = ROOT / "webapp"
OUT.mkdir(exist_ok=True)


def nn(x):
    """None-safe pass-through for CSV NaNs headed into JSON."""
    return None if pd.isna(x) else x


def main():
    shows = pd.read_csv(D / "shows.csv", parse_dates=["date"])
    perfs = pd.read_csv(D / "performances.csv", parse_dates=["date"])
    notes = pd.read_csv(D / "show_notes.csv")
    songs = pd.read_csv(D / "songs.csv")
    snapshot = pd.read_csv(A / "next_show_snapshot.csv")
    ts = pd.read_csv(A / "song_timeseries_compact.csv")
    surprisal = pd.read_csv(A / "show_surprisal.csv")
    surprisal_by_show = dict(zip(surprisal.show_id, surprisal.mean_surprisal_played_bits))

    # --- shows index (for the show/encore/solo-set browser) ---
    shows_out = {}
    for r in shows.itertuples():
        shows_out[r.show_id] = {
            "date": nn(r.date_raw),
            "iso": None if pd.isna(r.date) else r.date.strftime("%Y-%m-%d"),
            "venue": nn(r.venue),
            "city": nn(r.city),
            "region": nn(r.region),
            "tour": nn(r.tour),
            "type": nn(r.show_type),
            "act": r.act,
            "video": bool(r.has_video),
            "n": int(r.n_songs),
            "surprise": (
                round(surprisal_by_show[r.show_id], 2)
                if r.show_id in surprisal_by_show and not pd.isna(surprisal_by_show[r.show_id])
                else None
            ),
        }

    # --- songs index (search / autocomplete / video & trajectory lookup) ---
    # Built before setlists so setlist rows can reference songs/albums by
    # integer index instead of repeating title strings ~22.7k times.
    songs_out = [
        {
            "key": r.song_key,
            "title": r.song_title,
            "plays": int(r.n_plays),
            "shows": int(r.n_shows),
            "videos": int(r.n_videos),
            "first": nn(r.first_played),
            "last": nn(r.last_played),
            "album": nn(r.album),
            "cover": bool(r.is_cover),
        }
        for r in songs.itertuples()
    ]
    song_idx = {s["key"]: i for i, s in enumerate(songs_out)}
    albums = sorted(perfs.album.dropna().unique().tolist())
    album_idx = {a: i for i, a in enumerate(albums)}

    # --- setlists, keyed by show ---
    setlists = {}
    for show_id, g in perfs.sort_values("seq").groupby("show_id"):
        setlists[show_id] = [
            {
                "s": int(r.seq),
                "e": int(r.encore),
                "t": song_idx.get(r.song_key),
                "a": album_idx.get(r.album) if pd.notna(r.album) else None,
                "n": nn(r.note),
                "v": r.video_urls.split("|") if pd.notna(r.video_urls) else None,
            }
            for r in g.itertuples()
        ]

    notes_out = {}
    for show_id, g in notes.groupby("show_id"):
        notes_out[show_id] = g.sort_values("note_seq").note.tolist()

    # --- video index (song -> every performance with a video link) ---
    video_rows = perfs[perfs.video_urls.notna()].merge(
        shows[["show_id", "date_raw", "venue", "city", "region"]], on="show_id"
    )
    videos = {}
    for key, g in video_rows.groupby("song_key"):
        videos[key] = [
            {
                "show_id": r.show_id,
                "date": r.date_raw,
                "venue": nn(r.venue),
                "city": nn(r.city),
                "region": nn(r.region),
                "urls": r.video_urls.split("|"),
            }
            for r in g.sort_values("date_raw").itertuples()
        ]

    # --- "if the next show happens" predictor snapshot ---
    predictor = {
        "as_of": snapshot.as_of_date.iloc[0],
        "tour": snapshot.tour.iloc[0],
        "covers_excluded": int(perfs.is_cover.sum()),
        "songs": [
            {"key": r.song_key, "p": round(float(r.p_model), 4)}
            for r in snapshot.sort_values("p_model", ascending=False).itertuples()
        ],
    }

    # --- local favorites: most-played songs per city, for the predictor UI ---
    city_perfs = perfs.merge(shows[["show_id", "city", "region"]], on="show_id")
    city_perfs = city_perfs.dropna(subset=["city"])
    city_favorites = {}
    for city, g in city_perfs.groupby("city"):
        top = g.song_canonical.value_counts().head(8)
        if len(g.show_id.unique()) >= 2:  # skip one-off cities, not informative
            region = g.region.dropna().iloc[0] if g.region.notna().any() else None
            city_favorites[city] = {
                "region": region,
                "n_shows": int(g.show_id.nunique()),
                "songs": [{"key": k, "n": int(v)} for k, v in top.items()],
            }

    # --- trajectories: shared quarter grid + per-song sparse rate arrays ---
    quarters = sorted(ts.date.unique())
    qidx = {q: i for i, q in enumerate(quarters)}
    trajectories = {}
    for key, g in ts.groupby("song_key"):
        arr = [None] * len(quarters)
        for r in g.itertuples():
            arr[qidx[r.date]] = r.play_rate
        trajectories[key] = arr

    payload = {
        "meta": {
            "n_shows": len(shows),
            "n_setlisted": int((shows.n_songs > 0).sum()),
            "n_performances": len(perfs),
            "n_songs": len(songs),
            "date_min": shows.date_raw.iloc[shows.date.argmin()] if shows.date.notna().any() else None,
            "date_max": shows.date_raw.iloc[shows.date.argmax()] if shows.date.notna().any() else None,
        },
        "shows": shows_out,
        "setlists": setlists,
        "notes": notes_out,
        "videos": videos,
        "songs": songs_out,
        "albums": albums,
        "predictor": predictor,
        "city_favorites": city_favorites,
        "trajectory_quarters": quarters,
        "trajectories": trajectories,
    }

    out_path = OUT / "data.json"
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    out_path.write_text(text)
    print(f"Wrote {out_path} ({len(text) / 1e6:.2f} MB)")
    for k in ("shows", "setlists", "notes", "videos", "trajectories", "city_favorites"):
        print(f"  {k}: {len(payload[k])} entries")


if __name__ == "__main__":
    main()
