"""Splice webapp/data.json into webapp/app_template.html -> webapp/index.html.

Run export_webapp.py first to (re)generate data.json. The substitution
escapes any literal "</" inside the JSON text so a note or lyric snippet
can't accidentally close the surrounding <script> tag early ("\\/" is a
valid JSON escape for "/", so this is semantically a no-op after parsing).

Usage:
    /Users/bdoughty/opt/miniconda3/envs/tmg-scrape/bin/python build_webapp.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "webapp"


def main():
    template = (WEB / "app_template.html").read_text()
    data = (WEB / "data.json").read_text().replace("</", "<\\/")
    if "__APP_DATA__" not in template:
        raise SystemExit("template is missing the __APP_DATA__ placeholder")
    out = template.replace("__APP_DATA__", data)
    out_path = WEB / "index.html"
    out_path.write_text(out)
    print(f"Wrote {out_path} ({len(out) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
