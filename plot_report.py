"""Report figures for the Mountain Goats setlist project.

Follows the house dataviz method: sequential blue for ranked magnitude,
diverging blue/red for signed coefficients, fixed categorical order for
multi-line trajectories, thin marks, recessive gridlines/spines.

Usage:
    /Users/bdoughty/opt/miniconda3/envs/tmg-scrape/bin/python plot_report.py
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
A = ROOT / "analysis"
PLOTS = A / "plots"
PLOTS.mkdir(exist_ok=True)

# --- palette (validated default, see dataviz skill references/palette.md) ---
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
RED = "#e34948"
GREEN = "#008300"
MAGENTA = "#e87ba4"
YELLOW = "#eda100"
AQUA = "#1baf7a"
ORANGE = "#eb6834"
VIOLET = "#4a3aa7"
CAT_ORDER = [BLUE, GREEN, MAGENTA, YELLOW, AQUA, ORANGE, VIOLET, RED]
MUTED_FILL = "#e1e0d9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10.5,
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def clean_axes(ax, x_grid=True, y_grid=False):
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(length=0)
    if x_grid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    if y_grid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def hbar(ax, labels, values, color, value_fmt="{:.0f}"):
    y = np.arange(len(labels))
    ax.barh(y, values, color=color, height=0.62, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5, color=INK)
    ax.invert_yaxis()
    for yi, v in zip(y, values):
        ax.text(v, yi, f"  {value_fmt.format(v)}", va="center", ha="left",
                 fontsize=8.5, color=INK_SECONDARY)


def fig_song_frequency():
    s = pd.read_csv(A / "song_stats.csv")
    top = s.nlargest(15, "n_plays").iloc[::-1]
    deep = s[s.deep_cut].nsmallest(15, "play_rate_shrunk").sort_values("play_rate_shrunk", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    hbar(axes[0], top.song_title, top.n_plays, BLUE, "{:.0f}")
    clean_axes(axes[0])
    axes[0].set_title("Most-played songs", loc="left", fontsize=12, color=INK, pad=10)
    axes[0].set_xlabel("live performances")

    hbar(axes[1], deep.song_title.str.slice(0, 32), deep.n_plays, MUTED_FILL, "{:.0f}")
    for t in axes[1].get_yticklabels():
        t.set_color(INK_SECONDARY)
    clean_axes(axes[1])
    axes[1].set_title("Deepest cuts", loc="left", fontsize=12, color=INK, pad=10)
    axes[1].set_xlabel("live performances (despite 20+ opportunities)")

    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(PLOTS / "song_frequency.png", dpi=170)
    plt.close(fig)


def fig_model_weights():
    c = pd.read_csv(A / "model_coefficients.csv")
    c = c[c.feature != "intercept"].copy()
    label_map = {
        "ewma_10": "recent play rate (EWMA, 10-show half-life)",
        "shows_since_last": "shows since last played (log)",
        "played_last_show": "played the previous show",
        "career_rate": "career-long play rate",
        "song_age_years": "years since live debut",
        "new_material": "debuted < 1.5 years ago",
        "tour_rate": "play rate so far, this tour",
        "in_tour_pool": "played at least once, this tour",
        "is_special_show": "radio/festival/TV appearance",
    }
    c["label"] = c.feature.map(label_map).fillna(c.feature)
    c = c.reindex(c.coef_standardized.abs().sort_values().index)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [BLUE if v > 0 else RED for v in c.coef_standardized]
    y = np.arange(len(c))
    ax.barh(y, c.coef_standardized, color=colors, height=0.6, zorder=3)
    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(c.label, fontsize=9.5, color=INK)
    lo, hi = c.coef_standardized.min(), c.coef_standardized.max()
    ax.set_xlim(lo - 0.32 * (hi - lo), hi + 0.32 * (hi - lo))
    clean_axes(ax)
    ax.set_xlabel("standardized logistic-regression coefficient")
    ax.set_title("What predicts a song gets played tonight", loc="left", fontsize=12.5, color=INK, pad=12)
    for yi, v in zip(y, c.coef_standardized):
        ax.annotate(f"{v:+.2f}", xy=(v, yi), xytext=(6 if v >= 0 else -6, 0),
                     textcoords="offset points", va="center", ha="left" if v >= 0 else "right",
                     fontsize=8.5, color=INK_SECONDARY)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="raises odds"), Patch(color=RED, label="lowers odds")],
              loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS / "model_weights.png", dpi=170)
    plt.close(fig)


def fig_example_prediction():
    pred = pd.read_csv(A / "model_test_predictions.csv", parse_dates=["date"])
    last_id = pred.sort_values("date").show_id.iloc[-1]
    g = pred[pred.show_id == last_id]
    k = int(g.played.sum())
    top = g.nlargest(k, "p_model").sort_values("p_model")

    fig, ax = plt.subplots(figsize=(8, 7.5))
    colors = [BLUE if p else MUTED_FILL for p in top.played]
    y = np.arange(len(top))
    ax.barh(y, top.p_model, color=colors, height=0.62, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(top.song_key, fontsize=9, color=INK)
    clean_axes(ax)
    ax.set_xlim(0, 1)
    ax.set_xlabel("model probability of being played")
    hit = int(top.played.sum())
    ax.set_title(f"Top-{k} predictions vs. actual setlist — {hit}/{k} correct",
                 loc="left", fontsize=12, color=INK, pad=12)
    ax.text(0.99, -0.09, last_id.replace("_", " "), transform=ax.transAxes,
            ha="right", fontsize=8.5, color=INK_MUTED)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="actually played"), Patch(color=MUTED_FILL, label="predicted, not played")],
              loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS / "example_prediction.png", dpi=170)
    plt.close(fig)


def fig_surprising():
    s = pd.read_csv(A / "surprising_plays.csv", parse_dates=["date"])
    top = s.nlargest(15, "surprisal").iloc[::-1]
    label = top.song_key + "  —  " + top.show_id.str.slice(0, 10)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    hbar(ax, label, top.p_model, ORANGE, "{:.4f}")
    clean_axes(ax)
    ax.set_xlabel("model's predicted probability, before it happened")
    ax.set_title("Most surprising plays (test era, 2023+)", loc="left", fontsize=12.5, color=INK, pad=12)
    fig.tight_layout()
    fig.savefig(PLOTS / "surprising_plays.png", dpi=170)
    plt.close(fig)


def fig_example_comparison():
    import textwrap

    pred = pd.read_csv(A / "model_test_predictions.csv", parse_dates=["date"])
    last_id = pred.sort_values("date").show_id.iloc[-1]
    g1 = pred[pred.show_id == last_id]
    k1 = int(g1.played.sum())
    top1 = g1.nlargest(k1, "p_model").sort_values("p_model")

    top2 = pd.read_csv(A / "historical_tour_example.csv").sort_values("p_model")
    k2 = len(top2)
    hist_tour, hist_date = top2.tour.iloc[0], top2.show_id.iloc[0][:10]

    title1 = f"Album-cycle tour — {last_id[:10]}\ntop-{k1}: {int(top1.played.sum())}/{k1} correct"
    title2 = (
        "Between albums — " + "\n".join(textwrap.wrap(f"{hist_tour}, {hist_date}", 34))
        + f"\ntop-{k2}: {int(top2.played.sum())}/{k2} correct"
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 7.6))
    for ax, top, title in [(axes[0], top1, title1), (axes[1], top2, title2)]:
        colors = [BLUE if p else MUTED_FILL for p in top.played]
        y = np.arange(len(top))
        ax.barh(y, top.p_model, color=colors, height=0.62, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(top.song_key, fontsize=8.5, color=INK)
        clean_axes(ax)
        ax.set_xlim(0, 1)
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    axes[0].set_xlabel("model probability of being played")
    axes[1].set_xlabel("model probability of being played")
    fig.suptitle("Same model, an album-cycle tour vs. a between-albums tour", fontsize=13.5, x=0.02, ha="left", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(PLOTS / "example_comparison.png", dpi=170)
    plt.close(fig)


def fig_surprisal_over_time():
    s = pd.read_csv(A / "show_surprisal.csv", parse_dates=["date"]).dropna(subset=["mean_surprisal_played_bits"])
    s = s.sort_values("date")
    tours = pd.read_csv(A / "tour_surprisal.csv", parse_dates=["first_date", "last_date"])
    tours = tours[tours.n_shows >= 4].sort_values("first_date")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.scatter(s.date, s.mean_surprisal_played_bits, s=7, color=MUTED_FILL, zorder=2, linewidths=0)
    # Per-tour mean, drawn as a horizontal segment over each tour's date
    # span -- touring is bursty (weeks on, months off), so a trailing
    # show-count rolling average saws up and down at tour boundaries; the
    # tour is the natural unit of aggregation here, not a fixed window.
    for r in tours.itertuples():
        ax.plot([r.first_date, r.last_date], [r.mean_surprisal_bits] * 2,
                color=BLUE, linewidth=2.5, solid_capstyle="round", zorder=3)
    clean_axes(ax, x_grid=False, y_grid=True)
    ax.set_ylabel("bits of surprise / song played")
    ax.set_title("How surprising was each setlist? (blue = per-tour average)", loc="left", fontsize=12.5, color=INK, pad=12)
    ax.text(0.01, 0.03, "each grey dot = one show; blue segment = one tour's average, spanning its dates",
            transform=ax.transAxes, fontsize=9, color=INK_MUTED)
    fig.tight_layout()
    fig.savefig(PLOTS / "surprisal_over_time.png", dpi=170)
    plt.close(fig)


def fig_trajectories():
    ts = pd.read_csv(A / "song_timeseries.csv", parse_dates=["date"])
    picks = json.loads((A / "timeseries_examples.json").read_text())
    titles = {
        "classics": "Steady classics",
        "risers": "Rise and plateau",
        "decline": "Decline",
        "revivals": "Fall and revival",
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, (cat, keys) in zip(axes.flat, picks.items()):
        for color, key in zip(CAT_ORDER, keys):
            g = ts[ts.song_key == key]
            ax.plot(g.date, g.play_rate, color=color, linewidth=1.8, label=key, zorder=3)
        clean_axes(ax, x_grid=False, y_grid=True)
        ax.set_ylim(0, 1)
        ax.set_title(titles.get(cat, cat), loc="left", fontsize=11.5, color=INK, pad=8)
        ax.legend(loc="upper left", frameon=False, fontsize=8, handlelength=1.4, labelspacing=0.3)

    for ax in axes[:, 0]:
        ax.set_ylabel("p(played), trailing 2 yrs")
    fig.suptitle("Song popularity over time", fontsize=13.5, color=INK, x=0.02, ha="left", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(PLOTS / "song_trajectories.png", dpi=170)
    plt.close(fig)


def main():
    fig_song_frequency()
    fig_model_weights()
    fig_example_prediction()
    fig_example_comparison()
    fig_surprising()
    fig_surprisal_over_time()
    fig_trajectories()
    print(f"Wrote 7 figures to {PLOTS}")


if __name__ == "__main__":
    main()
