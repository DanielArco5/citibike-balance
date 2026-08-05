"""Phase 3 gate-check plot (RUNBOOK.md Phase 3): system-wide departures by
hour-of-week, read straight from data/processed/panel.parquet. Not part of
the pipeline -- rerun manually after panel.py changes to eyeball the gate:
twin weekday commuter peaks + a fat Sat/Sun midday hump. If that shape isn't
there, something (almost always a timezone) is wrong upstream.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = REPO_ROOT / "data" / "processed" / "panel.parquet"
OUT_PATH = REPO_ROOT / "reports" / "figures" / "departures_by_hour_of_week.png"

BLUE = "#2a78d6"
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def main() -> None:
    by_how = (
        pl.scan_parquet(PANEL_PATH)
        .group_by("hour_of_week")
        .agg(pl.col("departures").sum().alias("departures"))
        .sort("hour_of_week")
        .collect()
    )
    assert by_how.height == 168, f"expected 168 hour-of-week buckets, got {by_how.height}"

    x = by_how["hour_of_week"].to_numpy()
    y = by_how["departures"].to_numpy()

    fig, ax = plt.subplots(figsize=(11, 4.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(x, y, color=BLUE, linewidth=2, solid_capstyle="round")

    for d in range(1, 7):
        ax.axvline(d * 24, color=GRID, linewidth=1, zorder=0)

    ax.set_xlim(0, 167)
    ax.set_ylim(bottom=0)
    ax.set_xticks([d * 24 + 12 for d in range(7)])
    ax.set_xticklabels(DAY_LABELS, color=INK_SECONDARY, fontsize=10)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", colors=INK_MUTED, labelsize=9, length=0)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1000:,.0f}k"))

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)

    ax.set_title(
        "System-wide departures by hour of week",
        color=INK_PRIMARY,
        fontsize=13,
        loc="left",
        pad=14,
    )
    ax.set_ylabel("Departures", color=INK_SECONDARY, fontsize=10)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=180)
    print(f"[plot] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
