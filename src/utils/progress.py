"""Progress/memory logging for chunked full-panel pipelines (Phase 6
restructure -- see /Users/danielcrown1/.claude/plans/lucky-leaping-zebra.md).
stdlib only (`resource`), no psutil, per CLAUDE.md's no-new-deps-beyond-
what's-used pattern."""
from __future__ import annotations

import resource
import sys


def peak_rss_mb() -> float:
    """Peak resident set size of this process so far, in MB.
    `ru_maxrss` is bytes on macOS/BSD but KiB on Linux -- this dev machine is
    macOS, so we divide by 1024**2. Flip to /1024 if this ever runs on
    Linux CI."""
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return maxrss / divisor


def log_month(month_key: str, n_rows: int, elapsed_s: float, extra: str = "") -> None:
    suffix = f", {extra}" if extra else ""
    print(
        f"[pipeline] {month_key}: {n_rows:,} rows, {elapsed_s:.1f}s, "
        f"peak RSS {peak_rss_mb():.0f} MB{suffix}"
    )
