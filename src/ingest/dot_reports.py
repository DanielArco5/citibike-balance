"""Parse Citi Bike monthly operating reports into a table of system-wide
monthly rebalancing volume, for cross-checking Phase 4's inferred R against
DOT-reported figures (SPEC.md §4, validation #2).

Two DOT-adjacent report types exist in data/raw/dot_reports/ and are NOT
interchangeable -- confirmed by reading the actual PDFs, not assumed:

- quarterly_usage/ -- NYC DOT's "Bike Share Usage Data Report" (Local Law 99
  of 2015). Trip counts only: by month/quarter/year, and by council/community
  district. No rebalancing, fleet, or operations content anywhere in these
  files. Not parsed here -- there is nothing in them this module needs.

- monthly_operating/ -- Citi Bike's own monthly report to DOT. This is the
  one that reports rebalancing, but only as a single free-text sentence in
  the "Rebalancing Operations" section: "Citi Bike staff rebalanced a total
  of N bicycles during the month of X." One system-wide scalar per month.
  No station-level, no daily, no method (truck/valet/trike/Bike Angels)
  breakdown. Confirmed identical, in all 13 months checked, to the "N
  bike/dock actions" figure quoted separately in the Introduction section --
  it is the same number reported twice, not two independent measurements.

  SLA 11 ("Rebalancing" -- no station outage over 4 hours) reads
  "Performance Rate: NA" in every one of the 13 months checked. Never
  populated with a number. Not usable, not parsed here.

Net effect: validation #2 can only be a system-wide, monthly, order-of-
magnitude check -- never station-level or time-of-day. That's a ceiling on
what DOT data can tell us, not a bug in this parser.
"""
from __future__ import annotations

import calendar
import re
from pathlib import Path

import pdfplumber
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
MONTHLY_DIR = REPO_ROOT / "data" / "raw" / "dot_reports" / "monthly_operating"
OUT_PATH = REPO_ROOT / "data" / "interim" / "dot_monthly_rebalancing.parquet"

REBALANCE_RE = re.compile(
    r"rebalanced a total of ([\d,]+) bicycles during the month of (\w+)"
)

# Filename stem -> (year, month). The report site names the January file
# inconsistently ("JAN-2025" vs "<Month>-2025" for every other month) --
# hand-verified against each PDF's own title page and body text, not
# inferred from the filename.
FILENAME_MONTHS = {
    "December-2024": (2024, 12),
    "JAN-2025": (2025, 1),
    "February-2025": (2025, 2),
    "March-2025": (2025, 3),
    "April-2025": (2025, 4),
    "May-2025": (2025, 5),
    "June-2025": (2025, 6),
    "July-2025": (2025, 7),
    "August-2025": (2025, 8),
    "September-2025": (2025, 9),
    "October-2025": (2025, 10),
    "November-2025": (2025, 11),
    "December-2025": (2025, 12),
}


def _extract_rebalanced_from_text(text: str) -> tuple[int, str]:
    """Returns (rebalanced_bikes_total, month_name_as_reported). Split out
    from PDF reading so the regex is testable on a plain string fixture."""
    match = REBALANCE_RE.search(text)
    if not match:
        raise ValueError(
            "no 'rebalanced a total of N bicycles during the month of X' "
            "sentence found -- report wording changed, don't guess a fix"
        )
    return int(match.group(1).replace(",", "")), match.group(2)


def extract_rebalanced_total(pdf_path: Path, expected_month: int) -> int:
    with pdfplumber.open(pdf_path) as pdf:
        full_text = "\n".join((page.extract_text() or "") for page in pdf.pages)
    total, month_name = _extract_rebalanced_from_text(full_text)

    expected_name = calendar.month_name[expected_month]
    if month_name.lower() != expected_name.lower():
        raise ValueError(
            f"{pdf_path.name}: report text says month '{month_name}' but "
            f"filename implies {expected_name} -- mislabeled file, don't "
            "silently trust the filename"
        )
    return total


def build_table(monthly_dir: Path = MONTHLY_DIR) -> pl.DataFrame:
    rows = []
    for stem, (year, month) in FILENAME_MONTHS.items():
        path = monthly_dir / f"{stem}-Citi-Bike-Monthly-Report.pdf"
        if not path.exists():
            raise FileNotFoundError(path)
        total = extract_rebalanced_total(path, month)
        rows.append(
            {"year": year, "month": month, "rebalanced_bikes_total": total, "source_file": path.name}
        )

    df = (
        pl.DataFrame(rows)
        .with_columns(pl.date(pl.col("year"), pl.col("month"), 1).alias("month_start"))
        .select("month_start", "rebalanced_bikes_total", "source_file")
        .sort("month_start")
    )
    print(
        f"[dot_reports] parsed {df.height} monthly reports -- "
        f"{df['rebalanced_bikes_total'].min():,} to {df['rebalanced_bikes_total'].max():,} "
        "bikes rebalanced/month (system-wide totals, no finer breakdown available)"
    )
    return df


def main() -> None:
    df = build_table()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_PATH)
    print(f"[dot_reports] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
