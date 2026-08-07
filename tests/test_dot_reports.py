"""Synthetic-fixture tests for src/ingest/dot_reports.py's text parsing.

No real PDF is constructed -- the regex-extraction logic is split out into
a plain-string function specifically so it's testable without one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingest.dot_reports import _extract_rebalanced_from_text  # noqa: E402


def test_extracts_total_and_month_name():
    text = (
        "December 2025 Monthly Report\n"
        "Rebalancing Operations\n"
        "Citi Bike staff rebalanced a total of 52,444 bicycles during the "
        "month of December. The Service Delivery Department utilizes box "
        "trucks, vans, valets, and member incentives ('Bike Angels') to "
        "redistribute bikes system-wide."
    )
    total, month_name = _extract_rebalanced_from_text(text)
    assert total == 52444
    assert month_name == "December"


def test_handles_six_figure_totals():
    text = "Citi Bike staff rebalanced a total of 105,408 bicycles during the month of July."
    total, _ = _extract_rebalanced_from_text(text)
    assert total == 105408


def test_raises_on_missing_sentence():
    with pytest.raises(ValueError, match="no 'rebalanced a total of"):
        _extract_rebalanced_from_text("This report has no rebalancing section at all.")
