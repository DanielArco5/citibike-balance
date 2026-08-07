"""Synthetic-fixture tests for src/models/inventory.py.

Each test constructs a station's net_flow trajectory where the correct
answer is known analytically -- either because we injected a specific
non-trip movement and can check it's recovered, or because the physics
force a unique answer regardless of method (e.g. a trajectory that never
leaves bounds must be reconstructed with zero non-trip movement).

Terminology note: this quantity was originally called "rebalancing" (R);
renamed to "non-trip movement" (N) after the DOT cross-check showed the
flow-balance signal can't distinguish operator rebalancing from
maintenance pulls, broken-bike removal, or e-bike battery swaps -- see
inventory.py's module docstring and DECISIONS.md. The math is unchanged;
only the name is.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.inventory import (  # noqa: E402
    ReconstructionParams,
    reconstruct_station,
    reconstruct_station_greedy,
    solve_station_week_lp,
)

PARAMS = ReconstructionParams(anchor_prior_weight=2.0, week1_anchor_fraction_of_capacity=0.5)


def test_no_violation_needs_no_nontrip_movement():
    """A station that never leaves [0, capacity] under N=0 must be
    reconstructed with N=0 everywhere -- the minimal correction is the
    empty one, and the anchor should land exactly on the prior since
    nothing in the data contradicts it."""
    capacity = 20.0
    net_flow = np.array([1.0, -1.0, 2.0, -2.0, 0.0, 1.0, -1.0])
    prior = capacity * 0.5

    solved = solve_station_week_lp(net_flow, capacity, prior, PARAMS.anchor_prior_weight)
    assert np.allclose(solved["N_in"], 0.0)
    assert np.allclose(solved["N_out"], 0.0)
    assert solved["I"][0] == pytest.approx(prior)
    # I[t] should just be the prior plus the running net flow.
    expected = prior + np.concatenate([[0.0], np.cumsum(net_flow)])
    assert np.allclose(solved["I"], expected)


def test_recovers_injected_nontrip_inflow():
    """Departures alone would drive inventory negative -- the LP must
    inject exactly enough N_in, at the point of violation, to hold the
    floor at 0. With a single violation event the L1-minimal correction
    is unique and should match the shortfall exactly."""
    capacity = 20.0
    prior = 5.0
    # anchor=5, then depart 8 in one interval: naive would go to -3.
    net_flow = np.array([-8.0, 0.0, 0.0])

    solved = solve_station_week_lp(net_flow, capacity, prior, PARAMS.anchor_prior_weight)
    assert solved["N_in"][0] == pytest.approx(3.0)
    assert np.allclose(solved["N_out"], 0.0)
    assert (solved["I"] >= -1e-9).all()
    assert solved["I"][1] == pytest.approx(0.0)  # clipped to the floor


def test_recovers_injected_nontrip_outflow():
    """Symmetric case: arrivals alone would overflow capacity."""
    capacity = 10.0
    prior = 8.0
    net_flow = np.array([5.0, 0.0])  # naive: 8 + 5 = 13 > capacity 10

    solved = solve_station_week_lp(net_flow, capacity, prior, PARAMS.anchor_prior_weight)
    assert solved["N_out"][0] == pytest.approx(3.0)
    assert np.allclose(solved["N_in"], 0.0)
    assert solved["I"][1] == pytest.approx(10.0)


def test_bounds_never_violated_multiweek_chain():
    """A multi-week station with a mix of slack and violating weeks: after
    reconstruction every single interval (across the whole chained
    sequence, not just within one LP solve) must sit in [0, capacity]."""
    rng = np.random.default_rng(0)
    capacity = 15.0
    n_weeks = 4
    week_len = 20
    n = n_weeks * week_len

    interval_start = np.arange(n)
    # Deliberately large swings relative to capacity so several bound
    # violations are forced across the series.
    net_flow = rng.integers(-6, 7, size=n).astype(np.float64)
    week_start = np.repeat(np.arange(n_weeks), week_len)

    result = reconstruct_station(interval_start, net_flow, week_start, capacity, PARAMS)
    assert (result["inventory"] >= -1e-6).all()
    assert (result["inventory"] <= capacity + 1e-6).all()


def test_chained_anchor_uses_prior_week_ending_inventory():
    """Week 2's anchor prior should be week 1's ending inventory, not the
    week1 default -- verify by checking week 2's start is close to where
    week 1 left off when week 2 itself has no violations to override it."""
    capacity = 20.0
    week_len = 5
    interval_start = np.arange(2 * week_len)
    week_start = np.repeat([0, 1], week_len)
    # Week 1: net drift of +6 from a mid-capacity anchor (10 -> 16), no violations.
    # Week 2: flat (no net flow), so nothing should pull the anchor away from
    # wherever week 1 ended.
    net_flow = np.array([2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    result = reconstruct_station(interval_start, net_flow, week_start, capacity, PARAMS)
    week1_end = 10.0 + 6.0  # prior (capacity/2) + sum of week-1 net flow
    assert result["inventory"][week_len] == pytest.approx(week1_end)
    assert np.allclose(result["nontrip_in"], 0.0)
    assert np.allclose(result["nontrip_out"], 0.0)


def test_greedy_also_respects_bounds():
    capacity = 12.0
    net_flow = np.array([-5.0, -5.0, 8.0, 8.0, -3.0])
    result = reconstruct_station_greedy(net_flow, capacity, anchor=6.0)
    assert (result["inventory"] >= 0).all()
    assert (result["inventory"] <= capacity).all()


def test_lp_total_correction_is_at_most_greedy():
    """Not a formal proof, just a sanity property: the LP finds a globally
    minimal-|N| solution over the whole week, so it should never need MORE
    total correction than a reactive greedy clip over the same trajectory."""
    capacity = 10.0
    anchor = 5.0
    rng = np.random.default_rng(1)
    net_flow = rng.integers(-4, 5, size=30).astype(np.float64)
    week_start = np.zeros(30, dtype=int)
    interval_start = np.arange(30)

    lp = reconstruct_station(interval_start, net_flow, week_start, capacity, PARAMS)
    greedy = reconstruct_station_greedy(net_flow, capacity, anchor)

    lp_total = lp["nontrip_in"].sum() + lp["nontrip_out"].sum()
    greedy_total = greedy["nontrip_in"].sum() + greedy["nontrip_out"].sum()
    assert lp_total <= greedy_total + 1e-6
