# Project
Estimate censored bike demand, simulate station inventory, allocate a fixed
incentive budget to maximize fill rate. See SPEC.md for full design.

# Current phase
Phase 7 resolved (RESTATED gate, not abandoned -- see DECISIONS.md, "forward
simulator -- stockout-timing gate restated"): src/sim/simulator.py +
src/models/od_shares.py validate against the held-out week (2025-10-06 to
2025-10-13) on trip totals, per-zone volume, and continuous inventory
trajectory (`sanity_true_dest` mode: 3.44%/3.44%/0.901 corr) -- proof the
routing/capacity/N-replay mechanics are sound given true trip-level ground
truth. They do NOT reproduce per-interval stockout timing under stochastic
destination sampling (corr 0.05-0.12), and structurally cannot: destination
is the other half of a realized trip, and sampling it from historical
marginals gets aggregate flow right and realized pairing wrong. Read the
DECISIONS.md entry before touching od_shares.py's backoff hierarchy or the
stockout-correlation gate -- more OD conditioning trades pairing error for
worse sparsity, not a net win, and was deliberately not attempted.

**Follow-up, same day, supersedes the "revisit later" framing above: the
station-hour stockout-rate gap does NOT close with more weeks pooled.**
`src/sim/validate.py --multiweek` ran `stochastic` over 6 held-out weeks
and correlated pooled stockout rate per (station, hour-of-week): corr
0.043 (1 week) -> 0.079 -> 0.100 -> 0.097 (6 weeks) -- plateaus, doesn't
rise, actually ticks down 4->6. The simulator cannot supply P(stockout |
s, t) at station-hour resolution in ANY mode at ANY amount of pooling
tried. See DECISIONS.md's Phase 7 entry follow-up for the full numbers.

**Consequence for Phase 8: MV(s,t) is derived EMPIRICALLY from Phase 4-6
outputs, NOT by simulation.** demand.py's censored demand model +
substitution.py's net-lost estimate, averaged across the full ~52-week
panel (not simulated weeks), give per-(station, hour-of-week) expected
lost demand directly -- no destination assignment involved, so none of the
above applies to it.

**Phase 8 Part A (src/opt/marginal_value.py) is built and run -- see
DECISIONS.md, "station inventory is persistent," before touching this
module.** Short version: the SAME persistence fact (median birth-death
relaxation time ~6.4 days -- rebalancing successfully keeping stations
balanced is exactly what makes them mix slowest) broke a first stationary-
queueing attempt, then bounded a second transient first-passage attempt
(`hitting_probabilities`/`mv_curve`), then briefly looked like it broke
the empirical cross-check too (ratio median 2,062) until segmenting by
inventory level k showed both methods actually agree MV is negligible
above k~3-4 -- the huge ratio there was near-zero divided by near-zero,
not real disagreement. **MV(s,t,k) is therefore only reported for
k <= `mv_k_max` (4, config/params.yaml)** -- silence above that IS the
answer, not a gap, and `station_hour.parquet`/`mv_curve.parquet` should
never be read as claiming MV=0 exactly up there, only "too small to
measure precisely, agreed by both methods." Eligible cells (real inventory
ever reached <= mv_k_max): 265,525/379,019 (70.1%) -- NOT a small pool by
the lifetime-max "ever" criterion, though they do carry 99.9% of total
net-lost demand (near-tautological: a cell that never ran low mostly
can't have lost much).

**"Ever" is the wrong number to size the allocator's problem by --
`eligibility_frequency_report`'s FREQUENCY refinement (DECISIONS.md
follow-up) is the real concentration finding.** Bucketed by what fraction
of a cell's real weeks were low: cells low >50% of weeks (chronic) are
only 7.4% of all cells (27,942/379,019) but carry 44.2% of total
net-lost demand; cells low >25% of weeks (chronic + 25-50% buckets, 20.1%
of cells) carry 77.7%. Loss concentrates hard in the chronic tail, not
spread evenly across rare-and-chronic cells. Prioritize the allocator's
search by low_frac (data/processed/marginal_value/low_frequency.parquet),
not just by eligibility -- "70.1% eligible" answers "ever," not "where
does spend pay off."

**Chronic deficits are overwhelmingly schedulable -- Part B/C must be a
TWO-TIER policy, not uniformly dynamic (DECISIONS.md's "schedulability"
follow-up).** `chronic_timing_summary` found 27,836/27,942 (99.6%) of
chronic cells clear a modal_share >= 0.7 schedulability threshold (median
0.97) -- the deficit recurs at the same 15-min slot within the hour
almost every low week. Checked why before trusting it: 95.4% of those
first-crossings land at the FIRST slot of the hour (position 0), which is
the persistence finding again (a station chronically low at hour t is
already low walking in from hour t-1, per the ~6.4-day relaxation time),
not evidence of a precise mid-hour trigger -- but that's good news for
scheduling: hour-of-week granularity alone is enough, no sub-hour
targeting needed. **Tier 1**: chronic + schedulable cells get a standing
weekly incentive at (station, hour-of-week), no live inventory feed.
**Tier 2**: everything else (non-chronic cells, plus the ~106 chronic-
but-erratic ones) needs state-triggering, same as a uniformly-dynamic
baseline (what Bike Angels effectively does today) would assume for
everything. Building Part B/C as one uniformly-dynamic policy would spend
that operational complexity on the ~44% of net-lost demand that doesn't
need it.

**Chronic cells are NOT unserved -- they already show more inferred
NON-TRIP BIKE MOVEMENT (not "rebalancing" -- see below) than non-chronic
cells at matched demand, and stay chronic anyway (DECISIONS.md's
"chronicity vs. non-trip movement" follow-up). Elasticity for Tier 1 must
be conservative, and the headline claim is "targets residual deficit,"
NOT "reaches where trucks don't."** Checked with the estimation method's
own bias in mind first: Phase 4's N is L1-minimal, so non-chronic cells
are STRUCTURALLY near-guaranteed ~0 inferred inflow regardless of whether
they're actually serviced (inventory.py's own test proves this) -- that
half of the comparison is not fully informative on its own. But chronic
cells specifically show real, demand-scaling, non-zero inflow (median
rises 0.0 -> 0.154 bikes/interval across demand deciles; mean 7.5x
non-chronic's; normalized intensity 12.3x) and remain chronic despite it
-- robust across median, mean, and fraction-nonzero cuts, not an
artifact. This is enough to rule out "genuinely unserved" but NOT enough
to say trucks specifically visit more -- N can't be decomposed into
rebalancing vs. maintenance pulls vs. battery swaps (same limitation as
everywhere else N is used). Part B's Tier-1 elasticity should model
incentive-induced moves as ADDING TO an already-partially-corrected gap
(diminishing returns vs. a
naive full-gap assumption), not filling a virgin one.

**Phase 8 Part B (money config) and Part C (src/opt/allocate.py) are
built.** Read DECISIONS.md's "allocator ranked by net-value-per-dollar"
entry before touching allocate.py's value formula: the first version used
a FLAT per-trip destination value, which made induced_trips cancel
algebraically out of net_value_per_dollar, silently making the ranking
independent of the elasticity curve (a, b) entirely -- caught because the
elasticity sweep came back with Spearman correlation EXACTLY 1.000 across
all 25 draws, too clean to trust, and the algebra confirmed why. Fixed by
using the destination's actual cumulative, k_max-saturating mv curve
(`build_dest_cumulative_mv`) instead of a single ceiling value -- real
diminishing returns, so payout choice (and therefore ranking) genuinely
depends on elasticity now. Re-run: Spearman min=0.812, median=1.000 across
300 draw pairs; **stable core (every one of 25 draws' top 100): 80/100
targets -- that's the actual recommendation**, robust to the elasticity
guess being wrong, per data/processed/allocate/sweep_appearance.parquet.

Origin-side MV uses the same k<=mv_k_max caveat as destinations
(`qualifying_origins`): a "surplus" station's cost is looked up from
mv_curve.parquet at its own worst historical level, not assumed 0 --
0 for the large majority that never dropped into the measured region,
a real measured value for the minority that occasionally do.

Unplanned secondary finding worth knowing before citing Tier 1: the
top-100 ranked-by-$/dollar list is overwhelmingly Tier 2 (only 3-4 Tier 1
targets ever appear), even though Tier 1 carries 44.2% of total net-lost
demand in aggregate. NOT a contradiction -- "most aggregate loss" and
"best marginal $/dollar" are different questions, and Tier 1's
deliberately-conservative elasticity plus its cells' persistent (not
occasional) deficit both push it down THIS specific ranking. Tier 1's
funding case rests on volume + schedulability, not on winning this
ranking; don't conflate the two arguments in the writeup.

**That ranking gap is now known to be structural, not an elasticity-guess
artifact -- see DECISIONS.md's "equal-elasticity ablation" entry before
citing either number.** Re-ran the top-100 ranking with Tier 1 and Tier 2
given IDENTICAL elasticity (7 (a,b) choices spanning the full sweep range,
including each tier's own default): Tier 1 still lands at 3-4/100 every
time -- WORSE than the 35/100 it gets under tier-differentiated elasticity,
not better. Removing Tier 1's conservative handicap should have helped it
if the ranking gap were just that guess; instead it hurt, which rules the
artifact explanation out and points at the destinations themselves. Traced
to mv(k=1), the first-bike value: Tier 2's median runs 23% above Tier 1's,
and its p90/max run ~2x Tier 1's -- the top-100 is decided by that right
tail, and it's the SAME mechanism as the chronicity/non-trip-movement
finding above (a Tier 1 cell's marginal bike lands under sustained,
already-partially-corrected drain, capping its value; a Tier 2 cell's rare
dip is a sharper, fuller event), not a separate story. Keep the two Tier 1
funding arguments distinct in the writeup: it loses on marginal $/dollar
(now on structural grounds), and it wins on aggregate volume +
schedulability -- both real, answering different questions, neither one
evidence against the other.

Not yet applied: max_move_duration_min (no calibrated travel-time model --
same-zone_agg or real od_shares flow used as a plausibility proxy
instead); candidate destinations are prefiltered to the top 3,000 by
ceiling value before origin-matching (tractability, not exhaustive
search, though very unlikely to exclude a real top-100 contender given
how concentrated net-lost demand already is).

Three run modes exist in src/sim/simulator.py (`mode=` on run_simulation):
`sanity_od` and `sanity_true_dest` are diagnostic-only (real trip-level
ground truth, never available going forward). `stochastic` is what Phase 9
(policy comparison) runs -- but ONLY for system-level and per-zone
aggregate lift and continuous inventory trajectories, per the resolution
above; NOT station-level fill-rate/stockout numbers, confirmed to be
simulator noise at that resolution. Phase 9's lift numbers depend on an
unproven-but-stated assumption: destination-assignment error is present in
both the baseline and treatment run under the same seed and is expected to
largely cancel in the DIFFERENCE, not either run's absolute number --
state this explicitly wherever Phase 9 results are reported, don't let it
become an implicit assumption. Phase 8's MV(s,t) does not run the
simulator at all, so this caveat is Phase-9-specific.

Heatmap (Phase 6, heatmap.py) is DIAGNOSTIC ONLY -- ranked by per-dock
lower-confidence-bound (primary) and raw volume (secondary); it shows
where/when scarcity occurs, not where a bike is worth allocating. Ranking
for allocation is Phase 8's job. Next: Phase 8 Part B/C -- config/params.yaml's
money section (dollars_per_point, weekly_budget, elasticity_a/b,
max_induced_moves_per_station_hour, max_move_duration_min) and
src/opt/allocate.py (greedy allocator), per RUNBOOK.

# Memory (non-negotiable)
Machine has 16GB RAM; the panel is ~73-80M rows. Any full-panel operation
must be chunked, streamed, or checkpointed -- never materialize the full
frame in multiple copies (e.g. polars -> pandas conversions on the whole
year, or fitting/predicting over all months at once with no intermediate
write). Always test on one month and report peak RSS (stdlib
`resource.getrusage(resource.RUSAGE_SELF).ru_maxrss`, no psutil) before
running full-year.

# Non-negotiables
- NEVER write code against an assumed schema. Print columns from the actual
  file first. Citi Bike schema changed ~Feb 2021.
- Every model function gets a test with a synthetic fixture where the answer
  is known analytically.
- No silent NaN drops. Log row counts before/after every filter and assert
  expected magnitude.
- Cache expensive steps to parquet keyed by content hash. Never re-parse raw
  CSVs in an interactive loop.
- Money and elasticity assumptions live in config/params.yaml, never inline.

# Commands
make ingest / make features / make train / make simulate / make allocate / make report

# Style
Polars or DuckDB for the heavy joins, pandas only at the edges.
Type hints. No notebooks in src/.

## Gotchas
GBFS snapshots live OUTSIDE the repo at ~/citibike-gbfs-data/ because macOS
TCC blocks cron from touching anything under ~/Documents. Symlinked to
data/raw/gbfs/. Code should always use the repo-relative path. The reference
copy of the poll script is at scripts/gbfs_poll.sh but the live one runs from
~/citibike-gbfs-data/.

inventory.parquet's `inferred_nontrip_in`/`inferred_nontrip_out` columns
are NOT "rebalancing" -- don't call them that in Phase 5 code or writeup.
They're the flow-balance residual: any bike removed/added without a
matching trip, which the data can't distinguish from operator rebalancing
vs. maintenance pulls vs. broken-bike removal vs. e-bike battery swaps. All
of it affects station availability the same way, so it's still the right
input for censoring -- just don't compare it to DOT's rebalancing-only
figure expecting equality (ratio >= 1 is the expected relationship). See
DECISIONS.md, "the overcount direction itself was the real signal."
