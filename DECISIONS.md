# Decisions

Real design forks hit while building this, and the actual reasoning behind
how they were resolved -- not a changelog summary. Written to be reread
before an interview, when "why did I do it that way" needs a real answer.

---

## Phase 8 (2026-08-13): equal-elasticity ablation on the Tier 1/Tier 2 ranking gap -- the reversal that makes it structural, not an artifact of the conservative-elasticity guess

The entry below this one left a live objection unaddressed: Tier 1's
elasticity (`elasticity_tier1_scheduled`, a=2.0/b=0.15) is DELIBERATELY more
conservative than Tier 2's (a=3.0/b=0.25, per the chronicity-vs-non-trip-
movement follow-up two entries down) -- so a fair critic could say the
Tier1/Tier2 ranking gap is just that guess doing the work, not a real
difference between the destinations themselves. Tested directly rather than
left open: re-ran the top-100 ranking with Tier 1 and Tier 2 given IDENTICAL
elasticity parameters, across seven (a, b) choices spanning the full
plausible range -- Tier 2's own default, Tier 1's own default, the midpoint
(2.5, 0.2), and both sweep extremes.

**Result: Tier 1 lands at 3-4/100 under every identical-elasticity choice
tested -- LESS than the 35/100 it gets under the original tier-differentiated
setup, not more.** Removing Tier 1's "conservative" handicap and giving it
the SAME curve Tier 2 gets (including Tier 2's own, more generous, a/b) made
Tier 1's showing worse, not better.

**Why this is stronger evidence than a null result.** The test was posed
against three possible outcomes before running it: still loses under equal
elasticity -> structural, stands on its own; wins or competes -> the ranking
is largely an artifact of the conservative guess; in between -> report the
crossover ratio at which Tier 1 starts entering the top 100. A plain "still
loses, e.g. 20-30/100" would have been consistent with "the conservative
guess matters somewhat but isn't the whole story" -- compatible with either
the artifact explanation or the structural one, not decisive between them.
Getting WORSE when the handicap is removed rules the artifact explanation out
directly: if Tier 1's low ranking were an elasticity-assumption artifact,
giving it Tier 2's own elasticity curve should have moved it toward Tier 2's
territory, not further away. It moved further away. That direction can only
be explained by something in the destinations themselves, independent of
whichever elasticity curve either tier is assigned -- a stronger, more
specific claim than "the result is robust to this assumption," and one that
only the reversal, not a same-direction null result, could have supported
this cleanly.

**Traced to the actual value curves, not asserted from the ranking outcome
alone:**

| | ceiling value (median) | mv(k=1) median | mv(k=1) p90 | mv(k=1) max |
|---|---|---|---|---|
| Tier 1 | 1.45 | 0.58 | 0.91 | 2.58 |
| Tier 2 | 1.54 | 0.71 | 1.17 | 5.72 |

Ceiling values are close (within 6%) -- the gap that matters is in the FIRST
increment and its tail: Tier 2's median mv(k=1) runs 23% higher than Tier
1's, and its p90 and max both run roughly 2x Tier 1's. A top-100-by-net-
value-per-dollar ranking is decided by the best individual opportunities,
i.e. the right tail of this distribution, so Tier 2's fatter tail of
occasional, acute, sharply-resolvable deficits wins the ranking regardless of
which elasticity curve is layered on top -- exactly why re-weighting the
elasticity multiplier couldn't move the outcome.

**Same mechanism as the chronicity/non-trip-movement finding two entries
down, not a new story layered next to it.** That entry established that
chronic (Tier 1) cells already receive some non-trip inflow (median rising
0.0 -> 0.154 bikes/interval across demand deciles, 7.5x non-chronic cells'
mean, 12.3x their demand-normalized intensity) and remain chronic despite it
-- i.e. Tier 1's marginal bike lands at a station under sustained,
only-partially-corrected drain, which is precisely what a low, thin-tailed
mv(k=1) describes quantitatively. A Tier 2 cell's occasional dip is, by
construction, not under that kind of standing drain -- its rare stockouts are
sharper, fuller events, which is what a fat right tail describes. The
elasticity-sweep entry below named this qualitatively ("Tier 1's persistently
low[...] cells" capture "a smaller fraction of their large ongoing deficit");
this ablation is the mechanism-level confirmation of that same claim,
isolated from the elasticity guess entirely rather than argued from it.

**Consequence: Tier 1's funding case rests on two genuinely separate
arguments, and the writeup must keep them apart, not conflate them.**

- **Loses on marginal $/dollar, now on structural grounds, not an assumption
  artifact:** 3-4/100 to 35/100 across every elasticity choice tested,
  tier-differentiated or identical alike. This answers "where does the next
  incremental incentive dollar do the most good right now" -- and the answer
  is Tier 2, robustly.
- **Wins on aggregate volume and schedulability, a completely different
  question:** Tier 1 (chronic + schedulable cells) carries 44.2% of total
  system-wide net-lost demand, and 99.6% of chronic cells (27,836/27,942)
  clear the 0.7 schedulability threshold -- a standing weekly incentive keyed
  to (station, hour-of-week), no live inventory feed required, materially
  cheaper to operate than Tier 2's necessarily state-triggered pricing. This
  answers "where does a fixed operational policy capture the most total loss
  most cheaply" -- and the answer is Tier 1.

Both are true and both are legitimate reasons to fund Tier 1; neither implies
the other. The ranking result above is not evidence against the
volume/schedulability case, any more than the volume case is evidence against
the ranking result. The writeup should state both, name the question each
one answers, and not let "Tier 1 loses" (true of the marginal-dollar ranking)
bleed into "Tier 1 isn't worth funding" (not supported, and contradicted by
the volume/schedulability case).

---

## Phase 8 (2026-08-13): allocator ranked by net-value-per-dollar with a flat per-trip value -- and the elasticity sweep caught that it had silently stopped depending on elasticity at all

`src/opt/allocate.py`'s first version computed, for a candidate move at
payout p: `trips_value = induced_trips(p) * (dest_mv - origin_cost)` and
`dollar_cost = induced_trips(p) * p * dollars_per_point`, then ranked by
`net_value_per_dollar = trips_value/dollar_cost - 1`. Running the
elasticity sweep (a in [1,5], b in [0.05,0.5], 25 draws) against this
produced Spearman rank correlation of EXACTLY 1.000 across all 300
draw-pairs, zero variation, every draw's top-100 identical.

**That result was too clean to report as "the recommendation is
robust," and checking the algebra confirmed why: it was a formula
artifact, not a finding.** `induced_trips(p)` is a common factor in both
the numerator and denominator of `net_value_per_dollar` when `dest_mv` is
a single flat value applied regardless of volume -- it cancels
completely: `net_value_per_dollar = (dest_mv - origin_cost)/(p *
dollars_per_point) - 1`, independent of `a` and `b` entirely, and
monotonically decreasing in payout `p`, which meant the "best payout"
search was silently always picking the SMALLEST value in `PAYOUT_GRID`
regardless of elasticity. The sweep didn't fail to find instability --
the ranking criterion, as originally written, had no mechanism by which
elasticity COULD affect it. A perfectly stable result from a metric that
structurally cannot vary is not evidence of robustness; it's a sign the
test isn't testing what it looks like it's testing.

**Fix: `trips_value` uses the destination's actual CUMULATIVE, SATURATING
mv curve (`build_dest_cumulative_mv` -- MV(1) + MV(2) + ... + MV(k),
capped hard at the destination's own measured k_max), not a flat ceiling
value.** This reintroduces genuine diminishing returns: value stops
growing once induced volume passes k_max, cost keeps rising linearly with
payout, so there's a real interior trade-off, and WHICH payout is
"best" -- and therefore which destinations rank highest -- now actually
depends on how fast `a*(1-e^{-bp})` grows, i.e. on the elasticity curve.
Re-run: Spearman correlation min=0.812, median=1.000, max=1.000 across
the same 300 pairs -- most draws still agree closely (median 1.0), but a
real, non-degenerate range now exists, and it's now trustworthy because
the mechanism that would let it vary is actually present.

**Stable core (targets in every one of the 25 draws' top 100): 80 of
100.** Per this phase's framing, that's the actual recommendation --
station-hours worth funding regardless of what the true elasticity turns
out to be, not conditional on the specific (a, b) guess documented (and
flagged as ungrounded) in config/params.yaml.

**Secondary finding, unplanned but worth stating plainly: the top 100 by
net-value-per-dollar is overwhelmingly Tier 2 (dynamic), not Tier 1
(scheduled), even though Tier 1 carries 44.2% of total net-lost demand in
aggregate (DECISIONS.md's schedulability entry).** Only 4 of 120 targets
that ever appear across the sweep, and 3 of the 80-target stable core,
are Tier 1. This is not a contradiction of the earlier concentration
finding -- "which tier carries the most AGGREGATE loss" and "which
individual station-hours offer the best MARGINAL return per incentive
dollar" are different questions. Tier 1's elasticity is deliberately more
conservative (config's stated reasoning: chronic cells already show
measurable non-trip movement and stay chronic despite it, so an induced
move tops up an already-addressed gap) and Tier 1 cells are persistently
low, so a single nudge captures a smaller fraction of their large ongoing
deficit than the same nudge captures at a Tier 2 cell's occasional dip.
The allocator's ranked output is genuinely dominated by opportunistic
Tier 2 targets; Tier 1's case for funding rests on aggregate volume and
schedulability (a standing weekly cost with no live-feed requirement), not
on winning this specific per-dollar ranking. Both are legitimate reasons
to fund something; they are not the same reason, and the writeup should
not conflate them.

**Origin cost applies the same k <= mv_k_max discipline used everywhere
else** (`qualifying_origins`): a station only qualifies as a surplus
origin if its OWN low_frac is low (rarely low itself, config's
`origin_max_low_frac`), and its cost is looked up from the SAME
mv_curve.parquet at its own worst historical level, capped at mv_k_max --
0 for the large majority that never dropped into the measured region,
a real measured value for the minority that occasionally do. No origin's
cost is assumed zero blindly; it's zero because it was checked and
nothing was measured there.

**What this module does NOT yet do, stated rather than hidden:**
`max_move_duration_min` (config) is not applied -- there's no calibrated
travel-time model in this project, so candidate OD pairs are restricted
to same-zone_agg or real historical station-to-station flow (od_shares)
as a plausibility proxy instead. Candidate destinations are prefiltered to
the top `PREFILTER_N` (3,000) by ceiling value before origin-matching, for
tractability against the ~265K-cell eligible population -- given how
concentrated net-lost demand is in the chronic tail, this is very unlikely
to exclude a genuine top-100 contender, but it is a real, stated bound,
not an exhaustive search. Origin cost scales LINEARLY with induced trips
(no equivalent cumulative-saturation treatment on the origin side) -- a
smaller simplification than the destination-side bug just fixed, since
most qualifying origins cost 0 regardless, but a real one, left for a
later pass if origin-side volume ever becomes material.

---

## Phase 8 (2026-08-12): station inventory is persistent -- it broke the stationary queueing model, then bounded the transient one, then confounded the empirical cross-check. Same fact, three symptoms.

`src/opt/marginal_value.py` (MV(s,t,k), SPEC.md §7) went through two model
formulations before landing on one worth trusting, and the reason the
first one failed is the same reason the second one has a real, bounded
limitation rather than a clean answer everywhere. Worth stating once,
plainly, rather than as three separate footnotes.

**Attempt 1: stationary M/M/1/K queue, rejected on a checked number, not a
hunch.** Modeled each station-hour's bike count as a birth-death chain and
used its long-run equilibrium P(0 bikes) as P(stockout), varying capacity
as a proxy for "adding a bike." Checked the chain's relaxation time before
trusting the equilibrium assumption (median 9,249 minutes -- ~6.4 days --
across 379,019 station-hour cells; p90 ~206 days). Root cause, also
checked rather than assumed: 63% of cells sit at rho=lambda/mu in
[0.8, 1.25] -- because Phase 4's rebalancing is DESIGNED to keep stations
balanced, and successfully keeping a station balanced is exactly the
condition under which a birth-death chain mixes slowest. A station whose
supply is actively, successfully managed to stay near equilibrium is,
perversely, the station whose queue take longest to reach that
equilibrium from any given start. Not a modeling bug -- a real property
of the system this project is trying to describe.

**Attempt 2: transient first-passage.** Reframed the question to what it
actually was operationally: "P(this station runs out within the next
hour | it has k bikes now)," not "what fraction of all time is it at
zero." Same birth-death chain, same LATENT (censored-demand-model, not
observed) rates, but state 0 made absorbing and the one-step transition
matrix (net change = Skellam(lambda) - Skellam-equivalent departures,
same Poisson draw the forward simulator itself uses) raised to the
4-step (1-hour) power -- standard first-passage construction, no
stationarity assumption needed. `hitting_probabilities` /
`check_concavity` in marginal_value.py. This is NOT provably concave by
construction (unlike the stationary blocking probability) -- checked
directly: 370,128/379,019 cells (97.7%) concave everywhere, 8,891 with at
least one real violation, which the model reports (`first_concavity_
violation_k`), not papers over.

**The cross-check surfaced the SAME persistence fact a third time, and at
first looked like a much worse failure than it was.** The local-
sensitivity cross-check (empirical slope of real cross-week net-lost-
demand against real cross-week starting inventory, no model involved)
should, under the reframing, measure the same quantity as the model --
both are now conditional-on-a-starting-level, not the stationary-vs-
conditional mismatch that made the FIRST cross-check read near zero.
First full run: empirical/model ratio median 2,062, p90 908,620 --
looked catastrophic. Segmenting by the matched inventory level k
(not run further with vague suspicion, but a direct filter) showed why:
at k <= 3 (248 of ~372K matched cells), ratio median 1.88 -- same order
of magnitude, real but modest disagreement. At k > 3 (the other 99.93%),
BOTH model_mv and empirical_mv were within noise of zero (medians
0.000000 and -0.000000) -- the "ratio" there is near-zero divided by
near-zero, an artifact of the division, not evidence the two methods
disagree. The persistence that broke Attempt 1 is exactly why: once a
station carries more than a handful of bikes, its supply state is highly
autocorrelated across hours (same slow-mixing dynamics), so (a) the
model's 1-hour window genuinely won't see it run dry from a well-stocked
start, correctly, and (b) the empirical regression's real-but-small
residual signal at high k is itself most likely confounded BY that same
autocorrelation (a week's mean_inventory being on the low side of normal
at well-stocked levels is a marker for a broader lean stretch, not a
random draw) -- not a sign the model is missing something at those
levels.

**Resolution: MV(s,t,k) is reported only for k <= `mv_k_max` (4,
config/params.yaml), not because the model can't produce a number above
that, but because neither method's number is trustworthy up there, and
they don't need to disagree loudly to prove it -- they agree it's
negligible.** `build_mv_curve_table` truncates the emitted curve at
min(capacity, mv_k_max); a k with no row IS the answer for that k, not a
gap. Re-run with truncation in place: 1,143 cells now have both a
same-regime model AND cross-check estimate to compare (down from the
prior run's misleading 372,222 "matches," almost all in the meaningless
high-k regime) -- ratio median 4.71, p10 1.28, p90 107.28. Still not 1.0,
real remaining disagreement worth further work, but the same order of
magnitude, not the earlier four-orders-of-magnitude illusion.

**Eligibility, quantified, not assumed small.** Station-hours whose real
inventory ever reached <= mv_k_max bikes in at least one of the panel's
~52 weeks: **265,525 / 379,019 (70.1%)** of all (station, hour-of-week)
cells -- NOT a small pool. Those cells carry **99.9%** of total net-lost
demand (384,690 / 385,267 trips, hour-window units) -- close to a
near-tautology (a cell that never ran low mostly can't have lost much
demand) rather than a surprising concentration finding. Practical
consequence for Phase 8's allocator: it can exclude the 29.9% of cells
that never came close to empty with essentially zero cost, but this is a
real, useful trim of the search space, not the dramatic reduction "a
small set carries most of the loss" would have implied. State the actual
number, not the more flattering one that was hypothesized before it was
checked.

**Honest summary for whoever builds Phase 8's allocator on top of this:
MV is MEASURABLE where a station-hour is near-empty (k <= 4) --
cross-checked by two independent methods that now agree in magnitude --
and only BOUNDABLE (known to be small, not given a precise value) where
it isn't.** Both are real, reportable results. Treating the k > 4 region
as "MV = 0, precisely" would overclaim precision the data doesn't
support; treating it as "unknown" would discard what both methods agree
on. `station_hour.parquet`/`mv_curve.parquet` report exactly the
distinction: rows for k <= mv_k_max, silence above it.

**Follow-up, same day: "ever reached k <= mv_k_max" was the wrong
eligibility criterion, and the corrected one changes the answer to the
concentration question above.** Incentive spend is a RECURRING policy --
a cell that ran low once in 52 weeks and one that runs low every week are
not the same allocation target, even though "ever" treats them
identically. `eligibility_frequency_report` computes, per (station,
hour-of-week), the FRACTION of its real observed weeks that were low, not
just whether any were:

| bucket | cells | share of cells | share of net-lost demand |
|---|---|---|---|
| never low | 113,494 | 29.9% | 0.15% |
| low 0-10% of weeks | 122,623 | 32.4% | 6.7% |
| low 10-25% of weeks | 66,705 | 17.6% | 15.4% |
| low 25-50% of weeks | 48,255 | 12.7% | 33.5% |
| low >50% of weeks (chronic) | 27,942 | 7.4% | **44.2%** |

This IS the concentration finding the lifetime-max criterion missed.
Cells low more than a quarter of the time (25-50% + chronic, 20.1% of all
cells) carry 77.7% of total net-lost demand; the chronic bucket alone --
7.4% of cells, under a tenth -- carries 44.2%, nearly half, by itself.
Loss is NOT spread evenly across rare and chronic cells: it concentrates
hard in the chronic tail. Phase 8's allocator should prioritize by
low_frac, not just by eligibility -- the "70.1% eligible" figure above is
still the correct answer to "ever," but it is the wrong number to reason
about search-space size or expected payoff from; the chronic-and-frequent
tail is where the effective problem actually lives.

**Follow-up, same day: chronic deficits are overwhelmingly SCHEDULABLE,
and the reason why is the persistence finding showing up a fourth time,
not a new discovery of intra-hour clockwork.** The policy-shape question:
if a chronic cell (low_frac > 0.5) always crosses <= mv_k_max at the same
clock time each week, a standing weekly incentive catches it with no live
inventory feed -- operationally far cheaper than dynamic, state-triggered
pricing. If timing scatters within the hour, it needs state-triggering
like everything else, chronic or not. `chronic_timing_summary` locates,
per chronic cell, the 15-min sub-interval (0-3, i.e. :00/:15/:30/:45 into
the hour) where each low week FIRST crossed the threshold, and measures
how consistent that position is across the cell's low weeks (modal_share
= fraction of low weeks matching the single most common position).

Result, across all 27,942 chronic cells: modal_share median **0.97**
(p25 0.93, p75 1.00), std_position median 0.24 (out of a max ~1.29 for a
uniform 4-way split) -- **27,836 / 27,942 (99.6%) clear the 0.7
schedulable threshold** (config/params.yaml's
`schedulable_modal_share_threshold`). Split by how chronic: 3,535 cells
(12.7% of the chronic bucket) are low in >80% of weeks, 24,407 (87.3%)
in 50-80%.

**Checked why the number is this clean before reporting it at face value:
95.4% of all chronic-cell low-weeks first cross the threshold at
position 0 -- the very FIRST 15-min slot of the hour, not a specific
moment mid-hour.** 99.97% of chronic cells have position 0 as their
modal position. This is the SAME persistence fact as the rest of this
entry, not a new one: a station chronically low at hour t is, per the
~6.4-day relaxation time, almost certainly already low walking INTO hour
t from whatever came before it -- so "the deficit is present at :00" is
what persistence predicts, not evidence of a precise trigger event partway
through the hour. Practically this is good news, not a caveat that weakens
the finding: it means hour-of-week granularity (already this whole
module's unit) is sufficient for scheduling -- no need to reason about
which 15-min slot within the hour to target, since it's essentially always
the first one.

**Consequence: Part B/C should be a TWO-TIER policy, not a single dynamic
one.** Tier 1 -- chronic AND schedulable cells (27,836 of 27,942 chronic
cells, carrying most of the 44.2% chronic-bucket net-lost share): a
standing weekly incentive keyed to (station, hour-of-week), no live
inventory feed, materially cheaper to operate than continuous dynamic
pricing and a different recommendation than what Bike Angels does today
(which is state-triggered). Tier 2 -- everything else (all non-chronic
cells, plus the 106 chronic-but-erratic ones that miss the schedulability
threshold): needs state-triggering, same as any dynamic-pricing baseline
would assume uniformly. Building Part B/C as uniformly dynamic would spend
the same operational complexity on the 44% of loss that doesn't need it as
on the 56% that does.

**Follow-up, same day: chronic cells are NOT unserved -- they already
show materially MORE inferred NON-TRIP BIKE MOVEMENT than non-chronic
cells at matched demand, and stay chronic anyway.** Before trusting a
direction, the comparison method's own structural bias had to be checked
first: Phase 4's N is inferred by an L1-MINIMAL-correction LP
(`src/models/inventory.py`) that credits `inferred_nontrip_in` ONLY where
a station's organic (N=0) trajectory would otherwise violate the lower
bound -- so a station that never gets close to empty is structurally
GUARANTEED near-zero inferred inflow REGARDLESS of whether it's actually
serviced (confirmed by inventory.py's own
`test_no_violation_needs_no_nontrip_movement`). This means "non-chronic
cells show ~0 inferred inflow" is partly a MECHANICAL property of the
estimation method, not proof those cells are never serviced -- the
comparison can't speak to whether well-stocked stations are over- or
under-served. It CAN speak to what's happening at chronic cells
specifically, where the LP is forced to credit something whenever a
violation would otherwise occur.

Word choice matters here and is deliberate: this is NON-TRIP BIKE
MOVEMENT (`inferred_nontrip_in`), not "rebalancing." Per the Phase 4
R-to-N reframe above, the flow-balance signal can't distinguish operator
rebalancing from maintenance pulls, broken-bike removal, or e-bike
battery swaps -- N is a superset, and nothing in this comparison
decomposes it. That's still enough to rule out hypothesis 2 (a
genuinely-unserved station wouldn't show elevated non-trip inflow of ANY
kind, rebalancing or otherwise), but it does not establish that trucks
specifically visit chronic cells more, only that SOME non-trip cause adds
bikes there more than at matched non-chronic cells.

`rebalancing_vs_chronicity_report` (function name predates this wording
correction; the quantity it measures is non-trip inflow, not isolated
rebalancing), controlling for organic departure demand (`dep_rate`
decile, NOT `mu` which already folds non-trip flow in -- conditioning on
it would be circular): in EVERY SINGLE demand decile, non-chronic cells'
median `inferred_nontrip_in_rate` is exactly 0.0, while chronic cells'
median is positive and rises monotonically with demand (0.0 in the
lowest decile to 0.154 bikes/interval in the highest). Robust across
every cut checked, not a median artifact: 70.9% of chronic cells show
SOME positive inflow vs. 15.3% of non-chronic; mean inflow 0.060 vs.
0.008 (7.5x); mean demand-normalized "non-trip inflow intensity" (inflow
/ dep_rate) 0.122 vs. 0.010 (12.3x).

**Reading, with both caveats above in mind: chronic cells demonstrably DO
show measurable, demand-scaling non-trip bike movement -- more so than
non-chronic cells show, wherever either shows any at all -- and remain
chronic (low_frac > 0.5) despite it.** This is the FIRST of the two
hypothesized directions, not the second: NOT "genuinely unserved, outside
efficient truck routes" (which would need near-zero non-trip inflow at
chronic cells too, and it doesn't show that), but "already receiving
some non-trip inflow that isn't sufficient." What does occur (median
~6-8% of organic departure demand, chronic cells) is a partial patch, not
a complete absence of non-trip movement.

**Consequence for Part B: elasticity should be MORE CONSERVATIVE for the
scheduled (Tier 1) cohort, and the headline claim is the correspondingly
more modest one.** Tier 1's incentive-induced moves add to an ALREADY-
PARTIALLY-ADDRESSED gap, not a virgin one -- some non-trip inflow is
already reaching these cells and is demonstrably insufficient on its own,
so incremental moves should be modeled with diminishing returns relative
to a naive full-gap elasticity. The writeup should say the scheduled tier
targets RESIDUAL deficit that persists despite existing non-trip bike
movement, not unaddressed deficit -- "rider incentives reach exactly
where trucks don't" is NOT supported by this data (it isn't isolated to
trucks at all) and should not
be the framing. The honest, still-strong version: incentives can close
a gap that whatever's currently happening (mechanically inferred or
genuinely operational) does not.

---

## Phase 7 (2026-08-12): forward simulator -- stockout-timing gate restated after diagnosing a structural (not fixable) limitation

RUNBOOK Phase 7's original gate (SPEC.md §4's forward-simulation
validation) required per-interval stockout-minutes correlation > 0.7
against a held-out week (2025-10-06 to 2025-10-13, all 2,267 usable
stations, 15-min steps). The simulator (`src/sim/simulator.py`,
`src/models/od_shares.py`) never cleared that bar in any run mode --
stochastic: 0.077-0.115 across reruns; forced-departures/OD-share-
destinations: 0.05-0.06.

**Diagnosis, not tuning, per RUNBOOK's own instruction.** Three run modes
were built specifically to isolate WHERE the error lives, not to search
for a parameter that makes the number look better:

- `sanity_od`: departure COUNTS forced to real history, destinations still
  drawn from `od_shares`' backoff-hierarchy model (origin x hour-of-week ->
  zone x hour-of-week -> zone x daypart -> global).
- `sanity_true_dest`: departure counts AND destinations both forced to
  real history (`simulator.load_true_destination_trips`) -- isolates
  everything else (routing, capacity constraints, baseline-N replay) from
  destination-assignment error specifically.
- `stochastic`: the real forward simulation -- departures sampled from the
  fitted demand model, destinations from `od_shares`.

Baseline-N-replay clipping was ruled out first: net clipping as a fraction
of the ~37,690-bike fleet-size proxy is -6.4% (`sanity_od`) to -10.0%
(`stochastic`), but drops to **-0.5%** in `sanity_true_dest` -- a ~12x
reduction from fixing destinations alone, with nothing else touched.
Clipping is mostly a DOWNSTREAM SYMPTOM of destination error (wrong
destinations -> wrong local inventory state -> N, calibrated to the real
trajectory, mismatches worse), not an independent conservation break.

`sanity_true_dest` also clears the trip-count/volume gates outright: total
trips 3.44% (gate <=5%), per-zone WMAPE 3.44% (gate <=15%), continuous
inventory-level correlation **0.901** with a **1-bike median absolute
error**. That's proof the mechanics -- routing, capacity constraints,
reroute search, N replay -- are fundamentally sound given true trip-level
ground truth. `sanity_od` and `stochastic`, with OD-share-drawn
destinations, land at 7.8% total-trips error and continuous-inventory
correlation of only 0.48-0.57 (5-6 bike median error) -- a genuinely
degraded trajectory relative to `sanity_true_dest`, not a threshold
artifact.

**Why the per-interval stockout-timing gate cannot be fixed by more OD
conditioning.** Destination is the other half of a realized trip.
`od_shares.py`'s backoff hierarchy samples a destination from HISTORICAL
MARGINALS conditioned on origin and time -- it gets aggregate flow right
(that's what a marginal distribution is) and the REALIZED PAIRING wrong,
because the true joint distribution of (origin, destination, time) can't
be recovered from marginals alone without either the real paired data
(which `sanity_true_dest` has, and a genuine forward simulation never
will) or conditioning on enough context to reconstruct the pairing --
which reintroduces the sparsity problem the backoff hierarchy exists to
solve (see the Phase 7 plan-mode discussion on member/casual conditioning
specifically:
`~/.claude/plans/read-spec-md-4-forward-wise-meadow.md`).
Inventory is a PATH-DEPENDENT ACCUMULATION of realized pairings -- every
wrong destination this step changes next step's starting state,
compounding across 672 steps. More conditioning trades a little of that
per-trip pairing error for materially worse sparsity (thinner cells, more
fallback to zone/global tiers -- MORE marginal-distribution error, not
less). Not a net win, and not attempted.

**Resolution: the gate is restated to match what Phase 8 actually
consumes, not abandoned.** Phase 8's MV(s,t) needs stockout FREQUENCY and
recoverable volume at a station-hour-of-week granularity (SPEC.md §7), not
exact 15-minute timing. The simulator validates on:
- trip totals and per-zone volume (`sanity_true_dest`: 3.44% / 3.44%, both
  under gate)
- continuous inventory trajectory (`sanity_true_dest`: 0.901 corr, 1-bike
  median error)
- stockout rate by station-hour-of-week -- see the caveat below, honestly
  reported rather than assumed to work

It does NOT reproduce per-interval stockout timing under stochastic
destination sampling, and per the argument above, structurally cannot.

**Follow-up (same day): the single-week caveat above was retested across
multiple weeks, and the assumption is FALSIFIED, not merely untested.**
The theoretical argument (rate-based metrics should be more robust to
destination-scrambling than exact timing, given enough aggregation) implies
correlation should rise as more weeks are pooled into each (station,
hour-of-week) cell. It doesn't. `src/sim/validate.py --multiweek` ran the
`stochastic` simulator over 6 held-out weeks (2025-10-06 through
2025-11-10, same seed each time -- only the underlying week's real data
changes) and correlated pooled simulated-vs-actual stockout rate per
(station, hour-of-week) at n_weeks = 1, 2, 4, 6:

| n_weeks | n_cells | corr | WMAPE | mean\|diff\| |
|---|---|---|---|---|
| 1 | 367,190 | 0.043 | 218% | 0.075 |
| 2 | 368,185 | 0.079 | 202% | 0.071 |
| 4 | 369,594 | 0.100 | 195% | 0.070 |
| 6 | 371,419 | 0.097 | 200% | 0.070 |

Correlation plateaus around 0.10 and TICKS DOWN from 4 to 6 weeks, not up.
Mean absolute error barely moves (0.075 -> 0.070). More data confirms the
low correlation rather than resolving it -- this is not a sparsity
artifact of the single-week test; it reproduces under 6x the pooling.

**Consequence, stated plainly because it's bigger than an open risk: the
simulator cannot supply P(stockout | s, t) at station-hour resolution, in
any run mode, at any amount of week-pooling tried.** An MV(s,t) built by
perturbing simulated inventory and re-running the network simulator would
be ranking station-hours by simulator noise, not by real differences in
stockout risk -- and any Phase 9 lift number computed that way would be
measuring the optimizer learning to exploit its own simulator's
destination-assignment noise, not a real intervention effect.

**Resulting decision: Phase 8's MV(s,t) will be derived EMPIRICALLY from
Phase 4-6 outputs, not by simulation.** `demand.py`'s censored demand
model already produces, per (station, hour-of-week), a calibrated
gross-unmet-demand estimate (validated to a bias of about +-0.02 under
matched-sampling recovery testing -- see `calibrate_direction_matched`),
and `substitution.py` already nets that down to `dep_net_lost` per
interval. Averaged across the FULL YEAR (~52 weeks, not 6 simulated ones),
that's up to ~208 real sub-observations per (station, hour-of-week) cell
-- the exact statistical power problem that just failed above, solved by
using the real historical record directly instead of simulated
destination-shuffled weeks layered on top of it. The genuinely open part
is the CONCAVITY curve (marginal value of the 1st, 2nd, ... nth bike, not
just "some bike"): proposed as a birth-death/queueing approximation using
`demand.py`'s already-fitted departure AND arrival rate estimates (both
directions are already modeled) plus Phase 4's inferred N as the
birth/death rates, giving a closed-form P(stockout | bikes = k) for every
k and hence the full MV(s,t,k) curve without simulating the network at
all -- cross-checked against the empirical marginal effect at station-hours
where real cross-week starting-inventory variation exists, rather than
trusted blindly. Elaborated when Phase 8 is actually built; filed here so
the reasoning for NOT using the simulator for this step is on record
before that code exists.

**Mode usage going forward, stated explicitly so it isn't relitigated
later.** `sanity_true_dest` is a MECHANICS-VALIDATION mode only -- it
requires real trip-level ground truth that doesn't exist for a genuine
forward simulation, so it is never run going forward. `stochastic` is
what Phase 9 (policy comparison) runs -- but ONLY for system-level and
per-zone aggregate lift and continuous inventory trajectories under a
given policy, per the resolution above; NOT for station-level fill-rate or
stockout numbers, which the multiweek result above confirms are simulator
noise at that resolution. Phase 9 compares stochastic runs under different
induced-move policies using the SAME seed/sampling noise, so destination-
assignment error is present in both the baseline and treatment run and is
EXPECTED to largely cancel in the reported difference (lift) rather than
in either run's absolute numbers. **That cancellation is an ASSUMPTION
carried forward from this validation, not proven here** -- it should be
stated as such wherever Phase 9's lift numbers are reported, not treated
as already established. Phase 8's MV(s,t) does NOT run the simulator at
all (see above), so this caveat applies to Phase 9's policy-comparison
step specifically, not to the allocator's ranking decisions.

---

## Phase 6 (2026-08-12): volume vs. per-dock ranking -- and per-dock's own failure mode

The Phase 6 heatmap (`src/viz/heatmap.py`) originally ranked zones for both
the top-10 tables and the two-panel figure by raw weekly net-lost volume.
That's wrong as the PRIMARY metric: raw volume is dominated by zone size
(more stations = more absolute trips = higher rank almost by construction),
so it structurally buries genuine per-capita scarcity at small residential
zones behind a handful of huge Midtown office-district zones. Concretely:
Upper West Side zones (agglom_465: Riverside Dr & W72/Amsterdam Ave;
agglom_503: Central Park W & W68-72 St) showed the textbook residential
half of the commuter dipole -- bike-starved 8-9am, matching SPEC.md §5's
predicted pattern almost exactly -- but never appeared in the raw top-10
because their absolute volume is a fraction of Midtown East's.

**Fix:** `top_zones()` is now called with the per-dock columns
(`dep_net_lost_per_dock`/`arr_net_lost_per_dock`) as the PRIMARY ranking for
both the console top-10 and the main two-panel figure (unsuffixed
`net_lost_heatmap.png`); the raw-volume view is kept as a secondary
reference (`net_lost_heatmap_by_raw_volume.png`), not deleted, since it's
still the right lens for "where does an absolute dollar of budget move the
most trips" questions later in the project (§7's marginal-value work).

**This is not a strictly-better fix, and the flip side showed up
immediately on the arrivals side.** Per-dock ranking is exactly as
vulnerable to small denominators as raw volume is to small numerators: two
tiny Long Island City/Maspeth Queens zones (agglom_296: 61-63 St & Borden
Ave, 4 stations/76 total docks; agglom_464: 53 Ave & 62 St/65 Pl, 3
stations/57 docks) took the #1 and #2 dock-starved-per-dock slots with
peaks at Sat 03:00-07:00 and Fri 00:00-04:00 -- weekend overnight hours
that match no commute pattern SPEC.md describes. Meanwhile agglom_19
(Midtown West, 7th/8th Ave & W55, 10 stations/666 docks), which shows a
clean, real, commute-timed AM dock-starved signal, dropped to rank **50**
by per-dock and is effectively invisible in a top-10 view. Checked whether
this was a capacity-staleness artifact (arrival-side censoring is ~2.5x
worse at stale stations, see the Phase 5 calibration note) -- it isn't;
zero stale stations in either zone.

**Superseded, same day: replaced by lower-bound-of-CI ranking, not left as
an open caveat.** A dock-count or station-count floor was rejected as
arbitrary (see above) in favor of `aggregate_zone_hour_per_dock_ci()`:
for each (zone, hour_of_week), compute the per-dock rate SEPARATELY for
each of the ~52 individual calendar weeks in the panel, then rank on
`mean - 1.96 * SE` (standard error of the mean across those weekly
observations) instead of the mean itself. Zone size never enters the
formula -- the penalty comes entirely from week-to-week inconsistency,
which is orthogonal to zone size by construction.

**Result, checked against real numbers, not assumed:** on the departures
(bike-starved) side this worked exactly as hoped. Midtown East
(agglom_498/agglom_59, tight SE ~14% of the mean) and the Upper West Side
zones (agglom_465, agglom_503) all hold their ranks with comfortably
positive lower bounds -- a real, weekly-consistent signal survives the
penalty untouched.

**On the arrivals (dock-starved) side, the result was NOT what the
per-dock fix's own reasoning predicted, and that's worth stating plainly
rather than declaring victory.** agglom_296 and agglom_464 (the same two
Queens zones) still take the top slots. Pulled their full weekly time
series to check why (`agglom_296`, Sat 06:00): nonzero in 40 of 52 weeks,
mean 1.46, values clustered mostly in the 0.9-2.7 range -- this is NOT "a
couple of noisy spikes," it's a real, statistically reliable, weekly-
recurring elevated rate (SE only ~10% of the mean). The CI method did
exactly what it's supposed to do: it doesn't favor commute-shaped patterns,
it favors STATISTICALLY RELIABLE ones, and this Saturday-morning Long
Island City/Maspeth pattern (plausibly weekend/nightlife-driven arrival
pressure on a very small, 57-76-dock zone) genuinely is reliable -- just
not a commuter dipole. Meanwhile agglom_19 (Midtown West) has an equally
tight interval (SE ~23% of its mean) but its per-dock RATE is genuinely,
reliably lower -- 666 total docks dilutes even a large absolute trip
volume into a small per-dock number -- so it stays buried (still rank
~1,073 of ~2,100 by arrivals lower bound), not because of noise wrongly
winning, but because large zones are structurally lower per-dock even at
full statistical confidence. The lower-bound fix resolved the noise
problem it was built to resolve; it did not, and structurally cannot,
make a large zone's diluted per-dock rate outrank a small zone's
concentrated one.

**This tension is not a bug to fix at the ranking layer, and not left open
either -- it's dissolved by recognizing rate and decision-value are
different quantities, not by tuning the ranking further.** Per-dock rate
(however confidence-adjusted) answers "how scarce is this zone relative to
its own size" -- a diagnostic question. It does NOT answer "how many trips
would an extra bike here actually save," which is what an allocation
decision needs, and there is no normalization of a rate metric that makes
it answer that question, because zone size is exactly the information a
rate discards and an allocation decision needs back.

**Scope decision:** the Phase 6 heatmap (both panels, per-dock lower bound
primary / raw volume secondary) is DIAGNOSTIC ONLY -- it shows where and
when scarcity occurs, nothing more. Ranking for allocation is deferred to
Phase 8's marginal value, MV(s,t) = P(stockout) x E[unmet trips | stockout]
(SPEC.md §7), which is denominated directly in expected trips saved per
bike added -- not a per-dock or per-zone-total rate at all. This is why
Phase 8 dissolves the tension rather than inheriting it: MV picks up
stockout FREQUENCY and RECOVERABLE VOLUME simultaneously and multiplies
them, so a small zone's frequent-but-thin stockouts and a large zone's
rare-but-deep ones are both priced in trips, on the same footing, with
neither small-zone rate inflation (agglom_296's reliable-but-small effect)
nor large-zone rate dilution (agglom_19's real-but-diluted effect) able to
bias the comparison the way they bias any per-dock ranking. Phase 6 stays
useful specifically because it's honest about only answering the
diagnostic question -- Phase 8 is where the allocation question actually
gets asked.

**Worth one line in the writeup, filed here for when Phase 10 gets there:**
the Saturday-morning Long Island City/Maspeth dock-starved pattern
(agglom_296/agglom_464, verified reliable across 40+ of 52 weeks, not a
sampling artifact) is a genuine finding independent of the ranking-layer
question above -- plausibly weekend/nightlife-driven arrival pressure on a
very small-capacity zone -- and belongs in the writeup as a real secondary
pattern the commuter-dipole framing doesn't cover, not as a bug that got
filtered out.

## Phase 6 (2026-08-12): Jersey City/Hoboken excluded from zone rankings -- geography-based flag, not a name or station-id heuristic

Citi Bike's Jersey City/Hoboken deployment is a physically separate
sub-fleet (own PATH/light-rail-anchored network across the Hudson, not
contiguous with the Manhattan/Brooklyn/Queens system this project's SPEC.md
scopes to). It showed up as a real problem, not a hypothetical one: before
excluding it, 8 of the top-10 raw dock-starved zone-hours were Jersey City
zones (agglom_206: Exchange Pl/Grand St/Essex Light Rail) showing a
near-constant elevated dock-starved band across ALL seven days and every
hour in the heatmap PNG -- visually and structurally nothing like the
commute-peaked pattern the dipole check is looking for, and different
enough that averaging it into the same ranking as Manhattan zones muddies
both the top-10 list and the commuter-dipole sanity check.

**How zones were flagged, and why not the obvious-looking shortcuts.**
Station name substring matching ("JC", "Hoboken", "PATH") was tried first
and rejected: it produces false negatives (Hoboken's "12 St & Sinatra Dr
N", "Dixon Mills" name nothing NJ-specific) and at least one confirmed
false positive risk (a station literally named "Dey St" turned out to be
Jersey City's Dey Street near Journal Square -- coordinates (40.7377,
-74.0669) -- not Manhattan's FiDi Dey Street near the WTC; the name alone
is genuinely ambiguous between the two cities and would have been guessed
wrong). Station-ID prefix (the old system used explicit `JC*`/`HB*`
prefixes, visible in the null-capacity-dropped station list from Phase 1)
was rejected too -- newer stations use the same numeric `NNNN.NN` ID
scheme regardless of city, so the prefix rule silently stops working for
part of the fleet.

**What was used instead: a geographic bounding box on each zone's
centroid, calibrated against real coordinates, not guessed.** Manhattan's
westernmost stations (Battery Park City: e.g. "South End Ave & Albany St")
reach lng -74.017. Hoboken's easternmost (12 St & Sinatra Dr N) sit at lng
-74.024 -- the Hudson River itself creates that ~0.007-degree gap, and
`JC_HOBOKEN_LNG_MAX = -74.02` sits in the middle of it. A pure longitude
cutoff is NOT sufficient on its own, though -- Brooklyn's Bay Ridge reaches
a nearly identical longitude (-74.024, e.g. "4 Ave & 72 St") but at lat
40.62-40.65, so `JC_HOBOKEN_LAT_MIN = 40.68` was added specifically to
exclude it; checked directly against the full station list (`is_jc_hoboken
_zone()` in `src/viz/heatmap.py`), not assumed from a map by eye. Running
this rule over all 544 zones found 39 JC/Hoboken zones, not the 7-8 that
were visually obvious from the heatmap's top rows alone -- most of interior
Jersey City (Journal Square, Bergen-Lafayette, Greenville, West Side) has
low enough volume that it never would have been caught by eyeballing the
chronic-pattern rows, only by the systematic geographic pass.

**Scope note:** the exclusion applies to the zone-hour heatmap panels and
top-10 tables only (`src/viz/heatmap.py`). `plot_station_scatter`'s
per-station map still includes JC/Hoboken stations, since it isn't
zone-ranked and wasn't part of what was asked; revisit if the writeup wants
one consistent NYC-only scope everywhere.

## Phase 4 (2026-08-07): the overcount direction itself was the real signal -- renamed R to N, "non-trip bike movement," not rebalancing

The entry below this one fixed part of the DOT overcount (capacity==0
stations) and left a residual gap open (the 329-station stale-capacity
pool). Even after that fix, every real month still read HIGH against DOT --
1.06x to 1.79x, median around 1.4x. That residual is the subject of this
entry, and it's a different kind of finding than a data bug: it's the
estimator correctly measuring something the label "rebalancing" doesn't
cover.

**The argument that made this suspicious rather than just "close enough."**
SPEC.md's identifiability argument, restated in this module's own
docstring: a minimal-|correction| method can only ever *underestimate* true
rebalancing, because it credits the smallest possible correction consistent
with the bounds, and real operators rebalance preemptively and sometimes
redundantly (a truck tops off a station before it strictly needs to, then
organic demand partially undoes that -- invisible to a minimal-correction
estimator). That argument predicts ratio <= 1. Getting a ratio consistently
*above* 1, after already fixing the one clear data bug that was inflating
it, isn't "roughly validated" -- it's the sign of the error pointing the
wrong way, which means something is being counted that shouldn't be, not
just that a real quantity is being estimated a bit high.

**Diagnosis:** the flow-balance equation this module solves,
`I(t+1) = I(t) + arrivals(t) - departures(t) + [correction]`, cannot
distinguish *why* a bike appeared or disappeared without a matching trip --
only *that* it did. Operator rebalancing (trucks, vans, valets, Bike
Angels -- what DOT's monthly figure specifically counts, per
`src/ingest/dot_reports.py`) produces that signature. So does everything
else that moves a bike without a rider completing a trip: maintenance
pulls, broken-bike removal, and e-bike battery swaps. Citi Bike's own
monthly reports describe exactly these categories running alongside
rebalancing every month -- "17,956 bicycle inspections and repairs,"
"15,869 total unique bikes checked or repaired," a monthly net fleet
loss/gain from bikes permanently removed -- none of which are folded into
the "Citi Bike staff rebalanced a total of N bicycles" sentence the DOT
cross-check is built on. The model was never measuring DOT's category. It
was measuring the union of all of them, because that union is all the flow-
balance equation can see.

**Resolution: renamed the quantity throughout, not just relabeled the
column.** R(s,t) -> N(s,t), "operator rebalancing" -> "non-trip bike
movement," in the module docstring, the LP's internal variable names
(`R_in`/`R_out` -> `N_in`/`N_out`), every intermediate dict key
(`rebalance_in`/`rebalance_out` -> `nontrip_in`/`nontrip_out`), the output
parquet columns (`inferred_rebalance_in`/`_out` ->
`inferred_nontrip_in`/`_out`), `config/params.yaml`'s comments, and the
tests. `validate_against_dot()` itself changed shape, not just its print
strings: it now documents and checks a superset relationship (inferred
non-trip movement >= DOT's rebalancing-only figure, i.e. ratio >= 1 is the
*expected* result) instead of treating any deviation from ratio ~= 1 as
unexplained error. The 1.06-1.79x range is now a plausible reading of that
relationship, not a residual to keep chasing.

**Why this doesn't weaken Phase 5, and is in fact the quantity Phase 5
actually needs:** SPEC.md's censoring model cares about *station
availability* -- whether a bike or dock was there when a rider wanted it.
A bike pulled for a battery swap is exactly as unavailable to a would-be
renter as one pulled by a rebalancing truck. Renaming R to N doesn't
narrow what the reconstruction is useful for; it corrects what it was
claiming to be. The overclaim was in the label, not the number.

**Caught by:** taking the *sign* of the DOT discrepancy seriously, not
just its order of magnitude. "Within 2x of DOT, same ballpark" would have
been accepted as a reasonable validation result by most reasonable-
sounding standards -- SPEC.md itself says as much ("If you're within the
right order of magnitude, say so and move on"). But SPEC's own
identifiability argument makes a directional prediction, not just a
magnitude one, and checking that direction against the actual result is
what surfaced this. A validation number can look fine and still be
answering a different question than the one it claims to.

---

## Phase 4 (2026-08-06): DOT cross-check ran 1.2-1.9x HIGH, not low -- traced to stale GBFS capacity, only partly fixed

Validation #2 (SPEC.md §4: inferred total |R| vs. DOT's monthly operating
reports) came back the wrong direction. SPEC's own identifiability argument
says minimal-|R| reconstruction is a *lower bound* on true rebalancing --
the ratio should sit at or below 1.0. Instead every real month came back
**1.2x to 1.9x higher** than DOT's figure (e.g. Sep 2025: inferred 190,894
vs. DOT 101,039, ratio 1.89).

First check: was this an LP-specific artifact of the anchor-penalty fix
(the entry below)? No -- reran the same panel through the greedy clip
method as a diagnostic. Greedy's monthly totals track the LP's almost
exactly (Sep 2025: greedy 186,164 vs. LP 190,894). Whatever's driving the
overcount is upstream of both methods, in the data they're both fed, not
in either reconstruction algorithm.

Second check: is the excess spread evenly across the network (a broad
systematic bias) or concentrated (a data-quality problem in a subset of
stations)? Concentrated, heavily: the top 200 of 2,270 usable stations
account for 68.3% of total inferred system-wide R; the top 15 alone
account for 29.8%. That shape -- a handful of extreme outliers, not a
uniform shift -- points at bad inputs for specific stations, not a wrong
constant somewhere in the model.

**Confirmed root cause, at least in part:** capacity is joined from a
*single, present-day* GBFS `station_information` snapshot
(`src/ingest/gbfs.py`), applied retroactively across the entire
2024-12-31 to 2025-12-31 panel. Three stations report `capacity == 0` in
that snapshot despite clear, heavy real activity during the panel window
-- one of them, `5788.13`, logged 146,418 departures and 146,974 arrivals
over the year, among the highest-volume stations in the system. A capacity
of 0 makes every single trip at that station a "bounds violation" by
construction, forcing the LP (or greedy) to invent a rebalancing event on
nearly every active interval. These 3 stations alone accounted for **7.4%
of total inferred system-wide R** before being filtered. Fixed the same
way the 158 null-capacity stations were already handled in
`src/models/inventory.py`'s `prepare_panel()`: dropped, not imputed,
logged with the same station-count / row-count / trip-volume-share
accounting `gbfs.py` uses for its own capacity-match reconciliation, not a
silent row drop.

**Open question, not resolved by that fix:** 329 stations (14.5% of the
usable network) have capacity below their own observed single-15-minute-
interval peak throughput (departures + arrivals in one interval),
collectively associated with 44.0% of total inferred R. This is *weaker*
evidence than the capacity==0 cases -- a station's dock count is not a
throughput ceiling, since a dock frees the instant a bike leaves and can
immediately take an arrival, so legitimately high-turnover stations can
process more trips in 15 minutes than their capacity without anything
being wrong. It's a plausible pool of *additional* stale-capacity cases,
not a confirmed one, and it has not been investigated station-by-station.

**Why this is being left open rather than "fixed":** GBFS's
`station_information` feed only ever exposes *current* capacity -- there is
no field, no history endpoint, and (per the earlier historical-GBFS search,
see above) no third-party archive covering this panel's window, that gives
capacity *as of* a date a year in the past. If a station's true dock count
changed between whenever-it-was-in-2025 and today's snapshot, that error is
structurally unrecoverable from GBFS alone -- not a gap left because it
wasn't prioritized. Chasing it further would mean either (a) accepting the
329-station list at face value and dropping/reweighting them without real
evidence most of them are actually wrong, which risks quietly discarding
legitimate high-turnover stations along with genuinely bad ones, or (b)
finding an independent historical capacity source, which doesn't appear to
exist. Left as a stated limitation on the DOT cross-check's remaining gap,
not swept into the reconstruction as more filtering.

**Post-fix numbers, for the record:** dropping the 3 capacity==0 stations
(and the 158 null-capacity ones, unchanged from Phase 2/3) narrowed the
ratio but did not close it. Monthly inferred-vs-DOT ratios across the 12
full-coverage months now range 1.06 (Jan 2025) to 1.79 (Sep 2025), median
around 1.4 -- still consistently above 1.0, i.e. still the "too high"
direction SPEC's identifiability argument doesn't predict, and still
plausibly attributable in part to the unresolved 329-station pool above.
Bounds validation (#1) passed clean on the corrected run: 75,760,271
station-intervals checked, zero violations.

---

## Phase 2 (2026-08-05): DBSCAN zone clustering chained transitively into one 1,152-station mega-zone

SPEC.md §2 calls for clustering stations into "substitution neighborhoods"
-- the set of stations a rider would actually walk between -- and suggests
DBSCAN on projected coordinates as one option. First attempt used
`DBSCAN(eps=350)`, matching SPEC's "~300-400m walking distance" language
directly as the `eps` parameter.

This is the wrong parameter to control for the goal. `eps` bounds the
distance between a point and its *neighbor* for density-reachability, not
the diameter of the resulting cluster. On a network as dense as Citi Bike's
Manhattan/Brooklyn/Queens core, station spacing is frequently well under
350m, so DBSCAN's density-reachability chains transitively through the
entire dense region: station A is within 350m of B, B within 350m of C,
and so on, with no point in that chain checking whether A and C (or A and
the 1,000th station in the chain) are anywhere near each other. The
resulting "cluster" boundary is wherever the chain of sub-350m gaps
happens to break, which has nothing to do with walkability.

Verified directly (not just observed and patched around) by rerunning
`DBSCAN(eps=350, min_samples=2)` on the actual 2,463-station GBFS
`station_information` table: 39 clusters excluding noise, the largest
containing **1,152 stations** (47% of the entire network) with an
approximate diameter of **17.5 km**, spanning roughly lat 40.6355-40.7860
/ lng -74.0170--73.8323 -- central Brooklyn, up through Manhattan, into
western Queens. One "neighborhood."

Fix: switched to `AgglomerativeClustering(linkage="complete",
distance_threshold=700)` on the same projected coordinates
(`src/features/zones.py`). Complete linkage merges two clusters by the
*maximum* pairwise distance between any two points across them, so
`distance_threshold` is a direct, structural upper bound on cluster
diameter -- no cluster can exceed 700m end-to-end no matter how dense the
surrounding network is, by construction rather than by tuning. (700m
diameter ≈ same walkable span as SPEC's ~350m-radius framing, just stated
as a diameter bound instead of a neighbor-spacing bound.)

**Lesson:** in density-based clustering, `eps` bounds a *local* gap, not a
*global* extent -- it will chain arbitrarily far through any sufficiently
dense region, with cluster size an emergent, unbounded consequence rather
than a parameter you control. For a "walkable neighborhood" definition
specifically, the method's parameter needs to be a direct bound on cluster
diameter or radius, not on neighbor spacing.

---

## Phase 4 planning (2026-08-06): historical GBFS search came up empty

Before committing to the inventory-reconstruction LP, checked whether any
third-party historical GBFS archive covers the panel window (2024-12-31 to
2025-12-31) -- if so, it would restore SPEC.md §4's validation #3
(reconstructed inventory vs. observed ground truth) instead of leaving only
the DOT monthly cross-check. Two leads, both dead ends:

- **github.com/NYCComptroller/citi-bike-gbfs** -- exists, but is a
  fork-and-run-yourself *template*, not a live archive. Confirmed via the
  GitHub API directly, not inferred from the README: the org's own `main`
  branch has zero GitHub Actions runs ever (`total_count: 0`) and no
  `/data` directory (404) -- nobody, including the Comptroller's office,
  ever actually ran the collector on the canonical repo. The only real data
  lives on a `sample-data` branch: 1,000 raw GBFS JSON snapshot files
  (`station_status` + `station_information` pairs), Unix timestamps
  1710282230-1710845217, i.e. **2024-03-12 to 2024-03-19** -- one week,
  about 9.5 months before the panel starts. Zero overlap.

- **macwright.com bikeshare archive** -- a personal project, 5-minute
  polling into a **private** Cloudflare R2 bucket the author owns, running
  since July 2023. Read both of his posts on it (2023-09-17, 2024-10-14)
  directly rather than trusting a search snippet: neither mentions a public
  bucket URL, a dataset export, or any access path at all. The demo he
  links (bikesharecharts.com) is dead -- connection refused, not a 404.
  Nothing publicly downloadable exists here regardless of date coverage.

**Conclusion:** neither lead restores validation #3. DOT's monthly
operating reports are the only external check available for Phase 4, and
even those only give one system-wide monthly scalar with no station-level
or time-of-day resolution (see `src/ingest/dot_reports.py`'s module
docstring for what those reports actually contain, confirmed by reading
the PDFs rather than assumed from their name).

---

## Phase 4 (2026-08-06): LP anchor-penalty bug -- silently absorbed real rebalancing into anchor drift, reported zero R

The per-station-week LP (`src/models/inventory.py`) has two mathematically
available ways to explain any bounds violation within a week:

1. Credit real rebalancing: increase `R_in[t]` or `R_out[t]` at the
   objective's stated cost of **1.0 per bike**.
2. Redefine the week's assumed starting inventory: move the anchor
   variable `I[0]` away from its prior, at the objective's stated cost of
   `anchor_prior_weight` **per unit of deviation** -- originally set to
   0.1, deliberately small, on the theory (from the Phase 4 plan-mode
   discussion) that the anchor should only matter as a tie-breaker in
   weeks with genuine slack.

That reasoning was wrong, and a regression test caught it before the LP
ever ran on real data. Concrete case
(`tests/test_inventory.py::test_recovers_injected_rebalance_out`): anchor
prior = 5, capacity = 20, a single interval with `net_flow = -8`. The
physically intended read is "station had 5 bikes, lost 8, something must
have added 3 mid-interval" -- `R_in = 3`, cost `1 × 3 = 3.0`. What the
solver actually returned: `I[0] = 8` instead of 5 (a deviation of 3 from
the prior, cost `0.1 × 3 = 0.3`), and `R_in = R_out = 0` for the entire
week. Both are equally valid solutions to the same linear system and the
same bounds -- the LP is a cost minimizer, `0.3 < 3.0`, so it rationally
picked the cheaper one. The solver did its job correctly; the objective
handed to it was wrong.

**Why this would have failed silently on real data.** Bounds were still
satisfied everywhere (`I[t] ∈ [0, capacity]` throughout -- validation #1
would pass, and did, on the real 2,270-station run). The LP still reports
optimal/converged, no infeasibility, no warning. Nothing about output
shape, dtype, or the bounds check looks wrong. The only symptom is an
inferred total `|R|` that reads too low against the DOT monthly cross-check
(validation #2) -- and because that check is a single order-of-magnitude
scalar per month with no station-level ground truth to localize a
discrepancy to, a moderate systematic underestimate could plausibly have
been misread as exactly the identifiability floor SPEC.md already predicts
("minimal-|R| is a lower bound on true rebalancing activity"), rather than
diagnosed as a separate, fixable bug sitting on top of that floor. It would
have been very easy to accept a wrong number as an expected limitation.

**The fix is structural, not a tuned hyperparameter, and that distinction
matters here specifically because there is no ground truth in this project
to tune against** (panel: 2024-12-31 to 2025-12-31; GBFS snapshots start
2026-08-04 -- zero overlap, see the entry above). `anchor_prior_weight`
must be strictly greater than 1.0 -- R's per-unit cost -- full stop. Any
weight above that threshold makes crediting R at least as cheap as shifting
the anchor for any real violation, so the prior can only ever act as a
tie-breaker in station-weeks with zero violations under any feasible
anchor -- i.e. exactly the weeks where `R=0` and `anchor=prior` are already
the jointly cheapest answer, with nothing left to trade off. Set to 2.0 in
`config/params.yaml`: comfortably past the threshold, not fit to any
output.

**Caught by:** a synthetic recovery test written because SPEC.md and the
RUNBOOK both call for one ("synthetic station where I inject a known
rebalance and confirm recovery") -- not by inspecting real output. The full
2,270-station run was never eyeballed station-by-station before this test
failed. The bug was fixed before it touched real data, which is the entire
point of writing that test first.

---

## Phase 9 (2026-08-31): the "underpowered" bootstrap result was a measurement-design problem, not a power problem

The 40-replicate main bootstrap (`reports/policy_comparison.md`'s system-level
table) reported every policy's fill-rate lift CI straddling zero -- 0/5
policies significant -- and the writeup up to this point treated that as a
real power limitation: "the demand-residual and elasticity axes contribute
genuinely large week-to-week variance that a bigger N would narrow." That
framing was never checked against the alternative explanation before being
written down.

**The alternative, raised directly rather than discovered internally:** every
policy funds at most a few thousand induced trips (SPEC.md's own candidate
pool is 2,998 cells) against a system that moves ~875K trips/week. That's a
>99.5%-of-the-denominator dilution of any real effect BEFORE bootstrap noise
enters the picture -- a signal-to-noise ceiling set by the measurement's own
denominator, which no amount of additional replicates can raise. "Underpowered"
and "diluted past detectability by design" produce the identical symptom (CI
straddles zero) but have opposite fixes -- one needs more replicates, the
other needs a different denominator.

**Test:** restrict fill rate to just the (station, hour-of-week) cells a
policy itself funded (post `apply_move_cap`), paired against a do-nothing run
on the SAME cells and the SAME replicate seed -- same 40 replicates, same
seeds as the system-level table, just a narrower, paired denominator
(`src/sim/policy_compare.py`'s `run_one_replicate_treated` /
`compute_treated_cell_fill`, `reports/plot_policy_comparison.py`'s
`build_treated_comparison`). Required re-running the simulator -- the raw
per-station `station_intervals` were never checkpointed, only the
system/zone-aggregated fill table (Phase 7's per-station output ban, see
below) -- so this is a ~10-hour, 2-worker background job, not a re-query.

**Result: 4/5 policies now show a fill-rate lift CI that excludes zero**
(`uniform` +2.13pp [+0.85,+3.19], `proportional` +1.96pp [+0.68,+3.10],
`top_n_stockout` +2.56pp [+0.48,+4.03], `allocator_full_budget` +2.06pp
[+0.55,+3.08] -- only the natural-spend `allocator` at $377 stays
inconclusive, +0.56pp [-0.63,+1.94], P(lift>0)=72%, consistent with it simply
buying far less treatment than the other four's ~$10,000). Confirms the
dilution explanation, not the power explanation: the same 40 draws that
looked like pure noise at system level resolve into a clear, consistent
signal once the denominator matches the intervention's actual footprint.
`reports/policy_comparison.md`'s "Treated-cell paired comparison" section
carries the full table; the system-level table above it is left as-is, not
deleted -- it's still the honest answer to "what does this do to the whole
network," it was just never a fair test of "does the treatment work."

**A second, independent bug surfaced building this: `trips_recovered` must be
defined as the reduction in LOST trips at the treated cells, not the change
in arrivals.** An incentive move adds bikes at its destination, which fixes
that cell's `lost_no_bike` (departures that used to fail there for lack of a
bike) -- it does NOT directly change `direct_arrivals`/`rerouted_arrivals`
at that same cell, which are driven by other stations' routing and are
largely exogenous to this cell's own bike count. A first draft defined
`trips_recovered` as the arrivals-side change (matching the system-level
table's definition, which is correct AT system level, where arrivals ARE the
right proxy for total completed trips) and got a median NEGATIVE
trips_recovered for every policy despite a clearly positive, significant
fill-rate lift -- the tell that the metric, not the result, was wrong.
Fixed by tracking the LOST-side reduction directly
(`lost_treated_do_nothing - lost_treated`); all 40/40 replicates then
recovered positive trips for 4 of 5 policies, matching the lift-pp sign.
Caught before writing the number down, not after -- checked why the sign
looked wrong instead of reporting it.

**Does this reopen the Phase 7 per-station output ban (CLAUDE.md's "no
per-station fill-rate breakdown, ever")? No, and the distinction is
load-bearing enough to restate here.** Phase 7 found simulated
per-(station, hour-of-week) stockout timing doesn't correlate with REAL
ground truth at that grain (pooled corr plateaus ~0.10) -- a claim about
matching reality. This comparison never claims to match reality at station
resolution; it only compares two runs of the SAME simulator, same seed,
against each other, and every number reported is POOLED over hundreds to
thousands of cells per policy -- never a single station's fill rate.
`compute_treated_cell_fill` has no code path that can emit one.
