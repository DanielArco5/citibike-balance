# Decisions

Real design forks hit while building this, and the actual reasoning behind
how they were resolved -- not a changelog summary. Written to be reread
before an interview, when "why did I do it that way" needs a real answer.

---

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
