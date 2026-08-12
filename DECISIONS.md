# Decisions

Real design forks hit while building this, and the actual reasoning behind
how they were resolved -- not a changelog summary. Written to be reread
before an interview, when "why did I do it that way" needs a real answer.

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
