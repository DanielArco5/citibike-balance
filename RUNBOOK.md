# Runbook — Citi Bike Incentive Allocation

Companion to `SPEC.md`. Work through phases in order. **One Claude Code session per phase.** `/clear` and commit between phases — context bleed between phases is the main cause of drift.

Prompts are copy-paste. Bracketed `[...]` bits are yours to fill.

---

## Day 0 — Setup (30 min, mostly you not Claude)

```bash
mkdir citibike-balance && cd citibike-balance
git init
uv venv && source .venv/bin/activate      # or python -m venv
uv pip install polars duckdb pyarrow lightgbm scikit-learn statsmodels \
               matplotlib seaborn folium h3 pulp requests pyyaml pytest
mkdir -p data/{raw,interim,processed} src/{ingest,features,models,sim,opt,viz} tests notebooks reports config
printf 'data/\n.venv/\n__pycache__/\n*.parquet\n' > .gitignore
```

Drop `SPEC.md` in the root. Then create `CLAUDE.md` from §10 of the spec.

**Start the GBFS collector today** — every day you delay is a day of inventory truth you never get back. Even if you don't use it until Phase 4:

```bash
# save as scripts/gbfs_poll.sh, then: crontab -e  →  * * * * * /path/to/gbfs_poll.sh
curl -s https://gbfs.citibikenyc.com/gbfs/en/station_status.json \
  > "$HOME/citibike-balance/data/raw/gbfs/status_$(date +%Y%m%d_%H%M).json"
```

Download 12 months of trip zips from the S3 index linked off `citibikenyc.com/system-data` into `data/raw/trips/`. Several GB. Do it now, in the background.

Then `git commit -m "scaffold"`.

---

## Phase 0 — Schema discovery

**No modeling code. None.** This phase exists solely to stop Claude building a beautiful pipeline on hallucinated column names.

```
Read SPEC.md, then stop and do only this:

data/raw/trips/ has monthly Citi Bike zips spanning [YEAR RANGE]. For each
distinct schema era you find, unzip ONE file, read the first 200 rows, and
report:
- exact column names and dtypes, verbatim
- 3 sample rows
- station_id format (they changed at some point — show me actual values)
- timestamp format and whether tz-aware
- null rates per column

Also fetch https://gbfs.citibikenyc.com/gbfs/en/station_information.json and
report its fields.

Write findings to SCHEMA.md. Do NOT write any ingest, parsing, or modeling
code this session. If a file won't parse, say so rather than guessing.
```

**Gate:** you can read the actual column names in `SCHEMA.md`. Commit. `/clear`.

---

## Phase 1 — Ingest + normalize

```
Read SPEC.md and SCHEMA.md.

Build src/ingest/trips.py: normalize all monthly files into ONE parquet at
data/interim/trips.parquet with a stable schema:
ride_id, started_at, ended_at, start_station_id, end_station_id,
start_lat, start_lng, end_lat, end_lng, rideable_type, member_casual

Requirements:
- Use polars, lazy scan, streaming write. Do not load everything into memory.
- Handle each schema era explicitly per SCHEMA.md. No try/except that
  silently swallows a schema mismatch.
- Log row count before and after every filter, with the reason.
- Drop trips <60s and self-loop trips <2min, but log how many.
- Assert: no null timestamps, ended_at > started_at, station_ids resolve.
- Write tests/test_ingest.py with a synthetic fixture for EACH era.

Then print monthly trip totals so I can eyeball them against Citi Bike's
published counts.
```

**Gate:** monthly totals are within ~1% of published figures. If they're not, stop and find out why before moving on. Commit. `/clear`.

---

## Phase 2 — Stations, zones, weather

```
Read SPEC.md §2 and SCHEMA.md.

Three things:

1. src/ingest/gbfs.py — station table from station_information.json:
   station_id, name, lat, lng, capacity. Then reconcile against station_ids
   present in trips.parquet. Report: how many trip stations have no capacity
   match, and what fraction of trip volume they represent. This number
   matters — tell me plainly, don't paper over it.

2. src/features/zones.py — assign each station to a zone TWO ways:
   (a) H3 res 8, (b) DBSCAN clustering on projected coords, eps=350m.
   Report zone count and station-per-zone distribution for each.

3. src/ingest/weather.py — hourly NYC weather from Open-Meteo's historical
   archive for the trip date range: temp, precipitation, wind, humidity.
   Cache to parquet.

Tests for the zone assignment (known lat/lng → known cell).
```

**Gate:** unmatched-capacity stations are a small share of volume. Sanity-check a few zones on a map. Commit. `/clear`.

---

## Phase 3 — The panel

This table is the spine of everything downstream. Get it right.

```
Read SPEC.md §2.

Build src/features/panel.py producing data/processed/panel.parquet:
one row per (station_id, 15-min interval) covering the full date range,
INCLUDING intervals with zero activity (dense, not sparse).

Columns: departures, arrivals, capacity, zone_h3, zone_dbscan, plus
calendar features (hour, dow, hour_of_week, month, is_holiday) and the
joined weather.

Invariants to assert:
- sum(departures) == sum(arrivals) == trip count in trips.parquet
- no missing intervals per station between its first and last observed trip
- station-hours before a station opened are absent, not zero-filled

Test on one station-week against hand-computed counts.
Then show me: total rows, memory footprint, and a plot of system-wide
departures by hour-of-week.
```

**Gate:** the hour-of-week plot shows twin commuter peaks Mon–Fri and a fat midday hump Sat–Sun. If it doesn't, something is wrong with your timestamps (probably timezone). Commit. `/clear`.

---

## Phase 4 — Inventory reconstruction

Use **plan mode** here (`shift+tab` twice). Real design fork.

```
Read SPEC.md §4. Plan before coding.

Goal: infer inventory I(s,t) for every station-interval, given only
arrivals/departures and capacity. Operator rebalancing R(s,t) is unobserved.

Propose 2-3 approaches for inferring R such that I stays in [0, capacity],
with tradeoffs. Consider at minimum: greedy per-station-day clipping, and
an LP minimizing total |R| subject to bound constraints. Also propose how
to anchor the starting inventory each day.

Don't write code yet. Tell me your recommendation and why.
```

Pick one, then:

```
Implement [CHOICE] as src/models/inventory.py.

Outputs per (station, 15-min): inventory, is_bike_empty, is_dock_full,
minutes_empty, minutes_full, inferred_rebalance_in, inferred_rebalance_out.

Validate three ways and report all three:
1. Bounds never violated after correction.
2. Total inferred rebalancing volume vs. NYC DOT monthly operating report
   figures — order of magnitude check.
3. If data/raw/gbfs/ has snapshots, score reconstructed inventory against
   observed. Report correlation and MAE. Do not skip this if data exists.

Test: synthetic station where I inject a known rebalance and confirm recovery.
```

**Gate:** you have an honest accuracy number for the reconstruction, and you can state it out loud. Commit. `/clear`.

---

## Phase 5 — Demand model + censoring

The heart of the project. Plan mode again.

```
Read SPEC.md §3. Plan first, don't code.

I need latent demand D(s,t) where I only observe Y = min(D, availability).
Censoring flags come from Phase 4 (minutes_empty for departures,
minutes_full for arrivals).

Propose the modeling approach: candidate estimators (NB GLM, LightGBM
Poisson, Tobit/EM), feature set, how to weight partially-censored intervals,
and how to validate the censoring correction specifically — not just overall
fit. Recommend one and justify it.
```

Then:

```
Implement src/models/demand.py + src/models/censoring.py per the plan.

Train on UNCENSORED intervals only. Hold out the last 3 months by time,
never randomly.

Report:
- WMAPE and Poisson deviance on held-out uncensored intervals, overall and
  by demand decile
- feature importance
- THE RECOVERY TEST: take held-out uncensored intervals, artificially
  truncate them at a synthetic inventory cap, run the censoring correction,
  and report how well true counts are recovered — bias and MAE, broken out
  by censoring severity (light / moderate / heavy).

The recovery test is the deliverable. Print it as a clean table.
```

**Gate:** recovery test shows low bias under light/moderate censoring. Some degradation under heavy censoring is expected and fine — report it honestly, don't tune it away. Commit. `/clear`.

---

## Phase 6 — Unmet demand + heatmap

```
Read SPEC.md §3 step 4 and §5.

Two parts:

1. Unmet demand per (station, interval):
   gross_unmet = max(0, predicted_demand * censored_fraction - observed)
   Then estimate SUBSTITUTION: during a stockout at station s, do stations
   within 400m show departures above their own counterfactual? Attribute
   that uplift as displaced, not lost. Output both gross_unmet and net_lost.
   Report system-wide what fraction of gross unmet is displaced vs lost.

2. src/viz/heatmap.py — zone × hour-of-week heatmap of net_lost trips.
   TWO panels: bike-starved and dock-starved. Also a per-dock-normalized
   version. Plus a folium map of the worst zone-hours.

Save figures to reports/.
```

**Gate:** the commuter dipole is visible — outer residential zones bike-starved 7–9am, Midtown/FiDi dock-starved in the same window, mirrored at 6–8pm. If not, debug before continuing. Commit. `/clear`.

---

## Phase 7 — Simulator (hard gate)

**Do not proceed past this phase until it validates.** Everything downstream is meaningless if the simulator can't reproduce a week it has already seen.

```
Read SPEC.md §4 "Forward simulation".

Build src/sim/simulator.py — discrete-event, 15-min steps, all stations:
- draw departures from the fitted demand model
- assign destinations via an OD choice model (multinomial on historical OD
  shares conditioned on hour-of-week and origin)
- physical constraints: no bike → departure LOST (count it);
  no dock at destination → reroute to nearest station with a dock, log the
  extra distance, count as degraded
- inject baseline rebalancing from Phase 4's inferred R
- accept an optional list of induced moves (for Phase 8), default empty

VALIDATION, and this is the gate: replay a held-out historical week.
Compare simulated vs actual on: total trips, per-zone trips, stockout
minutes per station, hourly system profile. Report error on each.
Include a fixed seed and make runs reproducible.

Tell me straight whether it validates. If it doesn't, diagnose rather than
tuning until the numbers look nice.
```

**Gate:** simulated totals within ~5% of actual, per-zone within ~15%, stockout minutes correlated >0.7. If Claude reports it validates, spot-check yourself — this is the phase where an agent is most tempted to declare victory. Commit. `/clear`.

---

## Phase 8 — Marginal value + allocator

```
Read SPEC.md §6 and §7. Plan mode first for the optimizer formulation.

Part A — src/opt/marginal_value.py:
By simulation, compute MV(s,t) = expected trips saved by adding the 1st,
2nd, ... nth bike at station s at time t. Confirm concavity; if it isn't
concave somewhere, show me where and why.

Part B — config/params.yaml with ALL assumptions explicit:
dollars_per_point: 0.20
weekly_budget: 10000
elasticity_a, elasticity_b   # saturating curve, SPEC §6
max_induced_moves_per_station_hour   # capped by through-traffic
max_move_duration_min: 25

Part C — src/opt/allocate.py, greedy v1:
Candidate moves are (origin, dest, interval, payout_level) tuples.
Net value = MV(dest) - MV(origin) - cost. Origins must be surplus.
Restrict to plausible OD pairs (existing flow, under max duration).
Greedy by net-value-per-dollar, updating MVs as bikes move, until budget
is exhausted.

Output: allocation table + total spend + expected trips saved.
```

**Gate:** eyeball the top 20 allocations. Do they make geographic and temporal sense? Money flowing from Midtown to residential zones at 6pm should look obvious. If the optimizer wants to pay people to move bikes between two adjacent quiet stations at 3am, you have a bug. Commit. `/clear`.

---

## Phase 9 — Lift, baselines, sensitivity

```
Read SPEC.md §8.

Run the simulator under 5 policies at the same budget:
1. do nothing  2. uniform spend  3. spend ∝ trip volume
4. top-N stockout stations  5. my optimizer

For each report: fill rate, lift in pp vs. do-nothing, trips recovered,
cost per recovered trip.

Confidence intervals via bootstrap over (a) demand model residuals,
(b) elasticity params sampled over a plausible range, (c) sim seeds.
100 replicates minimum.

Then two sensitivity outputs:
- Lift vs. budget from $0 to 3x current, to find where returns flatten.
- Rank stability: across the elasticity range, how stable is the top-100
  ranking of allocated station-hours? Report rank correlation.

Save the policy comparison table and both plots to reports/.
```

**Gate:** you beat baseline #4 (top-N stockout). If you don't, that's still a legitimate and interesting finding — but understand *why* before you write it up. Commit. `/clear`.

---

## Phase 10 — Writeup

```
Read SPEC.md and every table/figure in reports/.

Draft README.md aimed at a marketplace DS hiring manager who will spend
90 seconds on it. Structure:

1. The decision (2 sentences): where the budget goes and expected fill-rate
   lift with CI.
2. Why unmet demand had to be estimated — the censoring problem, in plain
   language.
3. The heatmap, with the commuter dipole called out.
4. The allocation logic, emphasizing that moving a bike costs the origin.
5. Results table vs. all 4 baselines.
6. Limitations — lead with the elasticity assumption, then pivot to rank
   stability.
7. How I'd validate for real: switchback design + power calc.

Lead with the decision, not the pipeline. Assume they don't care what
libraries I used. Be specific with numbers everywhere. Do not oversell —
flag every assumption.
```

Then write `DECISIONS.md` yourself from your own memory of the forks — that one shouldn't be ghostwritten, and it's what you'll actually reread before an interview.

---

## Habits that make this work

- **Plan mode for phases 4, 5, 7, 8.** Those have genuine design forks. Let it propose options; you choose.
- **`/clear` between every phase.** Stale context makes it reference functions that no longer exist.
- **Commit at every gate.** When something breaks in Phase 7 you want to bisect, not archaeologize.
- **Never accept "it validates" without looking.** Especially Phase 7.
- Add to `CLAUDE.md` as you go — every time you correct the same mistake twice, it belongs in the file.

## When it goes sideways

Numbers look too good:
```
These results look better than I'd expect. Before I believe them, audit for
leakage: is any censoring-derived or post-outcome information reaching the
demand model's training features? Trace the feature lineage and show me.
```

Silent data loss:
```
Row count dropped between [step A] and [step B]. Find every filter, join,
and dropna between them and report rows removed by each with the reason.
Don't fix anything yet — just tell me where it went.
```

Overconfident agreement:
```
You agreed with my last three suggestions. Argue the strongest case against
[decision] and tell me what evidence would change your answer.
```
