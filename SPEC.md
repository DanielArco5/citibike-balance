# Citi Bike Incentive Allocation — Project Spec

**The operating question:** Given a fixed weekly incentive budget, which station-hours should we pay riders to rebalance, and how many otherwise-lost trips does that recover?

This doc is written to be dropped into a repo as `SPEC.md` and read by Claude Code. Sections 1–9 are the analytical design. Section 10 is the Claude Code operating manual. Section 11 is interview prep.

---

## 0. Why Citi Bike is the *harder, better* version of this project

Everyone does this with TLC taxi data. Citi Bike is better for three reasons, and all three are things an interviewer will light up about:

1. **Demand is censored.** Trip records only contain *completed* trips. A rider who walked to an empty station and gave up leaves no trace. With taxi data you at least see requests-ish behavior; here you must *estimate* unmet demand. This is the real marketplace problem and the analytical core of the project.
2. **Scarcity is two-sided at the same node.** A station can fail because it has no bikes (can't start) or no docks (can't end). Dock-blocking is real unfulfilled demand and most candidates forget it entirely.
3. **The incentive lever actually exists in production.** Citi Bike runs **Bike Angels** — riders see per-station point values (roughly ↘4 to ↗4) refreshed every 15 minutes, and earn points for riding from surplus stations to deficit stations. Points redeem for rewards; the Lyft-credit tier implies roughly **$0.20/point**, which gives you a defensible price anchor instead of a made-up cost per incentive. You are not inventing a hypothetical mechanism — you are re-deriving an allocation policy for a live one.

**Rebalancing is invisible in the trip data.** Trucks, bike trains, and valet moves do not appear as trips. You will have to infer them as residuals. This is a feature, not a bug — it's a great section of your writeup.

---

## 1. Data layer

### Sources

| Source | What it gives you | Notes |
|---|---|---|
| Citi Bike trip history (monthly CSV zips, linked from `citibikenyc.com/system-data`) | One row per completed trip: start/end time, station, lat/lng, rideable type, member vs casual | Schema changed ~Feb 2021. Big months are split across multiple CSVs inside one zip. |
| GBFS `station_information.json` | Station capacity (dock count), lat/lng, station ID crosswalk | **Capacity is not in the trip data.** You need this for the dock-full side. |
| GBFS `station_status.json` | Live bikes/docks available, updated ~every minute | Live only. Start a cron job on day 1 to accumulate your own history while you build the rest. |
| Community GBFS archives | Historical station_status | Best-effort. If you get 2–3 months of real inventory snapshots, your model quality jumps a tier. If you can't, use the reconstruction in §4. |
| NOAA / Open-Meteo hourly weather for NYC | Temp, precip, wind | Precipitation is the single largest demand shifter. Non-negotiable feature. |
| NYC DOT monthly operating reports | Aggregate rebalancing counts, fleet size | Use to **validate** your inferred rebalancing volume. |

### Hard rules for the ingest layer
- **Never write schema-dependent code before printing the actual schema.** Column names differ across the 2021 boundary; station IDs changed format. Build an explicit normalizer with a test per era.
- Land raw → `data/raw/`, normalized → `data/interim/*.parquet`, features → `data/processed/`. Parquet + a hash-keyed cache so you never re-parse 30M rows.
- Drop the same things Citi Bike drops, and document it: sub-60s trips, test stations, staff trips (already removed upstream, but verify).
- Scope: **pick 6–12 months, don't boil the ocean.** Suggested: 12 months so you get a full seasonal cycle, train on months 1–9, hold out 10–12.

---

## 2. The unit of analysis

You need a `(zone, hour)` grid. Two decisions:

**Spatial.** Station-level is ~2,000 units and sparse; borough-level is useless. The right zone is a **substitution neighborhood** — the set of stations a rider would actually walk between. Options, in order of preference:
1. Cluster stations by ~300–400m walking distance (DBSCAN on projected coords, or contiguity on the street network). Justify the radius with a substitution analysis (§3).
2. H3 resolution 8 (~460m edge). Fast, reproducible, easy to explain, plays well with maps.
3. NTA neighborhoods. Only if you need to join to census/ACS features.

Do **both** station-level (for the optimizer, since incentives are paid at stations) and zone-level (for the heatmap and forecast stability). Forecast at zone level, allocate at station level.

**Temporal.** Hourly for the forecast. But run the inventory simulation at **5 or 15-minute** resolution — an hour is long enough that a station can empty and refill within it, and hourly-only simulation will systematically understate stockouts.

---

## 3. Demand model (the censored part — this is the project)

### The setup
Let $D_{s,t}$ = latent trip-start demand at station $s$ in interval $t$. You observe $Y_{s,t} = \min(D_{s,t}, \text{available bikes})$. When the station has bikes the whole interval, $Y = D$ (uncensored). When it hits zero, $Y < D$ and you don't know by how much.

### The approach
1. **Flag censored intervals.** From observed or reconstructed inventory (§4): any interval where bikes_available hit 0 (for departures) or docks_available hit 0 (for arrivals). Record *stockout minutes*, not just a binary — a station empty for 4 minutes is barely censored; empty for 50 minutes is heavily censored.
2. **Fit the demand model on uncensored intervals only.** Negative binomial GLM or gradient boosting (LightGBM with Poisson objective) on:
   - hour-of-week, month, holiday flags
   - weather: temp, precip (current *and* lagged 1–2h), wind
   - station fixed effects / embeddings, capacity, elevation
   - nearby subway entrances, POI density, land use
   - lagged demand at the same station, and demand at neighbors
3. **Predict into censored intervals.** $\widehat{\text{unmet}}_{s,t} = \max(0, \hat{D}_{s,t} \cdot \frac{\text{stockout minutes}}{\text{interval minutes}} - Y_{s,t})$.
4. **Net out substitution.** Some of that unmet demand didn't vanish — the rider walked 200m and took a bike from the next station. Estimate this: for each stockout event at $s$, compare neighbor station departures against their own counterfactual. The uplift at neighbors is *displaced*, not *lost*. Report both **gross unmet** and **net-lost** demand. Skipping this overstates your opportunity by a lot, and an interviewer who knows the space will ask.

### Why this framing beats the alternatives
You'll see two shortcuts online: (a) treat observed trips as demand — wrong, it's exactly backwards, since the most undersupplied stations look *low-demand*; (b) count stockout minutes as the metric — better, but it doesn't tell you how many *trips* were lost, so you can't price the intervention. Estimating latent demand lets you convert stockouts into trips, and trips into dollars.

### Validate the censoring correction
Hold out intervals that *were* uncensored, artificially censor them (truncate at a fake inventory cap), and check that your model recovers the true count. Report recovery error. This is the single most convincing validation artifact in the whole project.

---

## 4. Supply / inventory model

### Reconstruct inventory when you lack GBFS history
Inventory follows a flow balance:

$$I_{s,t+1} = I_{s,t} + \text{arrivals}_{s,t} - \text{departures}_{s,t} + R_{s,t}$$

where $R$ is operator rebalancing (unobserved). Approach:
- Compute the naive trajectory with $R=0$ from trip data alone.
- The naive trajectory will drift out of $[0, \text{capacity}]$ — physically impossible. **Those violations are your rebalancing signal.** Infer the minimal $R_{s,t}$ that keeps every station in bounds (this is a small LP or a greedy clipping pass per station-day).
- Cross-check total inferred $|R|$ against the DOT monthly operating report rebalance counts. If you're within the right order of magnitude, say so and move on; if you're 5x off, your capacity data or station matching is wrong.
- Anchor each station-day at a plausible starting inventory (e.g. observed overnight steady state, or optimize the anchor to minimize bound violations).

If you *do* have GBFS history for even a few weeks, use it to score this reconstruction. Report the correlation. Honest error bars here > false precision.

### Forward simulation
Build a discrete-event simulator over the station network:
- Sample demand from the fitted model (departures per station-interval + a destination choice model, e.g. multinomial on historical OD shares conditioned on hour).
- Apply physical constraints: no bike → departure lost; no dock → arrival must re-route to nearest station with a dock (add the extra travel time, and count it as a degraded trip).
- Inject rebalancing (baseline schedule) and, later, incentive-induced moves.
- This simulator is what turns "we allocated $X here" into "fill rate went up Y%." Build it before the optimizer.

---

## 5. Heatmap deliverable

A `station × hour-of-week` matrix of **estimated net-lost trips**, plus a map view.

- Rows: zones ordered by total lost demand (or clustered so patterns pop).
- Columns: 168 hours of week.
- Cells: expected lost trips per week, or lost trips per capacity-dock (normalizes for station size — usually the more actionable view).
- **Two panels:** bike-starved (can't start) and dock-starved (can't end). They are different neighborhoods at different hours, and showing that split is a differentiator.

Expected finding to sanity-check against: residential outer-zone stations bleed bikes 7–9am and are dock-starved 6–8pm; Midtown/FiDi is the mirror image. If your heatmap doesn't show this commuter dipole, you have a bug.

---

## 6. The incentive response curve (be honest here)

**This is where the project is weakest and where you get the most credit for saying so.**

You need $\Delta\text{moves} = f(\text{incentive}, s, t)$ — how many extra rebalancing rides a given payout buys. Options, best to worst:

1. **Natural experiment on Bike Angels point variation.** Points update every 15 minutes and vary by station. If you can scrape or reconstruct point values (they're derived from inventory state, so partly predictable from inventory — which means naive regression is confounded), use a **regression discontinuity** at the thresholds where a station's point value flips, comparing trip flows just above and below. Control for inventory state, which is the confounder driving both.
2. **Elasticity from the literature.** Cite published Bike Angels analyses and bikeshare crowdsourced-rebalancing work; adopt a plausible response curve and state it as an assumption.
3. **Parametric assumption + sensitivity analysis.** Assume $\Delta\text{moves} = a \cdot (1 - e^{-b \cdot \text{payout}})$ — concave, saturating, zero at zero. Then show your allocation's *ranking* of station-hours is stable across a wide range of $a, b$. **The ranking being robust to elasticity assumptions is the actual finding.** That's a legitimate and defensible result.

Also model **participation supply**: you can't buy 50 moves at a station where only 3 riders per hour pass through. Cap induced moves by realistic through-traffic. Ignoring this is the classic way these optimizers produce nonsense.

Price anchor: use ~$0.20/point (from the Lyft-credit redemption rate) so your budget is denominated in something real. A $10,000/week budget ≈ 50,000 points ≈ a defensible number of incentivized moves.

---

## 7. The allocation optimizer

### Marginal value
The value of adding one bike to station $s$ at time $t$ is:

$$MV(s,t) = \mathbb{E}[\text{trips saved}] = P(\text{stockout} \mid s,t) \times \mathbb{E}[\text{unmet trips} \mid \text{stockout}]$$

This is **concave** in bikes added — the first bike at an empty station saves more than the tenth. Compute the full marginal curve by simulation, not a closed form.

### The key move: incentives move bikes, they don't create them
A rebalancing ride *removes* a bike from the origin. So the objective for incentivizing a move $o \to d$ at time $t$ is:

$$\text{Net value} = MV(d,t) - MV(o,t) - \text{cost}$$

This makes it a **transportation / min-cost flow problem**, not a simple ranking. Origins should be genuine surplus (negative or near-zero $MV$), destinations genuine deficit. Constrain $o \to d$ pairs to plausible rider trips (say, under 25 minutes and roughly along existing OD flows — you're subsidizing trips people would nearly take anyway).

### Formulations, in escalating order (build v1 first, ship it, then upgrade)
- **v1 — Greedy marginal-value-per-dollar.** Sort candidate (o, d, t, payout-level) tuples by net value per dollar, take until budget exhausted, updating marginal values as you go. For a concave separable objective under one budget constraint, greedy is near-optimal. **This is a perfectly good answer and it's interpretable.**
- **v2 — LP/MILP.** Decision variable = payout at each (station, hour) or moves on each (o, d, t) arc. Maximize expected trips saved s.t. total spend ≤ B, moves ≤ through-traffic cap, flow conservation on bikes. Use `PuLP`/`OR-Tools`/`HiGHS`.
- **v3 — Rolling-horizon / stochastic.** Re-optimize every 15 min against forecast uncertainty. Mention it as future work; don't build it unless you have time.

### Constraints that make it credible
- Budget: hard cap, weekly.
- Through-traffic cap per station-hour (see §6).
- Minimum incentive granularity (you can't pay 0.3 points).
- Optional fairness constraint: minimum spend in outer-borough / equity-priority zones. Marketplace teams have this constraint in real life, and adding it — then showing the efficiency cost of it — is a strong touch.

---

## 8. Fill-rate lift + baselines

**Metric:** fill rate = fulfilled trips / (fulfilled + net-lost). Report system-wide and by zone.

**Method:** re-run the §4 simulator with incentive-induced moves injected per your allocation. Compare against baselines — and you must include baselines, because a number with no comparison is meaningless:

1. Do nothing (current rebalancing only)
2. Uniform spend across all stations
3. Spend proportional to trip volume
4. Spend on top-N stockout stations (the naive "obvious" policy)
5. **Your optimizer**

Report lift as a point estimate **with an interval**, bootstrapped over (a) demand model residuals, (b) elasticity parameter uncertainty, (c) simulation stochasticity. A result like *"+2.1pp fill rate (90% CI: 1.2–3.4), vs +0.9pp for the naive top-N policy"* is enormously more credible than "+3%."

Also report **cost per recovered trip** — that's the number an ops leader actually budgets against, and it lets you answer "should the budget be bigger?" Plot lift vs. budget from $0 to 3× and find where marginal returns flatten. **That curve is arguably your best single slide.**

---

## 9. Validation and what you'd do with real access

Be explicit that counterfactual simulation is not causal proof, and propose the real test:

- **Switchback design.** Randomize incentive policy (yours vs. status quo) at the zone-week or zone-day level; zones are spatially interfering so you need geographic clustering + a buffer, or a synthetic-control approach.
- **Primary metric:** fill rate. **Guardrails:** cost per recovered trip, rider wait/walk time, whether gains cannibalize from adjacent zones.
- **Power calculation:** given observed variance in zone-level fill rate, how many zone-weeks to detect a 1pp lift? Do this arithmetic — it's a strong signal of experimental maturity.
- Note the gaming risk: Bike Angels is farmable (there's public reporting on people optimizing it hard). Any allocation policy needs anti-abuse constraints — cooldowns, per-user caps, minimum trip distance.

---

## 10. Building this with Claude Code

### Repo layout
```
citibike-balance/
  CLAUDE.md              # persistent context — see below
  SPEC.md                # this file
  data/{raw,interim,processed}/   # gitignored
  src/
    ingest/     trips.py  gbfs.py  weather.py  schema.py
    features/   zones.py  calendar.py  panel.py
    models/     demand.py  censoring.py  inventory.py
    sim/        simulator.py
    opt/        marginal_value.py  allocate.py
    viz/        heatmap.py  maps.py
  tests/
  notebooks/    # exploration only, nothing importable
  reports/      # figures + writeup
  Makefile
```

### CLAUDE.md — put this in the repo root
```markdown
# Project
Estimate censored bike demand, simulate station inventory, allocate a fixed
incentive budget to maximize fill rate. See SPEC.md for full design.

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
```

### Phase plan — one Claude Code session per phase, commit between each

| Phase | Goal | Definition of done |
|---|---|---|
| 0 | Schema discovery | A markdown table of actual columns per era, written by inspecting files. No modeling code yet. |
| 1 | Ingest + normalize | One parquet trips table, era-normalized, tested. Row counts reconcile to published monthly totals. |
| 2 | Stations + zones + weather | Station table with capacity, zone assignment, joined hourly weather. |
| 3 | Panel construction | `(station, 15-min)` panel of departures/arrivals. This is the spine of everything. |
| 4 | Inventory reconstruction | Inventory trajectories in-bounds; inferred rebalancing volume sanity-checked vs DOT reports. |
| 5 | Demand model + censoring | Fitted model, held-out WMAPE, **and the artificial-censoring recovery test**. |
| 6 | Unmet demand + heatmap | The two-panel heatmap. Commuter dipole visible. |
| 7 | Simulator | Replays a historical week within tolerance of actuals. **Gate: do not proceed until this validates.** |
| 8 | Marginal value + optimizer | v1 greedy allocation under budget. |
| 9 | Lift + baselines + sensitivity | Table of 5 policies with CIs; lift-vs-budget curve. |
| 10 | Writeup | README with the decision, not the pipeline. |

### How to drive Claude Code well on this
- **Start every phase by having it read SPEC.md and restate the phase's contract** (inputs, outputs, invariants) before writing code. Cheap, and catches drift immediately.
- **Make it write the test first** for anything numerical. Censored-demand recovery and inventory-balance conservation are both easy to test on synthetic data with known answers.
- **Use plan mode for phases 5, 7, and 8.** Those have real design forks; let it propose 2–3 approaches and pick deliberately rather than accepting the first thing it writes.
- **Fail loudly on data assumptions.** Ask it to add asserts: inventory ∈ [0, capacity], arrivals+departures reconcile to trip counts, no station-hour with negative unmet demand.
- **Beware of the confident pipeline.** The single most likely failure mode is a beautiful, well-structured, fully-tested pipeline built on a hallucinated column name or a wrong capacity join. Phase 0 exists to prevent exactly this. Don't skip it.
- Keep `notebooks/` for exploration but never import from them; make Claude promote anything useful into `src/` with a test.

---

## 11. Interview questions you will get — prepare these

1. *"Your demand is estimated, not observed. How do you know it's right?"* → the artificial-censoring recovery test, plus the substitution netting.
2. *"How much of your 'unmet demand' is just people walking to the next station?"* → you measured it; report gross vs net-lost.
3. *"Where does your elasticity come from?"* → be honest, then pivot to the sensitivity analysis showing the ranking is stable.
4. *"Why not just spend on the stations with the most stockouts?"* → that's baseline #4; show your lift over it, and explain that stockout count ignores both the marginal-value curve and the cost of pulling a bike out of the origin.
5. *"What breaks if you deploy this?"* → gaming/farming, participation caps, spatial spillover between adjacent incentivized zones, forecast degradation in weather regimes not in training.
6. *"How would you actually prove the lift?"* → switchback design, power calc, guardrail metrics.
7. *"Should the budget be bigger?"* → the lift-vs-budget curve and cost per recovered trip.

---

## Scope guardrail

The full build is ~3–4 focused weekends. If you have one weekend: do phases 0–6 and ship the heatmap with a **rank-ordered** priority list instead of an optimizer. An excellent censored-demand estimate with an honest heatmap beats a half-built MILP every time — and the censoring work is the part that actually distinguishes you.
