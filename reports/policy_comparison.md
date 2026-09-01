# Phase 9: policy comparison (SPEC.md §8)

> CAVEAT (Phase 7): fill rate and lift are reported at ZONE and SYSTEM level ONLY. The simulator's per-(station, hour-of-week) stockout rate is confirmed simulator noise (pooled correlation plateaus at ~0.10 across 6 held-out weeks, DECISIONS.md's Phase 7 entry) -- no per-station number appears anywhere in this output.
>
> CAVEAT (Phase 7): all policies within a bootstrap replicate share the SAME demand-residual, elasticity, and simulator-seed draw (paired design). Destination-assignment noise is assumed -- not proven -- to cancel in the reported lift (the difference vs. do_nothing), per Phase 7's validation.

**HEADLINE FINDING, read this before the table below:** at this replicate count, NO policy's fill-rate lift is statistically distinguishable from zero -- every system-level 90% CI straddles zero (0/5 policies significant), and the same is true at zone level (0/2,684 policy-zone pairs significant, checked directly, not assumed). This is a real power limitation from the bootstrap replicate count, not a coding defect -- the demand-residual and elasticity axes contribute genuinely large week-to-week variance that a bigger N would narrow. Point estimates below (e.g. a policy's median lift, or its cost-per-recovered-trip) should NOT be read as established effects; treat every ranking or dollar figure in this table as directionally suggestive at best until re-run with more replicates.

Zone-level rows with undefined (0/0) fill rate this simulated week: 2,763 / 130,320 (2.1%) -- excluded from zone-level statistics, not zeroed.

## System-level results

| Policy | Fill rate (median, 90% CI) | Lift vs. do-nothing (pp) | Trips recovered | Dollar cost | Cost / recovered trip |
|---|---|---|---|---|---|
| Do nothing | 0.8987 (0.8924–0.9049) | +0.00 (+0.00–+0.00) | +0 (+0–+0) | $0 | n/a |
| Uniform spend | 0.8983 (0.8922–0.9055) | +0.00 (-0.20–+0.26) | -1,681 (-11,204–+12,283) | $10,000 | $1.65 ($0.66–$13.80) [60% of replicates excluded, trips_recovered<=0] |
| Proportional to volume | 0.8995 (0.8922–0.9057) | +0.03 (-0.20–+0.26) | -816 (-10,566–+9,322) | $10,000 | $2.13 ($0.98–$4.64) [57% of replicates excluded, trips_recovered<=0] |
| Top-N stockout (naive) | 0.8993 (0.8931–0.9051) | +0.06 (-0.18–+0.21) | +2,512 (-12,341–+8,875) | $9,995 | $1.80 ($1.12–$9.58) [38% of replicates excluded, trips_recovered<=0] |
| Our allocator | 0.8981 (0.8897–0.9039) | -0.09 (-0.37–+0.07) | -4,852 (-17,891–+2,524) | $377 | $0.24 ($0.06–$1.73) [88% of replicates excluded, trips_recovered<=0] |
| Our allocator (full budget) | 0.8996 (0.8935–0.9060) | +0.09 (-0.15–+0.27) | +1,060 (-11,159–+9,983) | $9,999 | $1.95 ($0.88–$16.52) [43% of replicates excluded, trips_recovered<=0] |

## RUNBOOK gate: does `allocator` beat `top_n_stockout` (naive top-N)?

UNDERPOWERED TO CONCLUDE on raw fill-rate lift (-0.088pp vs. +0.058pp) -- both policies' lift CIs straddle zero, so this point-estimate comparison is not a resolved answer, just a tiebreaker if one is needed. Per RUNBOOK: "that's still a legitimate and interesting finding -- understand why." `allocator` spends $377 vs. `top_n_stockout`'s $9,995 -- see `allocator_full_budget` for the budget-exhausting comparison, and the module docstrings in src/opt/policy_baselines.py for the underlying candidate-pool-exhaustion finding.

Full zone-level table: `reports/policy_comparison_by_zone.parquet` (not reproduced here -- one row per policy per zone, machine-readable only, per the station/zone-level caveat above).

---

## Treated-cell paired comparison

Restricted to the (station, hour-of-week) cells each policy itself funded (post `apply_move_cap`), paired against a do-nothing run on the SAME cells and the SAME replicate seed -- added after the system-level result above turned out to be a measurement-design problem, not a power problem: pooling fill rate over the WHOLE network (~875K trips/week) dilutes a few-thousand-trip treatment below the bootstrap noise floor regardless of replicate count. Still never a per-station number (see the station-level caveat above) -- every figure below is pooled over hundreds to thousands of cells per policy. `trips_recovered` here is the REDUCTION IN LOST trips at the treated cells (what the incentive actually targets), not a change in arrivals -- see `build_treated_comparison`'s docstring for why that distinction matters.

**4/5 policies now show a fill-rate lift CI that excludes zero** -- vs. 0/5 at system level above, from the SAME 40 replicates and seeds. The treatment effect was real and measurable all along; the system-level table diluted it away.

| Policy | Fill rate on treated cells (median, 90% CI) | Lift vs. do-nothing, same cells (pp) | P(lift>0) | Trips recovered (fewer lost) | Dollar cost | Cost / recovered trip | Avg. treated cells |
|---|---|---|---|---|---|---|---|
| Uniform spend | 91.39% (89.97–92.49) | +2.13 (+0.85–+3.19) | 100% | +801 (+407–+1,167) | $10,000 | $12.49 ($8.57–$24.61) | 2,998 |
| Proportional to volume | 91.24% (89.96–92.53) | +1.96 (+0.68–+3.10) | 100% | +762 (+328–+1,146) | $10,000 | $13.15 ($8.72–$31.02) | 2,998 |
| Top-N stockout (naive) | 89.86% (88.06–91.12) | +2.56 (+0.48–+4.03) | 100% | +569 (+248–+874) | $9,995 | $17.14 ($10.45–$34.85) | 2,081 |
| Our allocator | 89.81% (88.63–91.02) | +0.56 (-0.63–+1.94) | 72% | +234 (-170–+735) | $377 | $1.34 ($0.51–$10.00) [20% excluded, trips_recovered<=0] | 2,998 |
| Our allocator (full budget) | 91.31% (89.99–92.31) | +2.06 (+0.55–+3.08) | 100% | +793 (+209–+1,174) | $9,999 | $12.64 ($8.52–$47.88) | 2,998 |
