# Project
Estimate censored bike demand, simulate station inventory, allocate a fixed
incentive budget to maximize fill rate. See SPEC.md for full design.

# Current phase
Phase 6 complete: unmet demand (demand.py, censoring.py) + substitution
netting (substitution.py) + two-panel heatmap (heatmap.py) all run
end-to-end on the real ~76M-row panel, chunked per month (see memory rule
below) after a first attempt thrashed for 8+ hours and was killed. Heatmap
is DIAGNOSTIC ONLY -- ranked by per-dock lower-confidence-bound (primary)
and raw volume (secondary); it shows where/when scarcity occurs, not where
a bike is worth allocating. Ranking for allocation is deferred to Phase 8's
marginal value, which dissolves the volume-vs-rate tension per-dock
ranking can't resolve -- see DECISIONS.md, "volume vs. per-dock ranking"
and its Phase 8 deferral note. Next: Phase 7, the forward simulator (hard
gate -- must validate against a historical week before Phase 8 proceeds).

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
