# Project
Estimate censored bike demand, simulate station inventory, allocate a fixed
incentive budget to maximize fill rate. See SPEC.md for full design.

# Current phase
Phase 3 complete (panel.parquet: 79.8M dense station-interval rows, all
invariants passing). Next: Phase 4, inventory reconstruction -- use plan
mode, real design fork per RUNBOOK.md.

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
