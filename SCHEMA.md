# Citi Bike trip data — schema discovery (Phase 0)

Findings from inspecting actual downloaded files. No ingest/parsing/modeling code was written this session — this is pure inspection, per RUNBOOK Phase 0.

## Source structure (important, differs from RUNBOOK's assumption)

`citibikenyc.com/system-data` links to the S3 bucket `https://s3.amazonaws.com/tripdata/`. Its actual layout is **not** one standalone monthly zip per month going back to the start:

- **2013–2023**: one zip per *year* (e.g. `2021-citibike-tripdata.zip`, ~1GB), each containing a nested zip per month (e.g. `202101-citibike-tripdata.zip`), which in turn contains 1–3 CSV part files for large months.
- **2024–present (through 2026-06 at time of writing)**: one standalone zip per month at the top level (e.g. `202606-citibike-tripdata.zip`), containing CSV part files directly (uncompressed/STORED, not DEFLATEd — unusual, makes these zips large for their row count).
- **`JC-*` prefixed keys** are a separate, smaller Jersey City system, published monthly throughout. Not used for this discovery (out of scope — the spec is about the NYC system).

Because of this, "download 12 months of trip zips" from RUNBOOK Day 0 is nontrivial pre-2024 — you'd pull whole-year archives. For this session I sampled by issuing HTTP range requests directly against each zip's central directory / local file headers to pull only the bytes needed for a ~200-row peek, rather than downloading full yearly archives (hundreds of MB to ~1GB each). Files actually inspected:

| Sample | Source | Bytes fetched |
|---|---|---|
| `2019-09` | `2019-citibike-tripdata.zip` → `9_September/201909-citibike-tripdata_1.csv` | 150,000 compressed bytes → 638,540 decompressed (partial-DEFLATE decode) |
| `2021-01` | `2021-citibike-tripdata.zip` → `202101-citibike-tripdata.zip` → `_1.csv` | full nested month zip (203MB) downloaded, first 201 lines streamed out, zip discarded |
| `2021-03` | `2021-citibike-tripdata.zip` → `202103-citibike-tripdata.zip` → `_1.csv` | full nested month zip (283MB) downloaded, first 201 lines streamed out, zip discarded |
| `2026-06` (latest available) | `202606-citibike-tripdata.zip` → `_1.csv` | 300,000 raw bytes (STORED, no decompression needed) |

Only two distinct schema eras were found across these four samples — see below. No file failed to parse.

## ⚠ Correction to the spec's boundary date

SPEC.md says the schema changed "~Feb 2021." **The 2021-01 sample already shows the new schema.** The transition happened before January 2021 — somewhere between Sept 2019 and Jan 2021, not at the Feb 2021 boundary. If Phase 1's normalizer needs the exact cutover month, that requires bisecting further (not done here — out of scope for this session).

---

## Era A — pre-transition (observed in 2019-09; old schema)

**Exact columns and dtypes** (as pandas infers them on read, 200-row sample):

| Column | dtype |
|---|---|
| `tripduration` | int64 |
| `starttime` | object (string) |
| `stoptime` | object (string) |
| `start station id` | int64 |
| `start station name` | object |
| `start station latitude` | float64 |
| `start station longitude` | float64 |
| `end station id` | int64 |
| `end station name` | object |
| `end station latitude` | float64 |
| `end station longitude` | float64 |
| `bikeid` | int64 |
| `usertype` | object — values seen: `Subscriber`, `Customer` |
| `birth year` | int64 |
| `gender` | int64 — values seen: `0`, `1`, `2` |

Note the column names contain **spaces**, not underscores (`start station id`, not `start_station_id`).

**3 sample rows** (verbatim):
```
tripduration,starttime,stoptime,start station id,start station name,start station latitude,start station longitude,end station id,end station name,end station latitude,end station longitude,bikeid,usertype,birth year,gender
327,2019-09-01 00:00:01.9580,2019-09-01 00:05:29.3410,3733,Avenue C & E 18 St,40.730563,-73.973984,504,1 Ave & E 16 St,40.73221853,-73.98165557,39213,Subscriber,1968,1
1145,2019-09-01 00:00:04.1430,2019-09-01 00:19:09.8360,3329,Degraw St & Smith St,40.6829151,-73.99318208,270,Adelphi St & Myrtle Ave,40.69308257,-73.97178913,21257,Customer,1969,0
1293,2019-09-01 00:00:07.3090,2019-09-01 00:21:40.7580,3168,Central Park West & W 85 St,40.78472675,-73.96961715,423,W 54 St & 9 Ave,40.76584941,-73.98690506,15242,Customer,1969,0
```

**station_id format**: plain small integers — `3733`, `3329`, `3168`, `504`, `270`, `486`. No decimals, no dots.

**Timestamp format**: `YYYY-MM-DD HH:MM:SS.ffff` — string, **4 fractional-second digits** (e.g. `2019-09-01 00:00:01.9580`). No timezone offset or `Z` suffix anywhere — **naive, not tz-aware**. Presumed local America/New_York time based on context; not asserted by the data itself.

**Null rate per column** (200-row sample): **0.0% for every column.** No missing values observed in this sample.

**Fields present in this era only**: `tripduration` (precomputed, redundant with started/stopped), `bikeid`, `birth year`, `gender`. These are **dropped entirely** in the new schema — any downstream code relying on rider demographics only works pre-transition.

---

## Era B — post-transition (observed in 2021-01, 2021-03, and 2026-06 — structurally identical across all three, i.e. stable since the transition)

**Exact columns and dtypes**:

| Column | dtype |
|---|---|
| `ride_id` | object (string) |
| `rideable_type` | object — values seen: `classic_bike`, `electric_bike` |
| `started_at` | object (string) |
| `ended_at` | object (string) |
| `start_station_name` | object |
| `start_station_id` | float64 (pandas-inferred — **see gotcha below**) |
| `end_station_name` | object |
| `end_station_id` | float64 (same gotcha) |
| `start_lat` | float64 |
| `start_lng` | float64 |
| `end_lat` | float64 |
| `end_lng` | float64 |
| `member_casual` | object — values seen: `member`, `casual` |

**3 sample rows** (2021-03, verbatim):
```
ride_id,rideable_type,started_at,ended_at,start_station_name,start_station_id,end_station_name,end_station_id,start_lat,start_lng,end_lat,end_lng,member_casual
9FC8CCE19D178879,classic_bike,2021-03-09 20:12:32.695,2021-03-09 20:44:16.900,E 25 St & 1 Ave,6004.07,1 Ave & E 110 St,7522.02,40.738176,-73.977386,40.7923272,-73.9383,member
D1456E5FF72D3C1F,classic_bike,2021-03-08 17:32:35.537,2021-03-08 17:36:27.574,5 Ave & E 29 St,6248.06,W 24 St & 7 Ave,6257.03,40.745167,-73.98683,40.74487634,-73.99529885,member
2CD7789A0765F26A,electric_bike,2021-03-06 09:59:22.657,2021-03-06 10:06:21.540,Columbia Heights & Cranberry St,4829.01,Columbia St & Kane St,4422.05,40.700378,-73.99548,40.68763155,-74.0016256,casual
```

**station_id format**: strings that look like decimals, e.g. `5406.02`, `4789.03`, `6004.07`, `7522.02`. Confirmed via raw CSV text (not pandas) that these are literal — `repr()` shows `'5406.02'` etc. **Gotcha: pandas silently reads this column as `float64`.** That's dangerous for an ingest normalizer — it works for these particular values, but relying on float parsing of what is semantically a string ID risks precision loss (e.g. a hypothetical `.10` suffix collapsing to `.1`) and silently breaks any join if a future station_id isn't cleanly numeric. Read as string, don't let it infer as float.

**Timestamp format**: `YYYY-MM-DD HH:MM:SS.fff` — string, **3 fractional-second digits** (milliseconds, e.g. `2021-01-19 19:43:36.986`), one fewer digit than Era A. No timezone offset — **naive, not tz-aware**, same as Era A.

**Null rate per column** (200-row samples):

| Column | 2021-01 | 2021-03 | 2026-06 |
|---|---|---|---|
| `end_station_name` | 2.5% | 0.0% | 0.0% |
| `end_station_id` | 2.5% | 0.0% | 0.0% |
| `end_lat` | 2.5% | 0.0% | 0.0% |
| `end_lng` | 2.5% | 0.0% | 0.0% |
| all other columns | 0.0% | 0.0% | 0.0% |

The 2021-01 nulls are all on the *same* 5 rows (missing end station entirely) — consistent with a trip that ended outside a dock (dockless/valet drop) rather than random missingness. Worth carrying an explicit "ended outside dock" flag in the normalizer rather than silently dropping these rows.

**Renamed/remapped fields vs Era A**: `usertype` (`Subscriber`/`Customer`) → `member_casual` (`member`/`casual`); no direct one-to-one value mapping asserted here, just noting the rename to catch in the normalizer.

---

## GBFS `station_information.json`

`https://gbfs.citibikenyc.com/gbfs/en/station_information.json`, GBFS version `1.1`, `last_updated` is a Unix epoch integer (e.g. `1785801590`). `data.stations` — 2,463 stations at fetch time.

**Fields** (all 100% present except `region_id` at 99.5% and `eightd_station_services`, present but empty on all but 1 station):

| Field | Type | Notes |
|---|---|---|
| `station_id` | string | **UUID**, e.g. `66dd1f44-0aca-11e7-82f6-3863bb44ef7c` |
| `external_id` | string | Identical to `station_id` in every record checked |
| `name` | string | Human-readable station name |
| `short_name` | string | e.g. `5785.05` — **this, not `station_id`, is what matches the trip data's `start_station_id`/`end_station_id` format** |
| `lat`, `lon` | float | |
| `region_id` | string | Numeric-looking string, e.g. `"71"` |
| `capacity` | int | Dock count — 100% populated |
| `rental_uris` | dict | `{android, ios}` deep-link URLs |
| `rental_methods` | list[string] | e.g. `["KEY", "CREDITCARD"]` |
| `has_kiosk` | bool | |
| `station_type` | string | e.g. `"classic"` |
| `electric_bike_surcharge_waiver` | bool | |
| `eightd_has_key_dispenser` | bool | |
| `eightd_station_services` | list | empty on 2,462/2,463 stations |

### ⚠ Crosswalk gotcha for Phase 2

**GBFS's `station_id` (UUID) is not the join key against trip data.** Trip data's `start_station_id`/`end_station_id` (e.g. `5406.02`) matches GBFS's **`short_name`** field (e.g. `5785.05`), not `station_id`. Joining trip data to GBFS on `station_id` will silently produce zero matches. This is exactly the kind of bug Phase 0 exists to catch before Phase 2's `src/ingest/gbfs.py` is written.

Also unresolved: **Era A's plain-integer station IDs (e.g. `3733`) have no obvious crosswalk to either GBFS field** (`station_id` is a UUID, `short_name` is a decimal-formatted string). Reconciling Era A stations against current GBFS capacity data will need either a historical station-ID mapping table or accepting that some fraction of old-schema stations won't resolve — flag this explicitly when Phase 2 reports its unmatched-capacity fraction.
