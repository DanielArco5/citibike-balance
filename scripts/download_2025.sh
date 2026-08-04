#!/bin/bash
# Downloads full 2025 monthly Citi Bike trip zips into data/raw/trips/,
# extracts the CSV part files, deletes the zip, and continues past any
# month that fails rather than aborting. Writes a summary at the end.
set -u

TRIPS_DIR="/Users/danielcrown1/Documents/Citibike/citibike-balance/data/raw/trips"
STATUS_FILE="$TRIPS_DIR/_download_status.txt"
MONTHS="202501 202502 202503 202504 202505 202506 202507 202508 202509 202510 202511 202512"

mkdir -p "$TRIPS_DIR"
: > "$STATUS_FILE"

succeeded=()
failed=()

for m in $MONTHS; do
  url="https://s3.amazonaws.com/tripdata/${m}-citibike-tripdata.zip"
  zip_path="$TRIPS_DIR/${m}-citibike-tripdata.zip"

  echo "[$(date +%H:%M:%S)] downloading $m ..."
  if ! curl -sf --retry 3 -o "$zip_path" "$url"; then
    echo "[$(date +%H:%M:%S)] $m: DOWNLOAD FAILED" | tee -a "$STATUS_FILE"
    failed+=("$m")
    rm -f "$zip_path"
    continue
  fi

  echo "[$(date +%H:%M:%S)] extracting $m ..."
  if ! unzip -oq "$zip_path" '*.csv' -d "$TRIPS_DIR"; then
    echo "[$(date +%H:%M:%S)] $m: EXTRACT FAILED" | tee -a "$STATUS_FILE"
    failed+=("$m")
    rm -f "$zip_path"
    continue
  fi

  rm -f "$zip_path"
  echo "[$(date +%H:%M:%S)] $m: OK" | tee -a "$STATUS_FILE"
  succeeded+=("$m")
done

{
  echo "---"
  echo "SUCCEEDED (${#succeeded[@]}): ${succeeded[*]:-none}"
  echo "FAILED (${#failed[@]}): ${failed[*]:-none}"
} | tee -a "$STATUS_FILE"
