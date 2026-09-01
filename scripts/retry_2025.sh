#!/bin/bash
set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRIPS_DIR="$REPO_ROOT/data/raw/trips"
STATUS_FILE="$TRIPS_DIR/_retry_status.txt"
MONTHS="202504 202505 202506 202507"

: > "$STATUS_FILE"
succeeded=()
failed=()

for m in $MONTHS; do
  url="https://s3.amazonaws.com/tripdata/${m}-citibike-tripdata.zip"
  zip_path="$TRIPS_DIR/${m}-citibike-tripdata.zip"
  echo "[$(date +%H:%M:%S)] downloading $m ..."
  ok=0
  for attempt in 1 2 3; do
    if curl -sf -o "$zip_path" "$url"; then
      ok=1
      break
    fi
    echo "[$(date +%H:%M:%S)] $m: attempt $attempt failed, retrying ..." | tee -a "$STATUS_FILE"
    rm -f "$zip_path"
  done

  if [ "$ok" -ne 1 ]; then
    echo "[$(date +%H:%M:%S)] $m: DOWNLOAD FAILED after 3 attempts" | tee -a "$STATUS_FILE"
    failed+=("$m")
    continue
  fi

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
