#!/bin/bash
# Reference copy for version control. The copy cron actually executes lives
# at ~/citibike-gbfs-data/gbfs_poll.sh -- macOS's TCC privacy protection
# blocks cron (no Full Disk Access) from reading/writing anything under
# ~/Documents, so both the script and its output live outside that tree.
# If you edit this, copy it to ~/citibike-gbfs-data/gbfs_poll.sh too.
OUT_DIR="$HOME/citibike-gbfs-data"
mkdir -p "$OUT_DIR"
curl -s https://gbfs.citibikenyc.com/gbfs/en/station_status.json > "$OUT_DIR/status_$(date +%Y%m%d_%H%M).json"
