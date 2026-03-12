#!/bin/bash

# =========================
# CONFIG
# =========================
PART_DIR="/mnt/newdisk/anass/dfp_parts"
OUT_DIR="/mnt/newdisk/anass/dfp_out"
SCRIPT="/mnt/newdisk/anass/scripts/collector_daily_features.py"

mkdir -p "$OUT_DIR"

# =========================
# RUN COLLECTORS IN PARALLEL
# =========================
echo "Starting parallel collectors..."

python3 "$SCRIPT" "$PART_DIR/mcelog_part_0.csv" "$OUT_DIR/out_part_0.csv" &
python3 "$SCRIPT" "$PART_DIR/mcelog_part_1.csv" "$OUT_DIR/out_part_1.csv" &
python3 "$SCRIPT" "$PART_DIR/mcelog_part_2.csv" "$OUT_DIR/out_part_2.csv" &
python3 "$SCRIPT" "$PART_DIR/mcelog_part_3.csv" "$OUT_DIR/out_part_3.csv" &

wait

echo "===================================="
echo "ALL COLLECTORS FINISHED"
echo "Outputs written to $OUT_DIR"
echo "===================================="
