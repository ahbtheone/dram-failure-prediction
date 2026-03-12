import sys
import pandas as pd
from collections import defaultdict, Counter
from datetime import timedelta

# =======================
# USAGE
# =======================
# python3 collector_daily_features_scaledfp.py \
#   mcelog_part_0.csv \
#   out_part_0.csv

# =======================
# ARGS
# =======================
IN_PATH = sys.argv[1]
OUT_PATH = sys.argv[2]

print(f"[Collector] Loading {IN_PATH}")

# =======================
# LOAD
# =======================
df = pd.read_csv(IN_PATH)

# =======================
# FIX TIMESTAMPS (SAFE)
# =======================
df["error_time"] = pd.to_datetime(
    df["error_time"].astype(str).str.replace("^0001", "2018", regex=True),
    errors="coerce"
)
df = df.dropna(subset=["error_time"])

df["day"] = df["error_time"].dt.date

# =======================
# SORT (CRITICAL)
# =======================
df = df.sort_values(["sid", "error_time"])

# =======================
# STATE (ScaleDFP-style)
# =======================
rows = []

state = {}  # sid -> state dict

def init_state():
    return {
        "ce_total": 0,
        "row_counter": Counter(),
        "bank_counter": Counter(),
        "timestamps": [],
    }

# =======================
# COLLECTOR LOOP
# =======================
total = df["sid"].nunique()
for i, (sid, g) in enumerate(df.groupby("sid"), 1):
    if i % 100 == 0:
        print(f"[{i}/{total}] {sid}")

    s = init_state()
    current_day = None

    for _, r in g.iterrows():
        day = r["day"]
        t = r["error_time"]
        row = r.get("row", None)
        bank = r.get("bankid", None)

        # new day → emit features BEFORE updating with today
        if current_day is not None and day != current_day:
            ts = s["timestamps"]
            deltas = (
                pd.Series(ts).diff().dt.total_seconds().dropna()
                if len(ts) > 1 else []
            )

            rows.append([
                current_day,
                sid,
                s["ce_total"],
                len(s["bank_counter"]),
                len(s["row_counter"]),
                deltas.mean() if len(deltas) else 0.0,
                s["row_counter"].most_common(1)[0][1] if s["row_counter"] else 0,
                len([c for c in s["row_counter"].values() if c >= 3]),
            ])

        # update state
        s["ce_total"] += 1
        if row is not None:
            s["row_counter"][row] += 1
        if bank is not None:
            s["bank_counter"][bank] += 1
        s["timestamps"].append(t)

        current_day = day

    # flush last day
    if current_day is not None:
        ts = s["timestamps"]
        deltas = (
            pd.Series(ts).diff().dt.total_seconds().dropna()
            if len(ts) > 1 else []
        )

        rows.append([
            current_day,
            sid,
            s["ce_total"],
            len(s["bank_counter"]),
            len(s["row_counter"]),
            deltas.mean() if len(deltas) else 0.0,
            s["row_counter"].most_common(1)[0][1] if s["row_counter"] else 0,
            len([c for c in s["row_counter"].values() if c >= 3]),
        ])

# =======================
# SAVE
# =======================
out = pd.DataFrame(
    rows,
    columns=[
        "day",
        "sid",
        "ce_count_past",
        "unique_banks_past",
        "unique_rows_past",
        "mean_inter_error_time_past",
        "max_row_ce_past",
        "num_hot_rows_past",
    ]
)

out.to_csv(OUT_PATH, index=False)

print("===================================")
print("DONE")
print(f"Saved: {OUT_PATH}")
print(f"Rows: {len(out)}")
print("===================================")
