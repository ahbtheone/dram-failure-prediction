import pandas as pd
import numpy as np
from collections import Counter
from datetime import timedelta
import math

# =========================
# CONFIG
# =========================
MCEL0G_PATH = "/mnt/newdisk/anass/raw_data/mcelog.csv"
TICKETS_PATH = "/mnt/newdisk/anass/raw_data/trouble_tickets.csv"
OUTPUT_PATH = "/mnt/newdisk/anass/collector_out_daily_row_burst_subset.csv"

MAX_SERVERS = 1000   # IMPORTANT: match single-machine subset

# =========================
# UTILS
# =========================
def entropy(counter):
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log(c / total + 1e-12) for c in counter.values())

# =========================
# LOAD DATA
# =========================
print("Loading data...")
mcelog = pd.read_csv(MCEL0G_PATH)
tickets = pd.read_csv(TICKETS_PATH)

# =========================
# FIX TIMESTAMPS
# =========================
print("Fixing timestamps...")
mcelog["error_time"] = pd.to_datetime(
    mcelog["error_time"].astype(str).str.replace("^0001", "2018", regex=True),
    errors="coerce"
)
tickets["failed_time"] = pd.to_datetime(
    tickets["failed_time"].astype(str).str.replace("^0001", "2018", regex=True),
    errors="coerce"
)

mcelog = mcelog.dropna(subset=["error_time"])
tickets = tickets.dropna(subset=["failed_time"])

mcelog["day"] = mcelog["error_time"].dt.floor("D")
tickets["day"] = tickets["failed_time"].dt.floor("D")

# =========================
# SERVER SUBSET (FAIR COMPARISON)
# =========================
print("Selecting server subset...")
selected_sids = mcelog["sid"].dropna().unique()[:MAX_SERVERS]
selected_sids = set(selected_sids)

mcelog = mcelog[mcelog["sid"].isin(selected_sids)]
tickets = tickets[tickets["sid"].isin(selected_sids)]

print(f"Using {len(selected_sids)} servers")

# =========================
# FAILURE LOOKUP
# =========================
failure_days = (
    tickets.groupby("sid")["day"]
    .apply(set)
    .to_dict()
)

# =========================
# DAILY FEATURE EXTRACTION
# =========================
rows = []
total_servers = mcelog["sid"].nunique()
server_idx = 0

print("Starting distributed-style daily collection...")

for sid, df_sid in mcelog.groupby("sid"):
    server_idx += 1
    print(f"[{server_idx}/{total_servers}] Processing SID: {sid}")

    df_sid = df_sid.sort_values("error_time")
    history = []

    for day, df_day in df_sid.groupby("day"):
        hist_df = pd.DataFrame(history)

        if not hist_df.empty:
            ce_count = len(hist_df)
            unique_banks = hist_df["bankid"].nunique()
            unique_rows = hist_df["row"].nunique()

            times = hist_df["error_time"].sort_values()
            deltas = times.diff().dt.total_seconds().dropna()
            mean_delta = deltas.mean() if not deltas.empty else 0.0

            # ===== row-aware =====
            row_counts = Counter(hist_df["row"])
            max_row_ce = max(row_counts.values())
            row_ent = entropy(row_counts)
            row_recurrence = sum(1 for v in row_counts.values() if v > 1)

            # ===== burst / subsequent =====
            recent_3d = hist_df[hist_df["error_time"] >= day - timedelta(days=3)]
            recent_7d = hist_df[hist_df["error_time"] >= day - timedelta(days=7)]

            ce_last_3d = len(recent_3d)
            ce_last_7d = len(recent_7d)

            burst_1h = (deltas < 3600).sum()
            burst_1d = (deltas < 86400).sum()
        else:
            ce_count = unique_banks = unique_rows = 0
            mean_delta = 0.0
            max_row_ce = row_ent = row_recurrence = 0
            ce_last_3d = ce_last_7d = 0
            burst_1h = burst_1d = 0

        failed_today = int(
            sid in failure_days and day in failure_days[sid]
        )

        rows.append([
            day,
            sid,
            ce_count,
            unique_banks,
            unique_rows,
            mean_delta,
            max_row_ce,
            row_recurrence,
            row_ent,
            ce_last_3d,
            ce_last_7d,
            burst_1h,
            burst_1d,
            failed_today
        ])

        history.extend(df_day.to_dict("records"))

# =========================
# SAVE OUTPUT
# =========================
daily_df = pd.DataFrame(
    rows,
    columns=[
        "day",
        "sid",
        "ce_count_past",
        "unique_banks_past",
        "unique_rows_past",
        "mean_inter_error_time_past",
        "max_row_ce_past",
        "row_recurrence_past",
        "row_entropy_past",
        "ce_count_last_3d",
        "ce_count_last_7d",
        "burst_count_1h",
        "burst_count_1d",
        "failed"
    ]
)

daily_df.to_csv(OUTPUT_PATH, index=False)

print("====================================")
print("DONE")
print(f"Saved to: {OUTPUT_PATH}")
print(f"Rows: {len(daily_df)}")
print(f"Failures: {daily_df['failed'].sum()}")
print("====================================")
