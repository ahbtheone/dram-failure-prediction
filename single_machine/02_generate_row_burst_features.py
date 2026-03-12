import pandas as pd
import numpy as np
from collections import Counter
import math

# =========================
# CONFIG
# =========================
MCELOG_PATH = "/mnt/newdisk/anass/raw_data/mcelog.csv"
TICKETS_PATH = "/mnt/newdisk/anass/raw_data/trouble_tickets.csv"
OUTPUT_PATH = "/mnt/newdisk/anass/daily_row_burst_full.csv"

# None = use ALL servers
MAX_SERVERS = None

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
mcelog = pd.read_csv(MCELOG_PATH)
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

mcelog["day"] = mcelog["error_time"].dt.date
tickets["day"] = tickets["failed_time"].dt.date

# =========================
# SERVER SELECTION
# =========================
print("Selecting servers...")

all_sids = mcelog["sid"].dropna().unique()

print("Total servers in dataset:", len(all_sids))

if MAX_SERVERS is None:
    selected_sids = set(all_sids)
else:
    selected_sids = set(all_sids[:MAX_SERVERS])

print("Servers selected:", len(selected_sids))

mcelog = mcelog[mcelog["sid"].isin(selected_sids)]
tickets = tickets[tickets["sid"].isin(selected_sids)]

# =========================
# FAILURE LOOKUP
# =========================
print("Preparing failure lookup...")

failure_days = (
    tickets.groupby("sid")["day"]
    .apply(set)
    .to_dict()
)

# =========================
# FEATURE EXTRACTION
# =========================
print("Starting daily row-aware + burst feature extraction...")

rows = []

total_servers = mcelog["sid"].nunique()
server_idx = 0

for sid, df_sid in mcelog.groupby("sid"):

    server_idx += 1

    if server_idx % 100 == 0:
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

            # row-aware
            row_counts = Counter(hist_df["row"])
            row_max = max(row_counts.values())
            row_ent = entropy(row_counts)

            # burst features
            now = pd.to_datetime(day)

            hist_df["dt"] = hist_df["error_time"]

            ce_last_1d = (now - hist_df["dt"] <= pd.Timedelta(days=1)).sum()
            ce_last_3d = (now - hist_df["dt"] <= pd.Timedelta(days=3)).sum()

            burst_1h = (deltas < 3600).sum()
            burst_1d = (deltas < 86400).sum()

        else:

            ce_count = 0
            unique_banks = 0
            unique_rows = 0
            mean_delta = 0.0
            row_max = 0.0
            row_ent = 0.0
            ce_last_1d = 0
            ce_last_3d = 0
            burst_1h = 0
            burst_1d = 0

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
            row_max,
            row_ent,
            ce_last_1d,
            ce_last_3d,
            burst_1h,
            burst_1d,
            failed_today
        ])

        history.extend(df_day.to_dict("records"))

# =========================
# SAVE OUTPUT
# =========================
print("Saving dataset...")

daily_df = pd.DataFrame(
    rows,
    columns=[
        "day",
        "sid",
        "ce_count_past",
        "unique_banks_past",
        "unique_rows_past",
        "mean_inter_error_time_past",
        "row_max_count_past",
        "row_entropy_past",
        "ce_last_1d",
        "ce_last_3d",
        "burst_count_1h",
        "burst_count_1d",
        "failed"
    ]
)

daily_df.to_csv(OUTPUT_PATH, index=False)

print("====================================")
print("DONE")
print("Saved to:", OUTPUT_PATH)
print("Rows:", len(daily_df))
print("Failures:", daily_df["failed"].sum())
print("====================================")
