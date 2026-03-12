import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
MCEL0G_PATH = "/mnt/newdisk/anass/raw_data/mcelog.csv"
TICKETS_PATH = "/mnt/newdisk/anass/raw_data/trouble_tickets.csv"
OUTPUT_PATH = "/mnt/newdisk/anass/daily_scaledfp_subset.csv"
MAX_SERVERS = 200   # OPTION 1: subset size

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

# Drop broken timestamps
mcelog = mcelog.dropna(subset=["error_time"])
tickets = tickets.dropna(subset=["failed_time"])

# Extract day
mcelog["day"] = mcelog["error_time"].dt.date
tickets["day"] = tickets["failed_time"].dt.date

# =========================
# SERVER SUBSET (CRITICAL)
# =========================
print("Selecting server subset...")
selected_sids = (
    mcelog["sid"]
    .dropna()
    .unique()[:MAX_SERVERS]
)
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

print("Starting daily feature extraction...")

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
        else:
            ce_count = 0
            unique_banks = 0
            unique_rows = 0
            mean_delta = 0.0

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
            failed_today
        ])

        # Update history AFTER feature computation
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
