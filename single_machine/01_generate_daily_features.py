import pandas as pd
import numpy as np
import time

# =========================
# CONFIG
# =========================
MCEL0G_PATH = "/mnt/newdisk/anass/raw_data/mcelog.csv"
TICKETS_PATH = "/mnt/newdisk/anass/raw_data/trouble_tickets.csv"
OUT_PATH = "/mnt/newdisk/anass/daily_features.csv"

# =========================
# LOAD DATA
# =========================
print("[STEP 1] Loading mcelog...")
start = time.time()
mcelog = pd.read_csv(MCEL0G_PATH)
print(f"    Done. Rows: {len(mcelog)} ({time.time()-start:.1f}s)")

print("[STEP 2] Loading trouble tickets...")
start = time.time()
tickets = pd.read_csv(TICKETS_PATH)
print(f"    Done. Rows: {len(tickets)} ({time.time()-start:.1f}s)")

# =========================
# FIX TIMESTAMPS
# =========================
print("[STEP 3] Fixing timestamps (0001 → 2018)...")
start = time.time()

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

print(f"    Done. ({time.time()-start:.1f}s)")

# =========================
# EXTRACT DAY
# =========================
print("[STEP 4] Extracting day...")
mcelog["day"] = mcelog["error_time"].dt.floor("D")
tickets["day"] = tickets["failed_time"].dt.floor("D")

# =========================
# FAILURE LOOKUP
# =========================
print("[STEP 5] Building failure lookup...")
start = time.time()

failure_days = (
    tickets.groupby("sid")["day"]
    .apply(set)
    .to_dict()
)

print(f"    Done. ({time.time()-start:.1f}s)")

# =========================
# DAILY FEATURE GENERATION
# =========================
print("[STEP 6] Generating daily features (ScaleDFP-style)...")

rows = []
total_sids = mcelog["sid"].nunique()
processed_sids = 0
start_all = time.time()

for sid, df_sid in mcelog.groupby("sid"):
    processed_sids += 1
    if processed_sids % 50 == 0:
        elapsed = time.time() - start_all
        print(f"    SID {processed_sids}/{total_sids} | elapsed {elapsed/60:.1f} min")

    df_sid = df_sid.sort_values("error_time")

    # Rolling history (PAST ONLY)
    past_times = []
    past_banks = []
    past_rows = []

    for day, df_day in df_sid.groupby("day"):
        if past_times:
            ce_count = len(past_times)
            unique_banks = len(set(past_banks))
            unique_rows = len(set(past_rows))

            times = pd.Series(past_times)
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

        # Update history AFTER emitting features
        past_times.extend(df_day["error_time"].tolist())
        past_banks.extend(df_day["bankid"].tolist())
        past_rows.extend(df_day["row"].tolist())

# =========================
# SAVE OUTPUT
# =========================
print("[STEP 7] Saving output...")
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

daily_df.to_csv(OUT_PATH, index=False)

print("====================================")
print("DONE")
print("Saved:", OUT_PATH)
print("Total rows:", len(daily_df))
print("Failure rows:", daily_df["failed"].sum())
print("Total time: %.1f min" % ((time.time() - start_all) / 60))
print("====================================")
