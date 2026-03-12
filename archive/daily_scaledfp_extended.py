import pandas as pd
import numpy as np
import os

# ===================== CONFIG =====================
MCELOG_PATH = "/mnt/newdisk/anass/raw_data/mcelog.csv"
FAILURE_PATH = "/mnt/newdisk/anass/raw_data/trouble_tickets.csv"
OUT_PATH = "/mnt/newdisk/anass/daily_scaledfp_row_features_1k.csv"
MAX_SERVERS = 1000

# ===================== LOAD =====================
print("Loading data...")
mcelog = pd.read_csv(MCELOG_PATH)
tickets = pd.read_csv(FAILURE_PATH)

# ===================== FIX TIMESTAMPS =====================
mcelog["error_time"] = pd.to_datetime(
    mcelog["error_time"].str.replace("^0001", "2018", regex=True),
    errors="coerce"
)
mcelog = mcelog.dropna(subset=["error_time"])
mcelog["day"] = mcelog["error_time"].dt.date

tickets["failed_time"] = pd.to_datetime(
    tickets["failed_time"].str.replace("^0001", "2018", regex=True),
    errors="coerce"
)
tickets["day"] = tickets["failed_time"].dt.date

# ===================== FAILURE LOOKUP =====================
failure_days = tickets.groupby("sid")["day"].apply(set).to_dict()

# ===================== SELECT 1K SERVERS =====================
sids = mcelog["sid"].unique()[:MAX_SERVERS]
mcelog = mcelog[mcelog["sid"].isin(sids)]

print(f"Processing {len(sids)} servers")

# ===================== FEATURE EXTRACTION =====================
rows = []

for sid, df_sid in mcelog.groupby("sid"):
    df_sid = df_sid.sort_values("error_time")
    history = []

    for day, df_day in df_sid.groupby("day"):
        hist = pd.DataFrame(history)

        if hist.empty:
            ce_count = 0
            unique_rows = 0
            mean_delta = 0
            max_row_ce = 0
            top_row_ratio = 0
            hot_rows = 0
            row_recurrence = 0
        else:
            ce_count = len(hist)
            unique_rows = hist["row"].nunique()

            deltas = hist["error_time"].sort_values().diff().dt.total_seconds().dropna()
            mean_delta = deltas.mean() if not deltas.empty else 0

            row_counts = hist["row"].value_counts()
            max_row_ce = row_counts.max()
            top_row_ratio = max_row_ce / ce_count
            hot_rows = (row_counts >= 2).sum()

            row_seq = hist["row"].values
            row_recurrence = np.mean(row_seq[1:] == row_seq[:-1]) if len(row_seq) > 1 else 0

        failed = int(day in failure_days.get(sid, set()))

        rows.append([
            day, sid,
            ce_count,
            unique_rows,
            mean_delta,
            max_row_ce,
            top_row_ratio,
            hot_rows,
            row_recurrence,
            failed
        ])

        history.extend(df_day.to_dict("records"))

# ===================== SAVE =====================
df = pd.DataFrame(rows, columns=[
    "day", "sid",
    "ce_count_past",
    "unique_rows_past",
    "mean_inter_error_time_past",
    "max_row_ce_past",
    "top_row_ratio_past",
    "num_hot_rows_past",
    "row_recurrence_past",
    "failed"
])

df.to_csv(OUT_PATH, index=False)

print("DONE")
print("Saved:", OUT_PATH)
print("Rows:", len(df))
print("Failures:", df["failed"].sum())
