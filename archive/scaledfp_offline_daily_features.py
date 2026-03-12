import os
import pandas as pd

# ===================== PATHS =====================
MCELOG_PATH = "/mnt/newdisk/anass/raw_data/mcelog.csv"
FAILURE_PATH = "/mnt/newdisk/anass/raw_data/trouble_tickets.csv"
OUT_DIR = "/mnt/newdisk/anass/scaledfp_offline_daily_features"

os.makedirs(OUT_DIR, exist_ok=True)

# ===================== LOAD DATA =====================
print("Loading mcelog...")
mcelog = pd.read_csv(MCELOG_PATH)

print("Loading failures...")
failures = pd.read_csv(FAILURE_PATH)

# ===================== FIX TIMESTAMPS =====================
mcelog["error_time"] = pd.to_datetime(mcelog["error_time"], errors="coerce")
mcelog = mcelog.dropna(subset=["error_time"])

# daily window
mcelog["date"] = mcelog["error_time"].dt.date
mcelog["year_month"] = mcelog["error_time"].dt.strftime("%Y-%m")

# ===================== FAILURE LABEL (ROBUST) =====================
# your file DOES NOT have failure_time → stop assuming it
# we only need sid + label

if "failed" not in failures.columns:
    failures["failed"] = 1

failures = failures[["sid", "failed"]].drop_duplicates()

# ===================== PROCESS PER MONTH =====================
for ym in sorted(mcelog["year_month"].unique()):
    print(f"Processing {ym}")

    df = mcelog[mcelog["year_month"] == ym]

    # CE only
    df = df[df["error_type"] == 1]

    if df.empty:
        continue

    # DAILY aggregation
    agg = df.groupby(["sid", "date"]).agg(
        ce_count=("error_type", "count"),
        unique_banks=("bankid", "nunique"),
        unique_rows=("row", "nunique"),
    ).reset_index()

    # attach failure label
    agg = agg.merge(failures, on="sid", how="left")
    agg["failed"] = agg["failed"].fillna(0).astype(int)

    out_path = os.path.join(OUT_DIR, f"daily_features_{ym}.csv")
    agg.to_csv(out_path, index=False)

print("DONE. Daily ScaleDFP-style offline features generated.")  
