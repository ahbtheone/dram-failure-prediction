import os
import pandas as pd
import numpy as np

# ==============================
# Config
# ==============================
MONTHLY_DATA_DIR = "/mnt/newdisk/anass/monthly_data_tsfix"
OUTPUT_DIR = "/mnt/newdisk/anass/scaledfp_offline_features"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# Helper
# ==============================
def compute_inter_error_times(df):
    """
    Compute inter-error time statistics (in seconds)
    """
    if len(df) < 2:
        return np.nan, np.nan, np.nan

    times = pd.to_datetime(df["error_time"])
    deltas = times.sort_values().diff().dt.total_seconds().dropna()

    if deltas.empty:
        return np.nan, np.nan, np.nan

    return deltas.mean(), deltas.min(), deltas.max()


# ==============================
# Main loop (ScaleDFP-style offline)
# ==============================
for fname in sorted(os.listdir(MONTHLY_DATA_DIR)):
    if not fname.endswith(".csv"):
        continue

    month = fname.replace(".csv", "")
    print(f"Processing {month}")

    path = os.path.join(MONTHLY_DATA_DIR, fname)
    df = pd.read_csv(path)

    # Ensure timestamp
    df["error_time"] = pd.to_datetime(df["error_time"], errors="coerce")
    df = df.dropna(subset=["error_time"])

    outputs = []

    # Group by server (near-data preprocessing unit)
    for sid, g in df.groupby("sid"):
        ce_count = len(g)

        num_unique_banks = g["bankid"].nunique()
        num_unique_rows = g["row"].nunique()

        mean_dt, min_dt, max_dt = compute_inter_error_times(g)

        outputs.append({
            "month": month,
            "sid": sid,
            "ce_count": ce_count,
            "num_unique_banks": num_unique_banks,
            "num_unique_rows": num_unique_rows,
            "mean_inter_error_time": mean_dt,
            "min_inter_error_time": min_dt,
            "max_inter_error_time": max_dt
        })

    out_df = pd.DataFrame(outputs)

    out_path = os.path.join(
        OUTPUT_DIR, f"offline_features_{month}.csv"
    )
    out_df.to_csv(out_path, index=False)

print("DONE. ScaleDFP-style offline features generated.")
