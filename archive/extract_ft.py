import pandas as pd
import os
from glob import glob

merged_dir = "/mnt/newdisk/anass/merged_data"
output_path = "/mnt/newdisk/anass/features_per_month.csv"

feature_rows = []

# Process each merged file
for path in sorted(glob(f"{merged_dir}/*.csv")):
    month_name = os.path.basename(path).replace(".csv", "")
    print(f"Processing {month_name} ...")

    df = pd.read_csv(path)

    # Clean timestamp just for ordering (string only)
    df['error_time'] = df['error_time'].astype(str)

    # Convert to datetime for time gap features
    df['error_dt'] = pd.to_datetime(df['error_time'], errors='coerce')

    # Group by server
    for sid, group in df.groupby('sid'):

        # Count errors
        correctable = (group['error_type'] == 1).sum()
        uncorrectable = (group['error_type'] == 2).sum()

        # Bank/row levels
        num_banks = group['bankid'].nunique()
        num_rows = group['row'].nunique()

        # Time gap stats
        times = group['error_dt'].dropna().sort_values()
        if len(times) > 1:
            diffs = (times.diff().dt.total_seconds().dropna())
            mean_gap = diffs.mean()
            min_gap = diffs.min()
            max_gap = diffs.max()
        else:
            mean_gap = min_gap = max_gap = None

        # Failure label (from merged data)
        failed = int(group['failed'].max())

        # Summarize features
        feature_rows.append({
            "month": month_name,
            "sid": sid,
            "correctable_errors": correctable,
            "uncorrectable_errors": uncorrectable,
            "num_unique_banks": num_banks,
            "num_unique_rows": num_rows,
            "mean_inter_error_time": mean_gap,
            "min_inter_error_time": min_gap,
            "max_inter_error_time": max_gap,
            "failed": failed
        })

# Save final feature matrix
features_df = pd.DataFrame(feature_rows)
features_df.to_csv(output_path, index=False)

print("DONE. Feature matrix saved to:", output_path)
