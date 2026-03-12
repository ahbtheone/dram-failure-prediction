import pandas as pd
import os
from glob import glob

merged_dir = "/mnt/newdisk/anass/merged_data"
output_dir = "/mnt/newdisk/anass/ce_baseline"
os.makedirs(output_dir, exist_ok=True)

monthly_summary = []

for path in sorted(glob(f"{merged_dir}/*.csv")):
    month = os.path.basename(path).replace(".csv", "")
    df = pd.read_csv(path)

    # Correctable Errors only
    ce_df = df[df["error_type"] == 1]

    # CE count per server
    ce_per_server = ce_df.groupby("sid").size().reset_index(name="ce_count")
    ce_per_server["month"] = month

    # Failure label per server (already merged)
    failures = df.groupby("sid")["failed"].max().reset_index()

    merged = ce_per_server.merge(failures, on="sid", how="left")
    merged.to_csv(f"{output_dir}/{month}_ce_servers.csv", index=False)

    # Monthly summary
    total_ce = len(ce_df)
    servers_with_ce = ce_per_server["sid"].nunique()
    total_servers = df["sid"].nunique()
    failures_this_month = failures["failed"].sum()

    monthly_summary.append({
        "month": month,
        "total_ce": total_ce,
        "servers_with_ce": servers_with_ce,
        "total_servers": total_servers,
        "failures": failures_this_month
    })

# Save monthly summary
pd.DataFrame(monthly_summary).to_csv(f"{output_dir}/ce_monthly_summary.csv", index=False)

print("CE baseline analysis completed.")
print("Output directory:", output_dir)
