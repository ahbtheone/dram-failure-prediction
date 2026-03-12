import pandas as pd
import os
from glob import glob

monthly_dir = "/mnt/newdisk/anass/monthly_data"
inventory_path = "/mnt/newdisk/anass/raw_data/inventory.csv"
tickets_path = "/mnt/newdisk/anass/raw_data/trouble_tickets.csv"
output_dir = "/mnt/newdisk/anass/merged_data"

os.makedirs(output_dir, exist_ok=True)

# Load static datasets
inv = pd.read_csv(inventory_path)
tickets = pd.read_csv(tickets_path)

# Extract year-month for failure events
tickets['year_month'] = tickets['failed_time'].astype(str).str.slice(0, 7)

# List all monthly mcelog files
month_files = sorted(glob(f"{monthly_dir}/*.csv"))

for path in month_files:
    month_name = os.path.basename(path).replace(".csv", "")
    print(f"Processing {month_name} ...")

    # Load monthly mcelog data
    df = pd.read_csv(path)

    # Merge with inventory on sid
    df = df.merge(inv, on='sid', how='left')

    # Add failure label for that month
    failed_sids = set(tickets[tickets['year_month'] == month_name]['sid'])
    df['failed'] = df['sid'].isin(failed_sids).astype(int)

    # Output merged file
    outpath = os.path.join(output_dir, f"{month_name}.csv")
    df.to_csv(outpath, index=False)

print("DONE. Merged files stored in:", output_dir)
