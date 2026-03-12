import pandas as pd
import os

input_file = "/mnt/newdisk/anass/raw_data/mcelog_clean.csv"
output_dir = "/mnt/newdisk/anass/monthly_data"

os.makedirs(output_dir, exist_ok=True)

print("Loading CSV (this may take a bit)...")
df = pd.read_csv(input_file)

print("Preparing year_month from error_time (string slicing)...")
# Ensure string and strip any null bytes / spaces
s = df["error_time"].astype(str).str.replace("\x00", "", regex=False).str.strip()

# Take first 7 chars: 'YYYY-MM'
df["year_month"] = s.str.slice(0, 7)

# Drop rows where this somehow failed
df = df[df["year_month"].notna()]

print("Writing monthly files...")
for month, group in df.groupby("year_month"):
    outpath = os.path.join(output_dir, f"{month}.csv")
    group.to_csv(outpath, index=False)

print("Done. Check:", output_dir)
