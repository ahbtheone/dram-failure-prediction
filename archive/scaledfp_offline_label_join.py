import pandas as pd
import glob
from pathlib import Path

FEATURE_DIR = "/mnt/newdisk/anass/scaledfp_offline_features"
FAILURE_FILE = "/mnt/newdisk/anass/raw_data/trouble_tickets.csv"
OUT_FILE = "/mnt/newdisk/anass/scaledfp_offline_labeled_features.csv"

# Load failures
failures = pd.read_csv(FAILURE_FILE)
failures["failed_time"] = pd.to_datetime(failures["failed_time"], errors="coerce")
failures["year_month"] = failures["failed_time"].dt.strftime("%Y-%m")

# Load all offline feature files
dfs = []
for f in sorted(glob.glob(f"{FEATURE_DIR}/offline_features_*.csv")):
    df = pd.read_csv(f)
    dfs.append(df)

features = pd.concat(dfs, ignore_index=True)

# Convert logical month to real next month (0001-01 -> 2020-02 etc.)
def logical_to_real_next_month(m):
    y, mo = map(int, m.split("-"))
    real_year = 2020
    real_month = mo + 1
    if real_month > 12:
        real_month = 1
        real_year += 1
    return f"{real_year:04d}-{real_month:02d}"

features["next_month"] = features["month"].apply(logical_to_real_next_month)

# Label join
features["failed"] = features.apply(
    lambda r: int(
        ((failures["sid"] == r["sid"]) &
         (failures["year_month"] == r["next_month"])).any()
    ),
    axis=1
)

features.drop(columns=["next_month"], inplace=True)
features.to_csv(OUT_FILE, index=False)

print("DONE. Labeled dataset saved to:", OUT_FILE)
print(features["failed"].value_counts())
