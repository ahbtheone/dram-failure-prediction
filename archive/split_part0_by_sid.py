import pandas as pd
import numpy as np
import os
import sys

# =====================
# CONFIG
# =====================
INPUT_PATH = "/mnt/newdisk/anass/dfp_parts/mcelog_part_0.csv"
OUT_DIR = "/mnt/newdisk/anass/dfp_parts/part0_sub"
N_PARTS = 4   # <-- change this (e.g. 2, 4, 8)

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading mcelog_part_0...")
df = pd.read_csv(INPUT_PATH)

print("Extracting unique servers...")
sids = df["sid"].dropna().unique()
print(f"Total servers: {len(sids)}")

print(f"Splitting into {N_PARTS} partitions...")
sid_splits = np.array_split(sids, N_PARTS)

for i, sid_subset in enumerate(sid_splits):
    sid_set = set(sid_subset)
    part_df = df[df["sid"].isin(sid_set)]

    out_path = os.path.join(OUT_DIR, f"mcelog_part_0_sub_{i}.csv")
    part_df.to_csv(out_path, index=False)

    print(f"Saved {out_path} | rows={len(part_df)} | servers={len(sid_subset)}")

print("DONE")
