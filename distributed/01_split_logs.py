import pandas as pd
import os

INPUT = "/mnt/newdisk/anass/raw_data/mcelog.csv"
OUT_DIR = "/mnt/newdisk/anass/dfp_parts"
NUM_PARTS = 4   # start small: 4 collectors

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading mcelog...")
df = pd.read_csv(INPUT)

# group by sid
sids = df["sid"].unique()
parts = [sids[i::NUM_PARTS] for i in range(NUM_PARTS)]

for i, sid_list in enumerate(parts):
    part_df = df[df["sid"].isin(sid_list)]
    out_path = f"{OUT_DIR}/mcelog_part_{i}.csv"
    part_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}, rows={len(part_df)}")

print("DONE")
