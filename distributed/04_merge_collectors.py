import pandas as pd

OUT_DIR = "/mnt/newdisk/anass/dfp_out"
FINAL_OUT = "/mnt/newdisk/anass/daily_scaledfp_distributed.csv"

files = [
    f"{OUT_DIR}/out_part_0.csv",
    f"{OUT_DIR}/out_part_1.csv",
    f"{OUT_DIR}/out_part_2.csv",
    f"{OUT_DIR}/out_part_3.csv",
]

print("Merging files:")
for f in files:
    print(" ", f)

dfs = [pd.read_csv(f, low_memory=False) for f in files]
final_df = pd.concat(dfs, ignore_index=True)

# sanity checks
assert "sid" in final_df.columns


final_df.to_csv(FINAL_OUT, index=False)

print("=================================")
print("DONE")
print("Saved to:", FINAL_OUT)
print("Total rows:", len(final_df))
print("Unique servers:", final_df["sid"].nunique())
print("=================================")
