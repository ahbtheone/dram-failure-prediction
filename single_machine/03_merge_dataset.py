import pandas as pd

scaled = pd.read_csv("/mnt/newdisk/anass/daily_scaledfp_distributed_labeled.csv")
rowburst = pd.read_csv("/mnt/newdisk/anass/daily_row_burst_full.csv")

merged = scaled.merge(rowburst, on=["sid","day","failed"], how="left")

merged.to_csv("/mnt/newdisk/anass/final_dataset.csv", index=False)

print("done")
