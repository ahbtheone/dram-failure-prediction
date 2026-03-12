import pandas as pd

FEATURES_PATH = "/mnt/newdisk/anass/daily_scaledfp_distributed.csv"
TICKETS_PATH  = "/mnt/newdisk/anass/raw_data/trouble_tickets.csv"
OUT_PATH      = "/mnt/newdisk/anass/daily_scaledfp_distributed_labeled.csv"

print("Loading features...")
df = pd.read_csv(FEATURES_PATH)

print("Loading tickets...")
tickets = pd.read_csv(TICKETS_PATH)

# fix timestamps
tickets["failed_time"] = pd.to_datetime(
    tickets["failed_time"].astype(str).str.replace("^0001", "2018", regex=True),
    errors="coerce"
)
tickets = tickets.dropna(subset=["failed_time"])
tickets["day"] = tickets["failed_time"].dt.date

# mark failures
tickets["failed"] = 1
tickets = tickets[["sid", "day", "failed"]].drop_duplicates()

print("Joining labels...")
df["day"] = pd.to_datetime(df["day"], errors="coerce").dt.date
df = df.dropna(subset=["day"])
df = df.merge(tickets, on=["sid", "day"], how="left")
df["failed"] = df["failed"].fillna(0).astype(int)

print("Saving labeled dataset...")
df.to_csv(OUT_PATH, index=False)

print("DONE")
print("Rows:", len(df))
print("Failures:", df["failed"].sum())
