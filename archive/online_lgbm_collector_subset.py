import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, f1_score

# =========================
# CONFIG
# =========================
DATA_PATH = "/mnt/newdisk/anass/collector_out_daily_row_burst_subset.csv"

FEATURES = [
    "ce_count_past",
    "unique_banks_past",
    "unique_rows_past",
    "mean_inter_error_time_past",
    "max_row_ce_past",
    "row_recurrence_past",
    "row_entropy_past",
    "ce_count_last_3d",
    "ce_count_last_7d",
    "burst_count_1h",
    "burst_count_1d",
]

LABEL = "failed"

# =========================
# LOAD DATA
# =========================
print("Loading collector features...")
df = pd.read_csv(DATA_PATH)

print("Total rows:", len(df))
print("Failures:", df[LABEL].sum())

df["day"] = pd.to_datetime(df["day"])
df = df.sort_values(["sid", "day"])

# =========================
# ONLINE SLIDING WINDOW
# =========================
y_true = []
y_pred = []

skipped = 0

unique_days = df["day"].unique()

print("\n=== ONLINE COLLECTOR SLIDING-WINDOW EVALUATION ===")

for current_day in unique_days[1:]:
    train_df = df[df["day"] < current_day]
    test_df  = df[df["day"] == current_day]

    if train_df[LABEL].nunique() < 2:
        skipped += len(test_df)
        continue

    X_train = train_df[FEATURES]
    y_train = train_df[LABEL]

    X_test = test_df[FEATURES]
    y_test = test_df[LABEL]

    model = lgb.LGBMClassifier(
        n_estimators=100,
        num_leaves=31,
        learning_rate=0.05,
        objective="binary",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    y_true.extend(y_test.tolist())
    y_pred.extend(preds.tolist())

# =========================
# RESULTS
# =========================
print("\nSamples evaluated:", len(y_true))
print("Skipped (no history / single class):", skipped)

if len(y_true) > 0:
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, zero_division=0))
    print("F1:", f1_score(y_true, y_pred, zero_division=0))
else:
    print("No valid samples evaluated.")
