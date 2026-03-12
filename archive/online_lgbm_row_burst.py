import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, f1_score

# =========================
# CONFIG
# =========================
DATA_PATH = "/mnt/newdisk/anass/daily_scaledfp_row_burst_1k.csv"

FEATURES = [
    "ce_count_past",
    "unique_rows_past",
    "mean_inter_error_time_past",
    "max_row_ce_past",
    "top_row_ratio_past",
    "num_hot_rows_past",
    "row_recurrence_past",
]

LABEL = "failed"

# =========================
# LOAD DATA
# =========================
print("Loading daily features...")
df = pd.read_csv(DATA_PATH)

# ensure ordering
df["day"] = pd.to_datetime(df["day"])
df = df.sort_values(["sid", "day"]).reset_index(drop=True)

print("Total rows:", len(df))
print("Failures:", df[LABEL].sum())

# =========================
# ONLINE SLIDING WINDOW
# =========================
y_true = []
y_pred = []

unique_days = sorted(df["day"].unique())

for i in range(1, len(unique_days)):
    train_days = unique_days[:i]
    test_day = unique_days[i]

    train_df = df[df["day"].isin(train_days)]
    test_df = df[df["day"] == test_day]

    # skip if no positives yet
    if train_df[LABEL].sum() == 0:
        continue

    X_train = train_df[FEATURES]
    y_train = train_df[LABEL]

    X_test = test_df[FEATURES]
    y_test = test_df[LABEL]

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    y_true.extend(y_test.tolist())
    y_pred.extend(preds.tolist())

# =========================
# METRICS
# =========================
print("\n=== ONLINE SLIDING-WINDOW EVALUATION ===")
print("Samples evaluated:", len(y_true))

if len(set(y_true)) > 1:
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1:", f1_score(y_true, y_pred))
else:
    print("Not enough positive samples for metrics.")
