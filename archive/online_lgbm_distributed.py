import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, f1_score

# =========================
# CONFIG
# =========================
DATA_PATH = "/mnt/newdisk/anass/daily_scaledfp_distributed_labeled.csv"

FEATURES = [
    "ce_count_past",
    "unique_banks_past",
    "unique_rows_past",
    "mean_inter_error_time_past",
    "max_row_ce_past",
    "num_hot_rows_past",
]

LABEL = "failed"
MIN_TRAIN_SAMPLES = 10   # relaxed but realistic

# =========================
# LOAD DATA
# =========================
print("Loading distributed daily features...")
df = pd.read_csv(DATA_PATH)

print("Total rows:", len(df))
print("Failures:", df[LABEL].sum())

df["day"] = pd.to_datetime(df["day"])
df = df.sort_values(["sid", "day"])

# =========================
# METRIC STORAGE
# =========================
y_true = []
y_pred = []

skipped_no_history = 0
evaluated = 0

# =========================
# ONLINE SLIDING-WINDOW
# =========================
for sid, df_sid in df.groupby("sid"):
    df_sid = df_sid.sort_values("day").reset_index(drop=True)

    for i in range(1, len(df_sid)):
        train = df_sid.iloc[:i]
        test = df_sid.iloc[i]

        if len(train) < MIN_TRAIN_SAMPLES:
            skipped_no_history += 1
            continue

        X_train = train[FEATURES]
        y_train = train[LABEL]

        # Skip degenerate windows
        if y_train.nunique() < 2:
            skipped_no_history += 1
            continue

        model = lgb.LGBMClassifier(
            n_estimators=50,
            max_depth=5,
            n_jobs=4
        )

        model.fit(X_train, y_train)

        y_hat = model.predict(test[FEATURES].values.reshape(1, -1))[0]

        y_true.append(test[LABEL])
        y_pred.append(y_hat)
        evaluated += 1

# =========================
# RESULTS
# =========================
print("\n=== ONLINE DISTRIBUTED SLIDING-WINDOW EVALUATION ===")
print("Samples evaluated:", evaluated)
print("Skipped (no history / single class):", skipped_no_history)

if evaluated > 0:
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, zero_division=0))
    print("F1:", f1_score(y_true, y_pred, zero_division=0))
else:
    print("No valid samples evaluated.")
