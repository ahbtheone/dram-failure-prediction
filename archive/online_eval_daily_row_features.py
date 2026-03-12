import pandas as pd
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, f1_score

# ===================== PATH =====================
DATA_PATH = "/mnt/newdisk/anass/daily_scaledfp_row_features_1k.csv"

# ===================== LOAD DATA =====================
print("Loading daily features...")
df = pd.read_csv(DATA_PATH)

print("Total rows:", len(df))
print("Failures:", df["failed"].sum())

# Ensure time ordering
df["day"] = pd.to_datetime(df["day"])
df = df.sort_values("day")

FEATURES = [
    "ce_count_past",
    "unique_rows_past",
    "mean_inter_error_time_past",
    "max_row_ce_past",
    "top_row_ratio_past",
    "num_hot_rows_past",
    "row_recurrence_past"
]

# ===================== ONLINE SLIDING WINDOW =====================
y_true, y_pred = [], []

unique_days = df["day"].unique()

for i in range(1, len(unique_days)):
    train_days = unique_days[:i]
    test_day = unique_days[i]

    train_df = df[df["day"].isin(train_days)]
    test_df = df[df["day"] == test_day]

    if train_df["failed"].sum() == 0 or test_df.empty:
        continue

    X_train = train_df[FEATURES]
    y_train = train_df["failed"]

    X_test = test_df[FEATURES]
    y_test = test_df["failed"]

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    y_true.extend(y_test.tolist())
    y_pred.extend(preds.tolist())

# ===================== RESULTS =====================
print("\n=== ONLINE SLIDING-WINDOW EVALUATION ===")
print("Samples evaluated:", len(y_true))
print("Precision:", precision_score(y_true, y_pred, zero_division=0))
print("Recall:", recall_score(y_true, y_pred, zero_division=0))
print("F1:", f1_score(y_true, y_pred, zero_division=0))
