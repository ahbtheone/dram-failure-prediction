
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, f1_score

# ================= CONFIG =================
DATA_PATH = "/mnt/newdisk/anass/final_dataset.csv"

# ================= LOAD =================
print("Loading dataset...")
df = pd.read_csv(DATA_PATH, parse_dates=["day"])
df = df.sort_values("day")

print("Rows:", len(df))
print("Failures:", df["failed"].sum())

# ================= FEATURES =================
features = [
    "ce_count_past_x",
    "unique_banks_past_x",
    "unique_rows_past_x",
    "mean_inter_error_time_past_x",
    "max_row_ce_past",
    "num_hot_rows_past",
    "row_max_count_past",
    "row_entropy_past",
    "ce_last_1d",
    "ce_last_3d",
    "burst_count_1h",
    "burst_count_1d"
]

# ================= ONLINE DAILY LOOP =================
days = sorted(df["day"].unique())

y_true = []
y_pred = []

for i in range(30, len(days)):

    train_days = days[:i]
    test_day = days[i]

    train_df = df[df["day"].isin(train_days)]
    test_df = df[df["day"] == test_day]

    X_train = train_df[features]
    y_train = train_df["failed"]

    X_test = test_df[features]
    y_test = test_df["failed"]

    model = lgb.LGBMClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:,1]
    preds = (probs > 0.9).astype(int)

    y_true.extend(y_test)
    y_pred.extend(preds)

    print("Processed day:", test_day)

# ================= METRICS =================
print("\nONLINE DAILY EVALUATION")

print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1:", f1_score(y_true, y_pred))
