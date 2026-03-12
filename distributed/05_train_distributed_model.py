import pandas as pd
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

DATA_PATH = "/mnt/newdisk/anass/daily_scaledfp_distributed_labeled.csv"

print("Loading distributed features...")
df = pd.read_csv(DATA_PATH)

FEATURES = [
    "ce_count_past",
    "unique_banks_past",
    "unique_rows_past",
    "mean_inter_error_time_past",
    "max_row_ce_past",
    "num_hot_rows_past",
]

TARGET = "failed"

X = df[FEATURES]
y = df[TARGET]

print("Total rows:", len(df))
print("Failures:", y.sum())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=True, random_state=42
)

print("Training LightGBM (OFFLINE)...")

model = lgb.LGBMClassifier(
    n_estimators=200,
    num_leaves=64,
    learning_rate=0.05,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n=== OFFLINE DISTRIBUTED DAILY (ROW + BURST) ===")
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
