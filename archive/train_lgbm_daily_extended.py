import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

PATH = "/mnt/newdisk/anass/daily_scaledfp_extended.csv"

print("Loading features...")
df = pd.read_csv(PATH)

FEATURES = [
    "ce_count_past",
    "unique_banks_past",
    "unique_rows_past",
    "mean_inter_error_time_past",
    "ce_count_last_3d",
    "ce_count_last_7d"
]

X = df[FEATURES]
y = df["failed"]

print("Total rows:", len(df))
print("Failures:", y.sum())

# temporal split (offline, but realistic)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=False
)

print("Training LightGBM...")
model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=200,
    num_leaves=31,
    learning_rate=0.05,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

precision = precision_score(y_test, preds, zero_division=0)
recall = recall_score(y_test, preds, zero_division=0)
f1 = f1_score(y_test, preds, zero_division=0)

print("=== OFFLINE DAILY (EXTENDED FEATURES) ===")
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
