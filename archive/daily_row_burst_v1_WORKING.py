import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# =========================
# LOAD DATA
# =========================
PATH = "/mnt/newdisk/anass/daily_scaledfp_row_burst_1k.csv"
print("Loading features...")
df = pd.read_csv(PATH)

print("Total rows:", len(df))
print("Failures:", df["failed"].sum())

# =========================
# FEATURES
# =========================
FEATURES = [
    "ce_count_past",
    "unique_rows_past",
    "mean_inter_error_time_past",
    "max_row_ce_past",
    "top_row_ratio_past",
    "num_hot_rows_past",
    "row_recurrence_past",
]

X = df[FEATURES]
y = df["failed"]

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=False
)

# =========================
# TRAIN LIGHTGBM
# =========================
print("Training LightGBM...")
model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test)

precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("=== OFFLINE DAILY (ROW + BURST FEATURES, 1K SERVERS) ===")
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
