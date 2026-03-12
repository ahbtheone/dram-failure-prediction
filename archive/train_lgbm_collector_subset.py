import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# =========================
# CONFIG
# =========================
FEATURES_PATH = "/mnt/newdisk/anass/collector_out_daily_row_burst_subset.csv"
LABEL_COL = "failed"
DROP_COLS = ["day", "sid"]

# =========================
# LOAD DATA
# =========================
print("Loading features...")
df = pd.read_csv(FEATURES_PATH)

print(f"Total rows: {len(df)}")
print(f"Failures: {df[LABEL_COL].sum()}")

# =========================
# AUTO-DETECT FEATURES
# =========================
FEATURES = [
    c for c in df.columns
    if c not in DROP_COLS + [LABEL_COL]
]

print("Using features:")
for f in FEATURES:
    print(" -", f)

# =========================
# PREPARE DATA
# =========================
X = df[FEATURES]
y = df[LABEL_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=False
)

# =========================
# TRAIN LIGHTGBM
# =========================
print("Training LightGBM...")

model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

# =========================
# EVALUATE
# =========================
y_pred = model.predict(X_test)

precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n=== OFFLINE COLLECTOR (ROW + BURST, 1K SERVERS) ===")
print(f"Precision: {precision}")
print(f"Recall:    {recall}")
print(f"F1:        {f1}")
