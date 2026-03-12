import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# ===================== LOAD DATA =====================
DATA_PATH = "/mnt/newdisk/anass/daily_scaledfp_subset.csv"

print("Loading features...")
df = pd.read_csv(DATA_PATH)

# ===================== BASIC SANITY =====================
df = df.dropna()
df["failed"] = df["failed"].astype(int)

# Features only (NO sid, NO date)
FEATURES = [
    "ce_count_past",
    "unique_banks_past",
    "unique_rows_past",
    "mean_inter_error_time_past"
]

X = df[FEATURES]
y = df["failed"]

print("Total rows:", len(df))
print("Failures:", y.sum())

# ===================== TRAIN / TEST SPLIT =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# ===================== LIGHTGBM =====================
model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced",
    random_state=42
)

print("Training LightGBM...")
model.fit(X_train, y_train)

# ===================== EVALUATION =====================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))

auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC:", round(auc, 4))
